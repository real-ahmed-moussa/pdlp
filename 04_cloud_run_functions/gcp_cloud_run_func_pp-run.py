import os, re, logging
import functions_framework
from cloudevents.http import CloudEvent
from urllib.parse import unquote
from google.cloud import aiplatform, storage


# -------- Config via env --------
# ============================================================================
# USER CONFIGURATION — REPLACE THE PLACEHOLDERS BELOW WITH YOUR OWN VALUES
# Never commit real project IDs, service-account emails, or resource IDs.
# ============================================================================
PROJECT_ID  = "your-gcp-project-id"                                         # TODO: your Google Cloud project ID
LOCATION    = "us-central1"                                                 # TODO: your Vertex AI region
PIPELINE_TEMPLATE_URI = "gs://your-bucket-name/pipeline.json"              # TODO: path to compiled pipeline.json in YOUR bucket
PIPELINE_ROOT = "gs://your-bucket-name/pipeline_root"                      # TODO: pipeline root directory in YOUR bucket
PIPELINE_SA   = "your-project-number-compute@developer.gserviceaccount.com" # TODO: runtime service account for the pipeline
DATA_PREFIX   = "data/span-"                                                # folder under bucket
READY_SUFFIX  = "_READY"                                                    # marker filename


SPAN_RE = re.compile(rf"^{re.escape(DATA_PREFIX)}(\d+)/(?:.*\/)?{re.escape(READY_SUFFIX)}$")
storage_client = storage.Client()

# Helper Functions
def _extract_bucket_object(data: dict):
    # 1. Direct GCS Payload
    b, n = data.get("bucket"), data.get("name")
    if b and n:
        return b, n
    # 2. AuditLog (Eventarc)
    proto = data.get("protoPayload") or {}
    rname = proto.get("resourceName", "")
    if "/buckets/" in rname and "/objects/" in rname:
        bucket = rname.split("/buckets/")[1].split("/")[0]
        name = unquote(rname.split("/objects/")[1])
        return bucket, name
    return None, None

def _span_from_object(objname: str):
    m = SPAN_RE.match(objname)
    return int(m.group(1)) if m else None

def _validate_span_contents(bucket: str, span: int):
    """Ensure train/val/test have at least one CSV each; adjust if you only train on train+val."""
    base = f"{DATA_PREFIX}{span}/"
    need = ["train/", "val/", "test/"]
    for split in need:
        found = False
        for _ in storage_client.list_blobs(bucket, prefix=base+split):
            if _.name.endswith(".csv"):
                found = True
                break
        if not found:
            logging.warning("No CSVs under gs://%s/%s%s", bucket, base, split)
            return False
    return True

def build_window(bucket="your-bucket-name",  # TODO: default bucket; callers normally pass the trigger bucket
                  root="data", k=3, window_dir="_window_train", require_complete=True, verbose=True):
    client = storage.Client()
    b = client.bucket(bucket)

    it = client.list_blobs(bucket, prefix=f"{root}/", delimiter="/")
    for _ in it:
        pass
    span_dirs = [p for p in it.prefixes if re.search(r"(?:^|/)span-0*\d+/$", p)]
    if not span_dirs:
        raise RuntimeError(f"No span-* folders under gs://{bucket}/{root}")

    def span_num(prefix: str) -> int:
        m = re.search(r"(?:^|/)span-0*(\d+)/$", prefix); return int(m.group(1)) if m else -1

    def has_all_splits(span_prefix: str) -> bool:
        for split in ("train/", "val/", "test/"):
            it2 = client.list_blobs(bucket, prefix=f"{span_prefix}{split}", max_results=1)
            if next(iter(it2), None) is None:
                return False
        return True

    if require_complete:
        span_dirs = [sd for sd in span_dirs if has_all_splits(sd)]
        if not span_dirs:
            raise RuntimeError("All spans are incomplete (missing train/val/test).")

    span_dirs.sort(key=span_num)
    chosen = span_dirs[-k:]
    chosen_ids = [span_num(sd) for sd in chosen]

    window_prefix = f"{root}/{window_dir}/"
    # Clear old window
    for blob in client.list_blobs(bucket, prefix=window_prefix):
        blob.delete()

    # Copy files into window
    for sd in chosen:
        sid = span_num(sd)
        for split in ("train", "val", "test"):
            src_prefix = f"{sd}{split}/"
            for blob in client.list_blobs(bucket, prefix=src_prefix):
                base = blob.name.rsplit("/", 1)[-1]
                dst = f"{window_prefix}{split}/span-{sid}-{base}"
                b.copy_blob(blob, b, dst)

    if verbose:
        print(f"Built window at gs://{bucket}/{window_prefix} from spans: {chosen_ids}")
    return f"gs://{bucket}/{window_prefix}"


@functions_framework.cloud_event
def trigger_pipeline(event: CloudEvent):
    data = event.data or {}
    bucket, name = _extract_bucket_object(data)
    if not bucket or not name:
        return ("ignored", 204)

    # Only react to marker files like: data/span-17/_READY (or data/span-17/whatever/_READY)
    if not name.startswith(DATA_PREFIX) or not name.endswith(READY_SUFFIX):
        return ("ignored", 204)

    span = _span_from_object(name)
    if span is None:
        logging.info("No span parsed from object: %s", name)
        return ("ignored", 204)

    # Per-span lock to avoid duplicate submits
    bkt = storage_client.bucket(bucket)
    lock_blob = bkt.blob(f"{DATA_PREFIX}{span}/.pipeline.lock")
    try:
        lock_blob.upload_from_string("lock", if_generation_match=0)  # only if not exists
    except Exception as e:
        logging.info("Lock exists; another invocation already handled span %s (%s)", span, e)
        return ("ok", 200)

    # Optional: assert required CSVs exist
    if not _validate_span_contents(bucket, span):
        logging.error("Span %s incomplete; aborting submit.", span)
        return ("failed", 400)

    # Rebuild both windows that ExampleGen reads
    build_window(bucket=bucket, root="data", k=3, window_dir="_window_train", require_complete=True)
    build_window(bucket=bucket, root="data", k=1, window_dir="_window_eval",  require_complete=True)

    # Submit Vertex Pipeline with --span
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=PIPELINE_ROOT)
    job = aiplatform.PipelineJob(
        display_name=f"ppln-span-{span}",
        template_path=PIPELINE_TEMPLATE_URI,
        pipeline_root=PIPELINE_ROOT,
    )
    job.submit(service_account=PIPELINE_SA)
    logging.info("Submitted pipeline for span=%s", span)
    return ("ok", 200)