import logging
import functions_framework
from cloudevents.http import CloudEvent
from urllib.parse import unquote
from google.cloud import aiplatform, storage
from google.cloud import aiplatform_v1


# ---- Env (safe reads; won't crash container if unset) ----
# ============================================================================
# USER CONFIGURATION — REPLACE THE PLACEHOLDERS BELOW WITH YOUR OWN VALUES
# Never commit real project IDs, service-account emails, or resource IDs.
# ============================================================================
PROJECT_ID = "your-gcp-project-id"     # TODO: your Google Cloud project ID
LOCATION = "us-central1"               # TODO: region where your endpoint and model live
ENDPOINT_ID = "your-endpoint-id"       # TODO: numeric Vertex AI Endpoint ID (Vertex AI > Online prediction > Endpoints)
PRED_IMAGE = "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-15:latest"

MACHINE_TYPE = "n1-standard-2"
TRAFFIC = int("100")
UNDEPLOY = "true"
PARENT_MODEL_ID = "your-parent-model-id"   # TODO: numeric Model Registry ID of the parent model (Vertex AI > Model Registry)


# ---- Clients ----
storage_client = storage.Client()
MODEL_CLIENT = aiplatform_v1.ModelServiceClient(client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


# ---- Trigger policy ----
PREFIX="serving_model/"
SUFFIXES=("saved_model.pb",)

def _endpoint_name(project, location, endpoint_id):
    return endpoint_id if endpoint_id and endpoint_id.startswith("projects/") \
           else f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"

def _should_process(obj_name: str) -> bool:
    return obj_name.startswith(PREFIX) and obj_name.endswith(SUFFIXES)

def _extract_bucket_object(data: dict):
    # Direct GCS event
    b, n = data.get("bucket"), data.get("name")
    if b and n: return b, n
    # AuditLog-shaped fallback
    proto = (data.get("protoPayload") or {})
    r = proto.get("resourceName", "")
    if "/buckets/" in r and "/objects/" in r:
        bucket = r.split("/buckets/")[1].split("/")[0]
        name = unquote(r.split("/objects/")[1])
        return bucket, name
    labels = (data.get("resource") or {}).get("labels", {})
    bucket = labels.get("bucket_name")
    name = labels.get("object_id") or labels.get("object_name")
    return (bucket, name) if bucket and name else (None, None)

def _model_prefix(obj_name: str) -> str:    # serving_model/<version>/saved_model.pb
    return obj_name.rsplit("/", 1)[0]       # parent of saved_model.pb

def _savedmodel_complete(bucket: str, obj_name: str) -> bool:
    """Ensure saved_model.pb + variables/* exist (prevents half-written deploys)."""
    prefix = _model_prefix(obj_name) + "/"
    have_pb = have_idx = have_data = False
    for b in storage_client.list_blobs(bucket, prefix=prefix):
        rel = b.name[len(prefix):]
        if rel == "saved_model.pb": have_pb = True
        elif rel == "variables/variables.index": have_idx = True
        elif rel.startswith("variables/variables.data-"): have_data = True
        if have_pb and have_idx and have_data:
            return True
    return False

def _version_exists_under_parent(version_tag: str) -> bool:
    if not PARENT_MODEL_ID:
        return False
    model_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{PARENT_MODEL_ID}"
    for m in MODEL_CLIENT.list_model_versions(name=model_name):
        if dict(m.labels or {}).get("version") == version_tag:
            return True
    return False


@functions_framework.cloud_event
def deploy_to_vertex(event: CloudEvent):
    # Sanity env check (never crash container)
    if not PROJECT_ID or not ENDPOINT_ID:
        logging.error("Missing env: PROJECT_ID or ENDPOINT_ID")
        return "misconfigured", 500
    
    data = event.data or {}
    bucket, name = _extract_bucket_object(data)
    if not bucket or not name:
        return "ignored", 204
    
    if not _should_process(name):
        logging.info("Not a marker file; ignoring: %s", name)
        return "ignored", 204

    model_dir = f"gs://{bucket}/{_model_prefix(name)}"
    version_tag = name.split("/", 2)[1] if "/" in name else "unknown"
    logging.info("Marker matched: %s | dir=%s | version=%s", name, model_dir, version_tag)

    # Guard 1: completeness (important for manual uploads)
    if not _savedmodel_complete(bucket, name):
        logging.info("SavedModel incomplete; skipping: %s", model_dir)
        return "ignored", 202
    
    # NEW: atomic lock per version to prevent duplicate uploads
    bkt = storage_client.bucket(bucket)
    lock_blob_path = _model_prefix(name) + "/.deploy.lock"
    lock_blob = bkt.blob(lock_blob_path)
    try:
        lock_blob.upload_from_string(version_tag, if_generation_match=0)
    except Exception as e:
        logging.info("Lock exists; another invocation handled deploy. %s", e)
        return "ok", 200
    
    # Guard 2: idempotency under parent model
    if _version_exists_under_parent(version_tag):
        logging.info("Version '%s' already exists under parent; skipping.", version_tag)
        return "ok", 200
    

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    parent_model = (f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{PARENT_MODEL_ID}"
                    if PARENT_MODEL_ID else None)
    logging.info("Using parent_model=%s", parent_model)

    # --- Register (upload) to Model Registry ---
    model = aiplatform.Model.upload(
                                        display_name=f"tfx-model-{version_tag}",
                                        artifact_uri=model_dir,
                                        serving_container_image_uri=PRED_IMAGE,
                                        parent_model=parent_model,
                                        labels={"source": "tfx", "prefix": "serving_model", "version": version_tag},
                                        sync=True,
                                    )

    # --- Deploy to existing Endpoint ---
    endpoint = aiplatform.Endpoint(_endpoint_name(PROJECT_ID, LOCATION, ENDPOINT_ID))

    if UNDEPLOY:
        for dm in endpoint.list_models():
            try:
                endpoint.undeploy(deployed_model_id=dm.id, sync=True)
            except Exception as e:
                logging.warning("undeploy warn: %s", e)

    model.deploy(
                    endpoint=endpoint,
                    traffic_percentage=TRAFFIC,
                    machine_type=MACHINE_TYPE,
                    min_replica_count=1, max_replica_count=1,
                    sync=True,
                )
    logging.info("Deployed %s to %s at %d%%", model.resource_name, endpoint.resource_name, TRAFFIC)
    return "ok", 200