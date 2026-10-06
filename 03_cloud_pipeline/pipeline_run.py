# [1] Import Libraries
import os
import sys
from typing import Text
from setup_logging import setup_pipeline_logging

from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as kfp_v2
import google.cloud.aiplatform as aip

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [2] Constants/Variables
# ============================================================================
# USER CONFIGURATION — REPLACE THE PLACEHOLDERS BELOW WITH YOUR OWN VALUES
# Never commit real project IDs, service-account emails, or resource IDs.
# ============================================================================
# 2.1. Project/Region
PROJECT_ID = "your-gcp-project-id"                                 # TODO: your Google Cloud project ID
REGION = "us-central1"                                              # TODO: your Vertex AI region

# 2.2. GCS Paths (Cloud Storage)
BUCKET = "your-bucket-name"                                         # TODO: your bucket name (names are global; pick your own)
DATA_DIR = "data"                                                   # directory containing data spans inside BUCKET

LOCAL_PIPELINE_ROOT = os.getenv("LOCAL_PIPELINE_ROOT", "/tmp/pipeline-root")
REAL_PIPELINE_ROOT = os.getenv("PIPELINE_ROOT", f"gs://{BUCKET}/pipeline_root")

SERVING_MODEL_DIR = f"gs://{BUCKET}/serving_model"
SCHEMA_FILE = f"{REAL_PIPELINE_ROOT}/schema/schema.pbtxt"

# 2.3. Container Image (lives in Artifact Registry)
# TODO: format is <region>-docker.pkg.dev/<project-id>/<artifact-repo>/<image>:<tag>
DEFAULT_IMAGE = "us-central1-docker.pkg.dev/your-gcp-project-id/tfx-pipeline/tfx-pipeline:latest"

# 2.4. Pipeline Identifiers
PIPELINE_NAME = "conc-ppln"
TEMPLATE_PATH = "pipeline.json" 

# 2.5. Module Path
MODULE_FILE = "/app/module.py"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [3] Logging
# Create logs directory
app_log, full_log = setup_pipeline_logging(logs_dir="/tmp/logs")

import logging as pylog
logger = pylog.getLogger(__name__)
logger.info(f"Main application log: {app_log}")
logger.info(f"Full detailed log: {full_log}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [4] Pipeline Function
def build_pipeline(components, pipeline_root: Text) -> tfx_pipeline.Pipeline:
    """Create a TFX pipeline object compatible with KubeflowV2DagRunner."""
    
    p = tfx_pipeline.Pipeline(
                                pipeline_name=PIPELINE_NAME,
                                pipeline_root=pipeline_root,
                                components=components,
                                enable_cache=True,
                            )
    
    return p

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [5] Pipeline Main Run Function
if __name__ == "__main__":
    
    # Ensure we can import base_pipeline.py from the Container Working Directory
    module_path = os.getcwd()
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    # Assemble Components (using cloud paths + module inside the image)
    from base_pipeline import init_components
    components = init_components(
                                    bucket_name=BUCKET,
                                    data_dir=DATA_DIR,
                                    schema_file=SCHEMA_FILE,
                                    module_file=MODULE_FILE,
                                    serving_model_dir=SERVING_MODEL_DIR
                                )
    
    # Compile to Vertex Pipeline Spec (pipeline.json)
    runner = kfp_v2.KubeflowV2DagRunner(
                                            config=kfp_v2.KubeflowV2DagRunnerConfig(default_image=DEFAULT_IMAGE),
                                            output_filename=TEMPLATE_PATH,
                                        )
    runner.run(build_pipeline(components, LOCAL_PIPELINE_ROOT))


    
    # --- Stage user wheels to GCS and patch the spec ---
    import glob, json
    from google.cloud import storage

    local_wheels = glob.glob(os.path.join(LOCAL_PIPELINE_ROOT, "_wheels", "*.whl"))
    if not local_wheels:
        raise RuntimeError("No wheels found under LOCAL_PIPELINE_ROOT/_wheels")

    gcs_base = f"gs://{BUCKET}/pipeline_root/_wheels"
    print("Uploading wheels to:", gcs_base)

    # Upload wheels to GCS using google-cloud-storage (works with WIF)
    st = storage.Client(project=PROJECT_ID)
    bkt = st.bucket(BUCKET)
    mapping = {}  # local -> gs
    for lp in local_wheels:
        name = os.path.basename(lp)
        blob = bkt.blob(f"pipeline_root/_wheels/{name}")
        blob.upload_from_filename(lp)
        mapping[lp] = f"{gcs_base}/{name}"
    print("Wheel mapping:", mapping)

    # Patch the compiled KFP spec to point to gs://… instead of /tmp/…
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        spec = f.read()
    for lp, gp in mapping.items():
        # TFX embeds "module@<path>". Replace both the bare path and the full "module@<path>"
        spec = spec.replace(lp, gp).replace(f"module@{lp}", f"module@{gp}")
    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        f.write(spec)
    print("Patched template to use gs:// wheels.")
    # --- end patch ---




    # Submit Compiled Spec to VertexAI Pipelines
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")
    print("Spec exists?", os.path.exists(TEMPLATE_PATH))
    print("Using pipeline_root:", REAL_PIPELINE_ROOT)
    print("Default image:", DEFAULT_IMAGE)

    runtime_sa = os.getenv("RUNTIME_SA") or ""   # may be empty in CI
    print("Runtime SA:", runtime_sa)

    job = aip.PipelineJob(
                            display_name=PIPELINE_NAME,
                            template_path=TEMPLATE_PATH,
                            pipeline_root=REAL_PIPELINE_ROOT,
                        )
    try:
        if runtime_sa.strip():
            job.submit(service_account=runtime_sa.strip())
        else:
            job.submit()  # let Vertex use the default compute SA
        print("Job submitted.")
        if getattr(job, "_gca_resource", None):
            print("Vertex pipeline job:", job.resource_name)
            print("Job ID:", job.job_id)
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        print("AIP submission failed type:", type(e).__name__)
        print("AIP submission failed msg:", str(e))
        sys.exit(1)
    
    # --- upload pipeline.json to GCS (optionally after completion) ---
    import time
    from google.cloud import storage

    WAIT_FOR_COMPLETION = os.getenv("WAIT_FOR_COMPLETION", "0") == "1"
    if WAIT_FOR_COMPLETION:
        print("Waiting for Vertex Pipeline to finish before uploading spec...")
        job.wait()

    st = storage.Client(project=PROJECT_ID)
    bkt = st.bucket(BUCKET)
    blob = bkt.blob("pipeline.json")          # gs://<BUCKET>/pipeline.json
    blob.upload_from_filename(TEMPLATE_PATH)  # overwrites latest generation
    print(f"Uploaded: gs://{BUCKET}/pipeline.json")
    # --- end upload ---
