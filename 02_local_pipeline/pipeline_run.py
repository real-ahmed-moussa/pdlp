# [1] Import Libraries
import os
import sys
import shutil
from typing import Text
from setup_logging import setup_pipeline_logging

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tfx.orchestration.metadata import Metadata
from tfx.orchestration.metadata import sqlite_metadata_connection_config

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [2] Pipeline Directories
# 2.1. Pipeline Name
pipeline_name = 'ff_pipeline'

# 2.2. Pipeline Inputs
pipeline_dir = os.getcwd()
data_dir = os.path.join(pipeline_dir, 'loc_dev_data')
module_file = os.path.join(pipeline_dir, 'module.py')
requirement_file = os.path.join(pipeline_dir, 'requirements.txt')

# 2.3. Pipeline Outputs
output_base = os.path.join(pipeline_dir, 'output')
serving_model_dir = os.path.join(output_base, 'serving_model')
pipeline_root = os.path.join(output_base, 'ppln_root')
schema_file = os.path.join(pipeline_root, "schema", "schema.pbtxt")
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [3] Logging
# Create logs directory
app_log, full_log = setup_pipeline_logging(logs_dir=os.path.join(output_base, 'logs'))

import logging as pylog
logger = pylog.getLogger(__name__)
logger.info(f"Main application log: {app_log}")
logger.info(f"Full detailed log: {full_log}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [4] Pipeline Functions
# 4.1. Function to Export Latest Schema
def export_latest_schema(metadata_path, output_path):
    config = sqlite_metadata_connection_config(metadata_path)
    with Metadata(config) as metadata:
        artifacts = metadata.store.get_artifacts_by_type("Schema")
        if not artifacts:
            logger.warning("No schema found.")
            return
        latest = sorted(artifacts, key=lambda a: a.id, reverse=True)[0]
        schema_uri = latest.uri
        schema_file = os.path.join(schema_uri, "schema.pbtxt")
        if os.path.exists(schema_file):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(schema_file, output_path)
            logger.info(f"Schema.pbtxt exported to {output_path}")
        else:
            logger.warning("Schema.pbtxt not found in artifact URI.")

# 4.2. Function to Inspect the Anomalies
def inspect_latest_validator_anomalies(pipeline_root, component_name="ExampleValidator"):
    # Path to ExampleValidator anomalies directory
    anomalies_dir = os.path.join(pipeline_root, component_name, "anomalies")
    logger.info(f"Checking for anomalies in: {anomalies_dir}")

    # Check if anomalies directory exists
    if not os.path.exists(anomalies_dir):
        logger.info(f"No anomalies directory found at {anomalies_dir}.")
        return

    # Get list of run directories (e.g., '5' for run ID 5)
    run_dirs = [d for d in os.listdir(anomalies_dir) if os.path.isdir(os.path.join(anomalies_dir, d))]
    if not run_dirs:
        logger.info(f"No run directories found in {anomalies_dir}.")
        return

    # Find the latest run directory (assuming run IDs are numeric)
    try:
        latest_run = max(run_dirs, key=lambda x: int(x) if x.isdigit() else -1)
    except ValueError:
        logger.error("Unable to determine latest run directory: run IDs are not numeric.")
        return

    latest_run_dir = os.path.join(anomalies_dir, latest_run)
    logger.info(f"Inspecting latest run directory: {latest_run_dir}")

    # Look for split directories (e.g., Split-train, Split-eval)
    split_dirs = [d for d in os.listdir(latest_run_dir) if os.path.isdir(os.path.join(latest_run_dir, d))]
    if not split_dirs:
        logger.info(f"No split directories found in {latest_run_dir}.")
        return

    found_anomalies = False
    for split_dir in split_dirs:
        split_path = os.path.join(latest_run_dir, split_dir)
        # Check for both SchemaDiff and SchemaDiff.pb
        for filename in ["SchemaDiff", "SchemaDiff.pb"]:
            schema_diff_path = os.path.join(split_path, filename)
            if os.path.exists(schema_diff_path):
                found_anomalies = True
                logger.info(f"Found {filename} for split {split_dir} at {schema_diff_path}")
                try:
                    with open(schema_diff_path, 'rb') as f:
                        anomalies = anomalies_pb2.Anomalies()
                        anomalies.ParseFromString(f.read())
                        if anomalies.anomaly_info:
                            logger.warning(f"Anomalies detected in {split_dir}:")
                            for feature, info in anomalies.anomaly_info.items():
                                logger.error(f"  - Feature: {feature}")
                                logger.error(f"    Description: {info.description}")
                                logger.error(f"    Severity: {info.severity}")
                                logger.error("  -----")
                        else:
                            logger.info(f"No anomalies detected in {split_dir}.")
                except Exception as e:
                    logger.error(f"Error reading {filename} for {split_dir}: {str(e)}")
            else:
                logger.debug(f"No {filename} found for split {split_dir} at {schema_diff_path}")

    if not found_anomalies:
        logger.info(f"No SchemaDiff or SchemaDiff.pb files found in any splits in {latest_run_dir}.")

# 4.3. Pipeline Instantiation Function
def init_beam_pipeline(components, pipeline_root: Text, direct_num_workers: int):
    
    logger.info(f'Pipeline root set to:{pipeline_root}')
    beam_arg = [
        f'--direct_num_workers={direct_num_workers}',
        #f'--requirements_file={requirement_file}',
        f'--direct_running_mode=multi_processing'
    ]
    
    p = pipeline.Pipeline(
                            pipeline_name=pipeline_name,
                            pipeline_root=pipeline_root,
                            components=components,
                            enable_cache=False,
                            metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                            beam_pipeline_args=beam_arg,
                        )
    
    return p

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# [5] Pipeline Main Run Function
if __name__ == "__main__":
    
    module_path = os.getcwd()
    if module_path not in sys.path:
        print(module_path)
        sys.path.append(module_path)
    
    from base_pipeline import init_components
    components = init_components(
                                    data_dir=data_dir,
                                    schema_file=schema_file,
                                    module_file=module_file,
                                    _serving_model_dir=serving_model_dir
                                )
    direct_num_workers = int(os.cpu_count()/2)
    direct_num_workers = 1 if direct_num_workers < 1 else direct_num_workers
    
    pipeline = init_beam_pipeline(components, pipeline_root, direct_num_workers)
    BeamDagRunner().run(pipeline)

    export_latest_schema(metadata_path, schema_file)
    inspect_latest_validator_anomalies(pipeline_root=pipeline_root)