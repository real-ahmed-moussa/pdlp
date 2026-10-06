import os
import sys
from datetime import datetime
import logging
from absl import logging as absl_log

def setup_pipeline_logging(logs_dir="logs", log_prefix="pipeline_log"):
    # Ensure log directory exists
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # [1] Create TWO log files
    app_log_path = os.path.join(logs_dir, f"{log_prefix}_app_{timestamp}.log")
    full_log_path = os.path.join(logs_dir, f"{log_prefix}_full_{timestamp}.log")

    # Clear any existing handlers
    logging.root.handlers = []
    absl_log.get_absl_handler().flush()

    # [2] Configure APPLICATION-ONLY log file (your messages)
    app_file_handler = logging.FileHandler(app_log_path)
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.addFilter(
        lambda record: record.name.startswith(('__main__', 'pipeline_run', 'base_pipeline')))
    app_file_handler.setFormatter(
        logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))

    # [3] Configure FULL log file (all messages)
    full_file_handler = logging.FileHandler(full_log_path)
    full_file_handler.setLevel(logging.INFO)
    full_file_handler.setFormatter(
        logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))

    # [4] Configure console handler (all messages)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))

    # [5] Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[app_file_handler, full_file_handler, console_handler]
    )

    # [6] Configure Abseil logging
    absl_log.set_verbosity(absl_log.INFO)
    absl_handler = absl_log.get_absl_handler()
    absl_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logging.info(f"Application logs will be written to: {app_log_path}")
    logging.info(f"Full logs will be written to: {full_log_path}")
    
    return app_log_path, full_log_path