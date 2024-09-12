import json
import argparse
import logging
from datetime import datetime

from lighteval.main_accelerate import main, EnvConfig, create_model_config, load_model

from src.envs import RESULTS_REPO, CACHE_PATH, TOKEN
from src.backend.manage_requests import EvalRequest
from src.logging import setup_logger

logging.getLogger("openai").setLevel(logging.WARNING)
logger = setup_logger(__name__)

def run_evaluation(eval_request: EvalRequest, task_names: str, batch_size: int, local_dir: str, accelerator: str, region: str, vendor: str, instance_size: str, instance_type: str, limit=None):
    """Runs one evaluation for the current evaluation request file using lighteval, then pushes the results to the hub.

    Args:
        eval_request (EvalRequest): Input evaluation request file representation
        task_names (list): Tasks to launch
        batch_size (int): Selected batch size
        accelerator (str): Inference endpoint parameter for running the evaluation
        region (str):  Inference endpoint parameter for running the evaluation
        vendor (str):  Inference endpoint parameter for running the evaluation
        instance_size (str):  Inference endpoint parameter for running the evaluation
        instance_type (str):  Inference endpoint parameter for running the evaluation
        local_dir (str): Where to save the results locally
        no_cache (bool, optional): Whether to use a cache or not.
        limit (int, optional): Whether to use a number of samples only for the evaluation - only for debugging
    """    

    if limit:
        logger.info("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    args_dict = {
            # Endpoint parameters
            "endpoint_model_name":eval_request.model,
            "accelerator": accelerator,
            "vendor": vendor,
            "region": region,
            "instance_size": instance_size,
            "instance_type": instance_type,
            "reuse_existing": False,
            "model_dtype": eval_request.precision,
            "revision": eval_request.revision,
            # Save parameters
            "push_results_to_hub": True,
            "save_details": True,
            "push_details_to_hub": True,
            "public_run": False,
            "cache_dir": CACHE_PATH,
            "results_org": RESULTS_REPO,
            "output_dir": local_dir,
            "job_id": str(datetime.now()),
            # Experiment parameters
            "override_batch_size": batch_size,
            "custom_tasks": "custom_tasks.py",
            "tasks": task_names,
            "max_samples": limit,
            "use_chat_template": False,
            "system_prompt": None,
            # Parameters which would be set to things by the kwargs if actually using argparse
            "inference_server_address": None,
            "model_args": None,
            "num_fewshot_seeds": None,
            "delta_weights": False,
            "adapter_weights": False
    }
    args = argparse.Namespace(**args_dict)

    try:
        results = main(args)

        results["config"]["model_dtype"] = eval_request.precision
        results["config"]["model_name"] = eval_request.model
        results["config"]["model_sha"] = eval_request.revision

        dumped = json.dumps(results, indent=2)
        logger.info(dumped)
    except Exception as e: # if eval failed, we force a cleanup
        env_config = EnvConfig(token=TOKEN, cache_dir=args.cache_dir)

        model_config = create_model_config(args=args, accelerator=accelerator)
        model, _ = load_model(config=model_config, env_config=env_config)
        model.cleanup()


    return results
