"""
submit_pipeline.py
Compiles the pipeline to YAML and uploads it to the running Kubeflow instance.

Usage:
    # Install deps first (once):
    pip install -r requirements.txt

    # Compile + upload:
    python submit_pipeline.py

    # Compile only (no upload):
    python submit_pipeline.py --compile-only

    # Compile + run immediately:
    python submit_pipeline.py --run
"""

import argparse
import os
import sys
from datetime import datetime

KFP_HOST = os.environ.get("KFP_HOST", "http://localhost:8080")
PIPELINE_FILE = "pipeline.yaml"
PIPELINE_NAME = "Stock Prediction Pipeline"
EXPERIMENT_NAME = "Stock Market"
RUN_NAME = "stock-prediction-run"


def compile_pipeline():
    from pipeline import stock_prediction_pipeline
    from kfp import compiler
    compiler.Compiler().compile(stock_prediction_pipeline, PIPELINE_FILE)
    print(f"[OK] Compiled -> {PIPELINE_FILE}")


def upload_pipeline(client):
    import kfp
    # Check if pipeline already exists
    pipelines = client.list_pipelines(page_size=100)
    existing = None
    if pipelines and pipelines.pipelines:
        for p in pipelines.pipelines:
            if p.display_name == PIPELINE_NAME:
                existing = p
                break

    if existing:
        print(f"[INFO] Pipeline '{PIPELINE_NAME}' already exists — uploading new version")
        version_name = datetime.now().strftime("v%Y%m%d-%H%M%S")
        pipeline_obj = client.upload_pipeline_version(
            pipeline_package_path=PIPELINE_FILE,
            pipeline_version_name=version_name,
            pipeline_id=existing.pipeline_id,
        )
    else:
        pipeline_obj = client.upload_pipeline(
            pipeline_package_path=PIPELINE_FILE,
            pipeline_name=PIPELINE_NAME,
        )
    print(f"[OK] Pipeline uploaded — ID: {pipeline_obj.pipeline_id}")
    return pipeline_obj


def create_run(client, pipeline_obj):
    # Get or create experiment
    try:
        experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
    except Exception:
        experiment = client.create_experiment(name=EXPERIMENT_NAME)
    print(f"[OK] Experiment: {experiment.experiment_id}")

    run = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_FILE,
        arguments={},
        run_name=RUN_NAME,
        experiment_name=EXPERIMENT_NAME,
    )
    print(f"[OK] Run started — ID: {run.run_id}")
    print(f"     View at: {KFP_HOST}/#/runs/details/{run.run_id}")
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true",
                        help="Compile to YAML only, do not upload")
    parser.add_argument("--run", action="store_true",
                        help="Also trigger a pipeline run after uploading")
    args = parser.parse_args()

    compile_pipeline()

    if args.compile_only:
        print("Compile-only mode — done.")
        return

    import kfp
    print(f"\nConnecting to Kubeflow at {KFP_HOST} ...")
    client = kfp.Client(host=KFP_HOST)

    pipeline_obj = upload_pipeline(client)

    if args.run:
        create_run(client, pipeline_obj)
    else:
        print("\n[INFO] To trigger a run:")
        print(f"  python submit_pipeline.py --run")
        print(f"  or open {KFP_HOST} and click 'Create Run'")


if __name__ == "__main__":
    main()
