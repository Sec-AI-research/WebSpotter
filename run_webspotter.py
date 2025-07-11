"""
WebSpotter Automation Script

This script automates the complete WebSpotter pipeline as described in the paper:
"Achieving Interpretable DL-based Web Attack Detection through Malicious Payload Localization".

The script performs the following steps sequentially:
1. Trains the DL-based detection model (TextCNN)
2. Computes MSU (Minimal Semantic Unit) importance scores via embedding attribution
3. Trains the malicious payload localization model and evaluates the localization performance

Usage: python run_webspotter.py <dataset_name>
Example: python run_webspotter.py CSIC
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Dataset-specific configurations from the paper
DATASET_CONFIG = {
    "CSIC": {"max_len": 700},
    "PKDD": {"max_len": 2100},  # PKDD requires longer sequence length because it includes HTTP headers
    "FPAD": {"max_len": 700},
    "CVE": {"max_len": 700}
}

EMBEDDING_DIM = 512        # Dimension of embedding vectors
SAMPLE_RATE = 0.01         # Fraction of location-labeled data for training
FEATURE_METHOD = "score_sort_with_textemb"  # Hybrid feature construction method

def run_command(cmd, step_name):
    print(f"[STEP] {step_name}")
    print(f"  Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    returncode = process.poll()
    if returncode != 0:
        print(f"[ERROR] {step_name} failed!")
        print(process.stderr.read())
        sys.exit(1)
    
    print(f"  {step_name} completed successfully\n")

import time

def main(dataset):
    """Execute the complete WebSpotter pipeline"""

    start_total = time.time()

    # Validate dataset name
    dataset = dataset.upper()
    if dataset not in DATASET_CONFIG:
        print(f"[ERROR] Unknown dataset: {dataset}. Supported datasets: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)

    config = DATASET_CONFIG[dataset]

    # Path configurations
    base_dir = Path.cwd()
    dataset_dir = base_dir / "datasets" / dataset
    model_dir = base_dir / "tmp_model"
    explain_dir = base_dir / "explain_result" / dataset / "post_explain"
    result_dir = base_dir / "explain_result" / dataset

    # Create necessary directories
    explain_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train the detection model (TextCNN)
    detection_model = f"textcnn-{config['max_len']}-{dataset}-{EMBEDDING_DIM}-None-0.pth"
    model_path = model_dir / detection_model

    print("[1/4] Training detection model...")
    start = time.time()
    if not model_path.exists():
        run_command([
            "python", "classification/run.py",
            "--tmp_dir", str(dataset_dir),
            "--tmp_model", str(model_dir),
            "--dataset", dataset.lower(),
            "--max_len", str(config['max_len'])
        ], "Training detection model")
    else:
        print(f"[INFO] Detection model already exists at: {model_path}, skipping training.")
    print(f"  Time taken: {time.time() - start:.2f} seconds\n")

    # 2a. Compute training set MSU importance scores
    print("[2/4] Computing training set MSU importance scores...")
    start = time.time()
    run_command([
        "python", "localization/post_explain/run_explain.py",
        "--model_path", str(model_path),
        "--outputdir", str(explain_dir / "train"),
        "--dataset", dataset.lower(),
        "--test_path", str(dataset_dir / "train.jsonl")
    ], "Computing training set MSU importance scores")
    print(f"  Time taken: {time.time() - start:.2f} seconds\n")

    # 2b. Compute test set MSU importance scores
    print("[3/4] Computing test set MSU importance scores...")
    start = time.time()
    run_command([
        "python", "localization/post_explain/run_explain.py",
        "--model_path", str(model_path),
        "--outputdir", str(explain_dir / "test"),
        "--dataset", dataset.lower(),
        "--test_path", str(dataset_dir / "test.jsonl")
    ], "Computing test set MSU importance scores")
    print(f"  Time taken: {time.time() - start:.2f} seconds\n")

    # 3. Train and evaluate the localization model
    print("[4/4] Training payload localization model...")
    start = time.time()
    run_command([
        "python", "localization/binary_based/run.py",
        "--feature_method", FEATURE_METHOD,
        "--dataset", dataset.lower(),
        "--train_path", str(explain_dir / "train/train.jsonl_withscore"),
        "--test_path", str(explain_dir / "test/test.jsonl_withscore"),
        "--output_path", str(result_dir),
        "--sample_rate", str(SAMPLE_RATE)
    ], "Training payload localization model")
    print(f"  Time taken: {time.time() - start:.2f} seconds\n")

    total_time = time.time() - start_total
    print("[COMPLETE] WebSpotter pipeline executed successfully")
    print(f"  Detection model: {model_path}")
    print(f"  MSU importance scores: {explain_dir}")
    print(f"  Localization results: {result_dir}")
    print(f"  Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate the WebSpotter pipeline for interpretable web attack detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset",
        help="Name of the dataset (case-insensitive, e.g., FPAD, CSIC, PKDD, CVE)",
        type=str
    )
    
    args = parser.parse_args()
    main(args.dataset)

