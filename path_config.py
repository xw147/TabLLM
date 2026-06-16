"""
Path Configuration for TabLLM Project

All filesystem paths that may differ between environments are defined here.
Edit ROOT_DIR (and optionally OUTPUT_DIR) to match your system, then import
from this module anywhere you need a path.

Examples:
  macOS:   ROOT_DIR = "/Users/yourname"
  Linux:   ROOT_DIR = "/home/yourname"
  Windows: ROOT_DIR = "C:/Users/yourname"
"""

import os
from pathlib import Path

# ============================================================================
# CONFIGURE YOUR ROOT DIRECTORY HERE
# ============================================================================

ROOT_DIR = Path("/Users/work")   # <-- EDIT THIS LINE for your system

# ============================================================================
# PROJECT ROOTS
# ============================================================================

TABLLM_ROOT       = ROOT_DIR / "TabLLM"
TFEW_ROOT         = ROOT_DIR / "t-few"

# ============================================================================
# DATA PATHS
# ============================================================================

# Raw input datasets (one sub-folder per dataset, e.g. datasets/ico/)
DATASETS_DIR      = TABLLM_ROOT / "datasets"

# Serialized / Arrow datasets produced by create_external_datasets.py
DATASETS_SERIALIZED_DIR = TABLLM_ROOT / "datasets_serialized"

# Default serialized ICO dataset (used as --input in query_gpt3.py)
ICO_SERIALIZED_DIR = DATASETS_SERIALIZED_DIR / "ico"

# ============================================================================
# TEMPLATE PATHS
# ============================================================================

TEMPLATES_DIR     = TABLLM_ROOT / "templates"

def get_template_path(task_name: str) -> Path:
    """Return the YAML template path for a given task name."""
    return TEMPLATES_DIR / f"templates_{task_name}.yaml"

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Model prediction / metric outputs (CSVs, JSONs)
OUTPUT_DIR        = TABLLM_ROOT / "output"

def get_gpt_output_csv(task_name: str, timestamp: str) -> Path:
    """Return the output CSV path for a GPT run"""
    return OUTPUT_DIR / f"outputs-{task_name}-{timestamp}.csv"

def get_model_results_json(dataset_name: str, model_name: str) -> Path:
    """Return the JSON results path for an ML model run."""
    return OUTPUT_DIR / f"{dataset_name}_{model_name}_results.json"

# ============================================================================
# GPT EVALUATION PATHS  ← edit these when switching GPT output files
# ============================================================================

# Input: GPT output CSV produced by query_gpt3.py.
# Update this to point at the CSV you want to evaluate.
GPT_EVAL_INPUT_CSV  = OUTPUT_DIR / "outputs-ico-20260616-120036.csv"

# Output: metrics summary CSV written by evaluate_gpt3_from_csv().
# Leave as None to auto-generate a timestamped filename at runtime.
GPT_EVAL_OUTPUT_CSV = OUTPUT_DIR / "ico_gpt3_metrics_summary.csv"

# ============================================================================
# HELPER: per-dataset data directory
# ============================================================================

def get_dataset_dir(dataset_name: str) -> Path:
    """Return the raw-data directory for a given dataset, e.g. datasets/ico/"""
    return DATASETS_DIR / dataset_name

def get_serialized_dataset_dir(dataset_name: str) -> Path:
    """Return the serialized Arrow dataset directory for a given dataset."""
    return DATASETS_SERIALIZED_DIR / dataset_name

# ============================================================================
# VALIDATION (optional — call at startup to catch misconfiguration early)
# ============================================================================

def validate_paths(verbose: bool = True) -> bool:
    """
    Check that the key directories exist.
    Returns True if all exist, False otherwise.
    """
    paths_to_check = {
        "ROOT_DIR":               ROOT_DIR,
        "TABLLM_ROOT":            TABLLM_ROOT,
        "DATASETS_DIR":           DATASETS_DIR,
        "DATASETS_SERIALIZED_DIR": DATASETS_SERIALIZED_DIR,
        "TEMPLATES_DIR":          TEMPLATES_DIR,
        "OUTPUT_DIR":             OUTPUT_DIR,
    }

    all_exist = True
    for name, path in paths_to_check.items():
        exists = path.exists()
        if verbose:
            status = "✓" if exists else "✗ MISSING"
            print(f"  {status}  {name}: {path}")
        if not exists:
            all_exist = False

    if verbose:
        print("All paths OK." if all_exist else "Some paths are missing — update ROOT_DIR.")
    return all_exist


if __name__ == "__main__":
    validate_paths(verbose=True)
