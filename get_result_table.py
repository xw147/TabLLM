"""
Generate summary CSV tables from experiment results.

Reads results from:
  1. evaluate_external_dataset.py JSON files ({dataset}_{model}_results.json)
  2. t-few experiment outputs (exp_out/{exp_name}/dev_scores.json)

Usage:
  # Summarize all result JSON files in current directory
  python get_result_table.py

  # Specify result files or glob patterns
  python get_result_table.py -f "ico_tabpfn_results.json" "ico_lr_results.json"

  # Include t-few experiment results
  python get_result_table.py --tfew -e "t03b_ico_*_ia3_pretrained100k"

  # Choose which metrics to display (default: all)
  python get_result_table.py -m auprc auroc f1_score accuracy

  # Custom output file
  python get_result_table.py -o my_summary.csv
"""

import argparse
import json
import os
from glob import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_NAMES = [
    'precision', 'recall', 'f1_score', 'auprc',
    'specificity', 'auroc', 'micro_f1', 'macro_f1', 'accuracy'
]

# Mapping from t-few dev_scores.json keys to our unified metric names
TFEW_METRIC_MAP = {
    'precision': 'precision',
    'recall': 'recall',
    'sensitivity': 'recall',       # sensitivity == recall
    'f1': 'f1_score',
    'PR': 'auprc',
    'AUC': 'auroc',
    'specificity': 'specificity',
    'micro_f1': 'micro_f1',
    'macro_f1': 'macro_f1',
    'accuracy': 'accuracy',
}


def fmt(mean, std):
    """Format mean (std) string."""
    return f"{mean:.2f} ({std:.2f})"


def load_evaluate_results(file_path):
    """
    Load results from evaluate_external_dataset.py JSON output.
    Returns dict: {shot_size: {metric: {mean, std, raw_values}}}
    """
    with open(file_path) as f:
        data = json.load(f)

    model_name = data.get("model_info", {}).get("model_name", Path(file_path).stem)
    dataset_name = data.get("experimental_setup", {}).get("dataset_name", "unknown")
    results_by_shot = data.get("results_by_shot_size", {})

    rows = {}
    for shot_str, metrics in results_by_shot.items():
        shot_label = shot_str
        rows[shot_label] = {}
        for metric_name in METRIC_NAMES:
            if metric_name in metrics:
                m = metrics[metric_name]
                rows[shot_label][metric_name] = {
                    'mean': m['mean'],
                    'std': m['std'],
                    'raw_values': m.get('raw_values', []),
                }
    return model_name, dataset_name, rows


def load_tfew_results(exp_name_template, exp_out_dir="exp_out"):
    """
    Load results from t-few dev_scores.json files matching a glob pattern.
    Groups by dataset+numshot across seeds, returns same structure as load_evaluate_results.
    """
    pattern = os.path.join(exp_out_dir, exp_name_template, "dev_scores.json")
    all_files = glob(pattern)
    if not all_files:
        print(f"  Warning: No files found for pattern: {pattern}")
        return None, None, {}

    # Group raw metric values by (dataset, numshot)
    # exp_name format: {model}_{dataset}_numshot{N}_seed{S}_{suffix}
    raw_by_shot = defaultdict(lambda: defaultdict(list))
    model_name = None
    dataset_name = None

    for fname in all_files:
        folder_name = os.path.basename(os.path.dirname(fname))
        parts = folder_name.split("_")
        # Parse: model, dataset, numshotN, seedS, ...
        if model_name is None:
            model_name = parts[0]
        dset = parts[1]
        if dataset_name is None:
            dataset_name = dset

        numshot_part = [p for p in parts if p.startswith("numshot")]
        if numshot_part:
            shot_str = numshot_part[0].replace("numshot", "")
        else:
            shot_str = "unknown"

        # Read last line of dev_scores.json (final eval)
        with open(fname) as f:
            lines = f.readlines()
            if not lines:
                continue
            entry = json.loads(lines[-1])

        for tfew_key, unified_key in TFEW_METRIC_MAP.items():
            if tfew_key in entry:
                raw_by_shot[shot_str][unified_key].append(entry[tfew_key])

    # Compute mean/std per shot
    rows = {}
    for shot_str in sorted(raw_by_shot.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        rows[shot_str] = {}
        for metric_name in METRIC_NAMES:
            vals = raw_by_shot[shot_str].get(metric_name, [])
            if vals:
                rows[shot_str][metric_name] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'raw_values': [float(v) for v in vals],
                }

    label = exp_name_template.replace("*", "★")
    return label, dataset_name, rows


def build_summary_table(all_experiments, metrics_to_show=None):
    """
    Build a list of DataFrames, one per metric, from all loaded experiments.

    all_experiments: list of (label, dataset_name, rows_dict)
    Returns: dict of {metric_name: DataFrame}
    """
    if metrics_to_show is None:
        metrics_to_show = METRIC_NAMES

    # Collect all shot sizes across all experiments
    all_shots = set()
    for label, dataset_name, rows in all_experiments:
        all_shots.update(rows.keys())
    shot_order = sorted(all_shots, key=lambda x: int(x) if x.isdigit() else float('inf'))

    tables = {}
    for metric_name in metrics_to_show:
        table_rows = []
        for label, dataset_name, rows in all_experiments:
            row = {'experiment': label, 'dataset': dataset_name}
            for shot in shot_order:
                col = f"numshot{shot}"
                if shot in rows and metric_name in rows[shot]:
                    m = rows[shot][metric_name]
                    row[col] = fmt(m['mean'], m['std'])
                else:
                    row[col] = ""
            table_rows.append(row)
        tables[metric_name] = pd.DataFrame(table_rows)

    return tables


def main():
    parser = argparse.ArgumentParser(description="Generate summary CSV from experiment results.")
    parser.add_argument(
        "-f", "--files", nargs="*", default=None,
        help="Result JSON files from evaluate_external_dataset.py. "
             "If not specified, auto-discovers *_results.json in current directory."
    )
    parser.add_argument(
        "--tfew", action="store_true",
        help="Also include t-few experiment results."
    )
    parser.add_argument(
        "-e", "--exp_name_templates", nargs="*", default=None,
        help="t-few experiment name glob patterns (e.g. 't03b_ico_*_ia3_pretrained100k')."
    )
    parser.add_argument(
        "--exp_out_dir", default="exp_out",
        help="Directory containing t-few experiment outputs (default: exp_out)."
    )
    parser.add_argument(
        "-m", "--metrics", nargs="*", default=None,
        help=f"Which metrics to include. Default: all. Choices: {METRIC_NAMES}"
    )
    parser.add_argument(
        "-o", "--output", default="summary_results.csv",
        help="Output CSV file path (default: summary_results.csv)."
    )
    args = parser.parse_args()

    metrics_to_show = args.metrics if args.metrics else METRIC_NAMES

    all_experiments = []

    # 1. Load evaluate_external_dataset.py results
    if args.files is not None:
        result_files = args.files
    else:
        result_files = sorted(glob("*_results.json"))

    for fpath in result_files:
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue
        print(f"Loading: {fpath}")
        label, dataset_name, rows = load_evaluate_results(fpath)
        all_experiments.append((f"{dataset_name}_{label}", dataset_name, rows))

    # 2. Load t-few results
    if args.tfew and args.exp_name_templates:
        for tmpl in args.exp_name_templates:
            print(f"Loading t-few: {tmpl}")
            label, dataset_name, rows = load_tfew_results(tmpl, args.exp_out_dir)
            if rows:
                all_experiments.append((label, dataset_name, rows))

    if not all_experiments:
        print("No results found. Provide --files or place *_results.json in the current directory.")
        return

    # Build tables
    tables = build_summary_table(all_experiments, metrics_to_show)

    # Write combined CSV with sections per metric
    output_path = args.output
    with open(output_path, 'w') as f:
        for i, (metric_name, df) in enumerate(tables.items()):
            if i > 0:
                f.write("\n")
            f.write(f"# Metric: {metric_name}\n")
            df.to_csv(f, index=False)

    print(f"\nSaved summary to: {output_path}")
    print(f"Metrics included: {', '.join(metrics_to_show)}")

    # Also print to console
    for metric_name, df in tables.items():
        print(f"\n{'='*80}")
        print(f"Metric: {metric_name}")
        print(f"{'='*80}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
