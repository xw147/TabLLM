"""
Generate summary CSV tables from experiment results.

Reads results from exp_out/{exp_name}/dev_scores.json, which is produced by
both evaluate_external_dataset.py (ML models: lr, xgboost, lightgbm, …) and
the t-few training pipeline.

Folder naming convention (ML models):
  exp_out/{model}_{dataset}_numshot{N}_seed{S}_{label_strategy}/dev_scores.json
  e.g. exp_out/lr_ico_numshot4_seed42_all/dev_scores.json

Folder naming convention (t-few):
  exp_out/{model}_{dataset}_numshot{N}_seed{S}_{spec}/dev_scores.json
  e.g. exp_out/t03b_ico_numshot4_seed42_ia3_pretrained100k/dev_scores.json

Usage:
  # Summarise ML runs for lr with label strategy 'all'
  python get_result_table.py -e "lr_ico_*_all"

  # Summarise t-few runs
  python get_result_table.py -e "t03b_ico_*_ia3_pretrained100k"

  # Multiple experiment patterns (one row per pattern)
  python get_result_table.py -e "lr_ico_*_all" "xgboost_ico_*_all" "t03b_ico_*_ia3_pretrained100k"

  # Choose which metric to report (default: AUC / auroc)
  python get_result_table.py -e "lr_ico_*_all" -m auroc auprc macro_f1 accuracy

  # Custom output file
  python get_result_table.py -e "lr_ico_*_all" -o my_summary.csv
"""

import argparse
import json
import os
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd


METRIC_NAMES = [
    'precision', 'recall', 'f1_binary', 'auprc',
    'specificity', 'auroc', 'macro_f1', 'accuracy'
]

# Mapping from dev_scores.json keys → unified internal metric names.
# Covers both ML model output (AUC, PR, sensitivity, …) and t-few output.
TFEW_METRIC_MAP = {
    'precision': 'precision',
    'recall': 'recall',
    'sensitivity': 'recall',       # sensitivity == recall
    'PR': 'auprc',
    'AUC': 'auroc',
    'specificity': 'specificity',
    'f1_binary': 'f1_binary',
    'micro_f1': 'f1_binary',
    'macro_f1': 'macro_f1',
    'accuracy': 'accuracy',
}


def fmt(mean, std):
    """Format mean (std) string."""
    return f"{mean:.2f} ({std:.2f})"


def load_tfew_results(exp_name_template, exp_out_dir="exp_out"):
    """
    Load results from dev_scores.json files matching a glob pattern.

    Works for both ML models (lr, xgboost, lightgbm, …) and t-few runs —
    both write the same dev_scores.json format.

    Folder naming: {model}_{dataset}_numshot{N}_seed{S}_{...}
    Groups values by numshot across seeds and returns mean/std.
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
        "-e", "--exp_name_templates", nargs="+", required=True,
        help="Experiment name glob patterns matching folders under exp_out/. "
             "Each pattern produces one row in the table. "
             "Examples: 'lr_ico_*_all'  'xgboost_ico_*_high_only'  't03b_ico_*_ia3_pretrained100k'"
    )
    parser.add_argument(
        "--exp_out_dir", default="exp_out",
        help="Directory containing experiment outputs (default: exp_out)."
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
    for tmpl in args.exp_name_templates:
        print(f"Loading: {tmpl}")
        label, dataset_name, rows = load_tfew_results(tmpl, args.exp_out_dir)
        if rows:
            all_experiments.append((label, dataset_name, rows))

    if not all_experiments:
        print("No results found. Check that exp_out/ contains matching folders with dev_scores.json.")
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
