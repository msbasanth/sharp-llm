"""Aggregate ICMLDE five-seed metrics and run paired significance tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import shapiro, ttest_rel, wilcoxon
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config


METRIC_COLUMNS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "macro_fpr",
    "macro_f1_across_cwe_mean",
    "macro_f1_across_cwe_std",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ICMLDE reproducibility results")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--output-root", default=None, help="Override ICMLDE output root")
    parser.add_argument("--metric", default="macro_f1", help="Primary metric for significance tests")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def find_metrics_file(output_root: Path, seed: int, variant: str) -> Path:
    return output_root / f"seed_{seed}" / variant / "evaluation" / "metrics.json"


def format_mean_std(mean_val: float | None, std_val: float | None, digits: int = 4) -> str:
    if mean_val is None or std_val is None:
        return "N/A"
    return f"{mean_val:.{digits}f} +/- {std_val:.{digits}f}"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    icmlde_cfg = cfg.get("icmlde", {})

    output_root = Path(args.output_root or icmlde_cfg.get("output_root", "outputs/icmlde2026/juliet118"))
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in icmlde_cfg.get("seeds", [42, 43, 44, 45, 46])]
    models: list[dict[str, str]] = icmlde_cfg.get("models", [])
    if not models:
        raise ValueError("No icmlde.models configured in config.yaml")

    raw_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for model in models:
        variant = model["variant"]
        tag = model.get("tag", f"ICMLDE:Juliet118:{model['name']}")
        for seed in seeds:
            metrics_path = find_metrics_file(output_root, seed, variant)
            if not metrics_path.exists():
                missing_rows.append(
                    {
                        "tag": tag,
                        "variant": variant,
                        "seed": seed,
                        "path": str(metrics_path),
                    }
                )
                continue

            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            row: dict[str, Any] = {
                "tag": tag,
                "model_name": model["name"],
                "model_id": model["model_id"],
                "variant": variant,
                "seed": seed,
                "metrics_path": str(metrics_path),
                "run_seed": data.get("run_seed"),
                "experiment_tag": data.get("experiment_tag"),
            }
            for metric in METRIC_COLUMNS:
                row[metric] = safe_float(data.get(metric))
            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    missing_df = pd.DataFrame(missing_rows)

    if not raw_df.empty:
        raw_df.to_csv(summary_dir / "raw_seed_metrics.csv", index=False)

    if not missing_df.empty:
        missing_df.to_csv(summary_dir / "missing_runs.csv", index=False)

    aggregate_rows: list[dict[str, Any]] = []
    for model in models:
        tag = model.get("tag", f"ICMLDE:Juliet118:{model['name']}")
        model_df = raw_df[raw_df["tag"] == tag] if not raw_df.empty else pd.DataFrame()
        row: dict[str, Any] = {
            "tag": tag,
            "model_name": model["name"],
            "model_id": model["model_id"],
            "variant": model["variant"],
            "expected_seed_count": len(seeds),
            "completed_seed_count": int(model_df["seed"].nunique()) if not model_df.empty else 0,
        }

        for metric in METRIC_COLUMNS:
            vals = model_df[metric].dropna().astype(float).values if not model_df.empty else np.array([])
            if len(vals) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
            elif len(vals) == 1:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = 0.0
            else:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1))

        row["macro_f1_mean_std"] = format_mean_std(row.get("macro_f1_mean"), row.get("macro_f1_std"))
        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    if not aggregate_df.empty:
        aggregate_df.to_csv(summary_dir / "aggregate_metrics.csv", index=False)

    aggregate_json = {
        "generated_at": utc_now(),
        "output_root": str(output_root),
        "seed_count": len(seeds),
        "seeds": seeds,
        "models": aggregate_rows,
        "missing_count": int(len(missing_rows)),
        "missing_runs": missing_rows,
    }
    with (summary_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_json, f, indent=2)

    # Significance tests: best proposed model vs baselines
    # Best proposed model is ICMLDE CodeT5-Base per plan.
    proposed_tag = "ICMLDE:Juliet118:CodeT5-Base"
    significance_results: list[dict[str, Any]] = []

    for baseline in models:
        baseline_tag = baseline.get("tag", f"ICMLDE:Juliet118:{baseline['name']}")
        if baseline_tag == proposed_tag:
            continue

        if raw_df.empty or "seed" not in raw_df.columns or args.metric not in raw_df.columns:
            significance_results.append(
                {
                    "comparison": f"{proposed_tag} vs {baseline_tag}",
                    "metric": args.metric,
                    "paired_seed_count": 0,
                    "paired_seeds": [],
                    "proposed_values": [],
                    "baseline_values": [],
                    "test_name": None,
                    "p_value": None,
                    "significant": None,
                    "error": "No completed seed metrics found",
                }
            )
            continue

        proposed_df = raw_df[raw_df["tag"] == proposed_tag][["seed", args.metric]] if not raw_df.empty else pd.DataFrame()
        baseline_df = raw_df[raw_df["tag"] == baseline_tag][["seed", args.metric]] if not raw_df.empty else pd.DataFrame()

        merged = proposed_df.merge(baseline_df, on="seed", suffixes=("_proposed", "_baseline"))
        merged = merged.dropna()

        result: dict[str, Any] = {
            "comparison": f"{proposed_tag} vs {baseline_tag}",
            "metric": args.metric,
            "paired_seed_count": int(len(merged)),
            "paired_seeds": merged["seed"].astype(int).tolist(),
            "proposed_values": merged[f"{args.metric}_proposed"].astype(float).tolist() if not merged.empty else [],
            "baseline_values": merged[f"{args.metric}_baseline"].astype(float).tolist() if not merged.empty else [],
            "test_name": None,
            "p_value": None,
            "significant": None,
            "error": None,
        }

        if len(merged) < 2:
            result["error"] = "Insufficient paired runs for significance testing"
            significance_results.append(result)
            continue

        if not _SCIPY_AVAILABLE:
            result["error"] = "scipy not available"
            significance_results.append(result)
            continue

        diffs = (
            merged[f"{args.metric}_proposed"].astype(float).values
            - merged[f"{args.metric}_baseline"].astype(float).values
        )

        try:
            normality_p = shapiro(diffs).pvalue if len(diffs) >= 3 else 0.0
            if normality_p > 0.05:
                stat = ttest_rel(
                    merged[f"{args.metric}_proposed"].astype(float).values,
                    merged[f"{args.metric}_baseline"].astype(float).values,
                    alternative="two-sided",
                )
                result["test_name"] = "paired_t_test"
                result["p_value"] = float(stat.pvalue)
            else:
                stat = wilcoxon(
                    merged[f"{args.metric}_proposed"].astype(float).values,
                    merged[f"{args.metric}_baseline"].astype(float).values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                result["test_name"] = "wilcoxon_signed_rank"
                result["p_value"] = float(stat.pvalue)
            result["significant"] = bool(result["p_value"] < 0.05)
            result["normality_p_value"] = float(normality_p)
        except Exception as exc:
            result["error"] = str(exc)

        significance_results.append(result)

    significance_obj = {
        "generated_at": utc_now(),
        "output_root": str(output_root),
        "metric": args.metric,
        "alpha": 0.05,
        "results": significance_results,
    }
    with (summary_dir / "significance_tests.json").open("w", encoding="utf-8") as f:
        json.dump(significance_obj, f, indent=2)

    # Manuscript-facing compact table
    manuscript_rows = []
    for row in aggregate_rows:
        manuscript_rows.append(
            {
                "Tag": row["tag"],
                "Model": row["model_name"],
                "Accuracy (mean ± std)": format_mean_std(row.get("accuracy_mean"), row.get("accuracy_std"), 4),
                "Precision (mean ± std)": format_mean_std(row.get("macro_precision_mean"), row.get("macro_precision_std"), 4),
                "Recall (mean ± std)": format_mean_std(row.get("macro_recall_mean"), row.get("macro_recall_std"), 4),
                "Macro-F1 (mean ± std)": format_mean_std(row.get("macro_f1_mean"), row.get("macro_f1_std"), 4),
                "MCC (mean ± std)": format_mean_std(row.get("mcc_mean"), row.get("mcc_std"), 4),
                "Macro-FPR (mean ± std)": format_mean_std(row.get("macro_fpr_mean"), row.get("macro_fpr_std"), 6),
                "Completed Seeds": row.get("completed_seed_count", 0),
            }
        )
    pd.DataFrame(manuscript_rows).to_csv(summary_dir / "manuscript_table_model_comparison.csv", index=False)

    print(f"Aggregate metrics written to: {summary_dir}")


if __name__ == "__main__":
    main()
