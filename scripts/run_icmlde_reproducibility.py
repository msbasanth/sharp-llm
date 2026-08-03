"""Run ICMLDE five-seed reproducibility experiments with restartable status tracking.

This script runs the paper-scoped encoder models over a fixed Juliet split,
one seed at a time, and writes status to disk so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICMLDE reproducibility matrix")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--output-root", default=None, help="Override ICMLDE output root")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list")
    parser.add_argument("--variants", default=None, help="Comma-separated model variants to run")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to next run on failure")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def normalize_seeds(seed_arg: str | None, cfg_seeds: list[int]) -> list[int]:
    if seed_arg:
        return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]
    return [int(x) for x in cfg_seeds]


def normalize_variants(variant_arg: str | None) -> set[str] | None:
    if not variant_arg:
        return None
    parsed = {x.strip() for x in variant_arg.split(",") if x.strip()}
    return parsed if parsed else None


def load_or_init_json(path: Path, default_obj: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default_obj


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def key_for(seed: int, variant: str) -> str:
    return f"seed_{seed}:{variant}"


def required_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "best_checkpoint": run_dir / "checkpoints" / "best.pt",
        "latest_checkpoint": run_dir / "checkpoints" / "latest.pt",
        "epoch_metrics": run_dir / "logs" / "epoch_metrics.json",
        "metrics": run_dir / "evaluation" / "metrics.json",
        "classification_report": run_dir / "evaluation" / "classification_report.txt",
        "confusion_pairs": run_dir / "evaluation" / "confusion_pairs.csv",
    }


def training_artifacts_complete(run_dir: Path) -> bool:
    train_artifacts = [
        run_dir / "checkpoints" / "best.pt",
        run_dir / "checkpoints" / "latest.pt",
        run_dir / "logs" / "epoch_metrics.json",
    ]
    return all(p.exists() for p in train_artifacts)


def is_run_complete(run_dir: Path) -> bool:
    for artifact_path in required_artifacts(run_dir).values():
        if not artifact_path.exists():
            return False
    return True


def run_cmd(cmd: list[str], dry_run: bool) -> tuple[int, str | None]:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return 0, None

    proc = subprocess.run(cmd, text=True)
    if proc.returncode == 0:
        return 0, None
    return proc.returncode, f"Command failed ({proc.returncode}): {' '.join(cmd)}"


def update_status(
    status_obj: dict[str, Any],
    run_key: str,
    *,
    tag: str,
    seed: int,
    model_name: str,
    model_id: str,
    variant: str,
    run_dir: Path,
    state: str,
    step: str,
    error: str | None = None,
) -> None:
    entry = status_obj.setdefault("runs", {}).setdefault(run_key, {})
    entry.update(
        {
            "tag": tag,
            "seed": seed,
            "model_name": model_name,
            "model_id": model_id,
            "variant": variant,
            "status": state,
            "last_completed_step": step,
            "output_dir": str(run_dir),
            "updated_at": utc_now(),
            "error": error,
        }
    )
    status_obj["updated_at"] = utc_now()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    icmlde_cfg = cfg.get("icmlde", {})

    output_root = Path(args.output_root or icmlde_cfg.get("output_root", "outputs/icmlde2026/juliet118"))
    seeds = normalize_seeds(args.seeds, icmlde_cfg.get("seeds", [42, 43, 44, 45, 46]))
    variants_filter = normalize_variants(args.variants)

    models: list[dict[str, str]] = icmlde_cfg.get("models", [])
    if not models:
        raise ValueError("No icmlde.models configured in config.yaml")

    train_path = icmlde_cfg.get("train_path", cfg.get("train_path"))
    test_path = icmlde_cfg.get("test_path", cfg.get("test_path"))
    label_map_path = icmlde_cfg.get("label_map_path", cfg.get("label_map_path"))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    status_path = output_root / "status.json"

    manifest_obj = {
        "name": "ICMLDE five-seed reproducibility",
        "created_at": utc_now(),
        "config": args.config,
        "output_root": str(output_root),
        "split_seed": int(icmlde_cfg.get("split_seed", cfg.get("seed", 42))),
        "seeds": seeds,
        "train_path": train_path,
        "test_path": test_path,
        "label_map_path": label_map_path,
        "models": models,
        "tag_prefix": "ICMLDE:",
    }
    if not manifest_path.exists():
        save_json(manifest_path, manifest_obj)

    status_obj = load_or_init_json(
        status_path,
        {
            "name": "ICMLDE run status",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "runs": {},
        },
    )

    for seed in seeds:
        for model in models:
            model_name = model["name"]
            model_id = model["model_id"]
            variant = model["variant"]
            if variants_filter and variant not in variants_filter:
                continue
            tag = model.get("tag", f"ICMLDE:Juliet118:{model_name}")
            run_dir = output_root / f"seed_{seed}" / variant
            run_key = key_for(seed, variant)

            if is_run_complete(run_dir):
                update_status(
                    status_obj,
                    run_key,
                    tag=tag,
                    seed=seed,
                    model_name=model_name,
                    model_id=model_id,
                    variant=variant,
                    run_dir=run_dir,
                    state="done",
                    step="aggregate-ready",
                    error=None,
                )
                save_json(status_path, status_obj)
                print(f"SKIP complete: {run_key}")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            eval_dir = run_dir / "evaluation"
            best_ckpt = checkpoint_dir / "best.pt"

            should_run_train = True
            if training_artifacts_complete(run_dir) and not is_run_complete(run_dir):
                should_run_train = False

            if should_run_train:
                update_status(
                    status_obj,
                    run_key,
                    tag=tag,
                    seed=seed,
                    model_name=model_name,
                    model_id=model_id,
                    variant=variant,
                    run_dir=run_dir,
                    state="running",
                    step="train",
                    error=None,
                )
                save_json(status_path, status_obj)

                train_cmd = [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--config",
                    args.config,
                    "--model",
                    model_id,
                    "--seed",
                    str(seed),
                    "--train-path",
                    train_path,
                    "--test-path",
                    test_path,
                    "--label-map-path",
                    label_map_path,
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--checkpoint-dir-is-model-dir",
                    "--log-dir",
                    str(log_dir),
                    "--log-dir-is-model-dir",
                    "--run-tag",
                    tag,
                ]
                print(f"RUN train: {run_key}")
                rc, err = run_cmd(train_cmd, args.dry_run)
                if rc != 0:
                    update_status(
                        status_obj,
                        run_key,
                        tag=tag,
                        seed=seed,
                        model_name=model_name,
                        model_id=model_id,
                        variant=variant,
                        run_dir=run_dir,
                        state="failed",
                        step="train",
                        error=err,
                    )
                    save_json(status_path, status_obj)
                    if not args.continue_on_error:
                        raise SystemExit(1)
                    continue

                if args.dry_run:
                    update_status(
                        status_obj,
                        run_key,
                        tag=tag,
                        seed=seed,
                        model_name=model_name,
                        model_id=model_id,
                        variant=variant,
                        run_dir=run_dir,
                        state="missing",
                        step="planned",
                        error=None,
                    )
                    save_json(status_path, status_obj)
                    continue

                if not best_ckpt.exists():
                    err = f"Missing checkpoint after training: {best_ckpt}"
                    update_status(
                        status_obj,
                        run_key,
                        tag=tag,
                        seed=seed,
                        model_name=model_name,
                        model_id=model_id,
                        variant=variant,
                        run_dir=run_dir,
                        state="failed",
                        step="train",
                        error=err,
                    )
                    save_json(status_path, status_obj)
                    if not args.continue_on_error:
                        raise SystemExit(1)
                    continue
            else:
                print(f"SKIP train (artifacts present): {run_key}")

            update_status(
                status_obj,
                run_key,
                tag=tag,
                seed=seed,
                model_name=model_name,
                model_id=model_id,
                variant=variant,
                run_dir=run_dir,
                state="running",
                step="eval",
                error=None,
            )
            save_json(status_path, status_obj)

            eval_cmd = [
                sys.executable,
                "-m",
                "src.evaluate",
                "--config",
                args.config,
                "--model",
                model_id,
                "--checkpoint",
                str(best_ckpt),
                "--test-path",
                test_path,
                "--label-map-path",
                label_map_path,
                "--output-dir",
                str(eval_dir),
                "--seed",
                str(seed),
                "--experiment-tag",
                tag,
            ]
            print(f"RUN eval: {run_key}")
            rc, err = run_cmd(eval_cmd, args.dry_run)
            if rc != 0:
                update_status(
                    status_obj,
                    run_key,
                    tag=tag,
                    seed=seed,
                    model_name=model_name,
                    model_id=model_id,
                    variant=variant,
                    run_dir=run_dir,
                    state="failed",
                    step="eval",
                    error=err,
                )
                save_json(status_path, status_obj)
                if not args.continue_on_error:
                    raise SystemExit(1)
                continue

            run_state = "done" if is_run_complete(run_dir) else "missing"
            last_step = "aggregate-ready" if run_state == "done" else "eval"
            update_status(
                status_obj,
                run_key,
                tag=tag,
                seed=seed,
                model_name=model_name,
                model_id=model_id,
                variant=variant,
                run_dir=run_dir,
                state=run_state,
                step=last_step,
                error=None if run_state == "done" else "Missing one or more required artifacts",
            )
            save_json(status_path, status_obj)
            print(f"DONE {run_state}: {run_key}")

    print("ICMLDE reproducibility run complete.")


if __name__ == "__main__":
    main()
