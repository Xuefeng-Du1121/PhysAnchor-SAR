"""Audit development calibration and freeze the corresponding formal SOTA plan.

Calibration is a numerical-stability gate only.  It is deliberately forbidden
from selecting methods by outcome or contributing values to paper tables.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .protocol import file_sha
from . import run_formal_study


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sota-calibration-stability-gate-v1"


def _option(command, name):
    if name not in command or command.index(name) + 1 >= len(command):
        raise ValueError(f"command lacks required option {name}")
    return command[command.index(name) + 1]


def _resolve(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _assert_finite(value, label="value"):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"non-finite tensor in {label}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite array in {label}")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite number in {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{label}[{index}]")


def _read_json_lines(path):
    if not path.is_file():
        raise ValueError(f"missing JSONL file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def _validate_events(run_dir, jobs):
    rows = _read_json_lines(run_dir / "events.jsonl")
    if rows[-1].get("status") != "complete":
        raise ValueError("calibration event stream does not end in complete")
    if any(row.get("status") == "job-failed" for row in rows):
        raise ValueError("calibration event stream contains a failed job")
    success = set()
    for row in rows:
        if (row.get("status") == "finished" and row.get("returncode") == 0 and
                row.get("stage") in ("train", "evaluate")):
            success.add((row.get("job"), row["stage"]))
        if row.get("status") == "skipped-complete":
            success.add((row.get("job"), row.get("stage")))
    expected = {(job["id"], stage) for job in jobs
                for stage in ("train", "evaluate")}
    missing = sorted(expected - success)
    if missing:
        raise ValueError(f"calibration stages lack successful events: {missing}")
    return rows


def audit_calibration(plan_path, required_methods, expected_seeds):
    plan_path = Path(plan_path).resolve()
    if plan_path.name != "plan.json" or not plan_path.is_file():
        raise ValueError("calibration plan must be an existing plan.json")
    root = plan_path.parents[2]
    if root != ROOT:
        raise ValueError("calibration plan must belong to this project clone")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "formal-study-plan-v1":
        raise ValueError("unsupported calibration plan schema")
    if plan.get("scope") != "development-calibration":
        raise ValueError("gate accepts development-calibration scope only")
    if plan.get("evaluation_split") != "validation" or plan.get("test_unlocked"):
        raise ValueError("calibration must use locked validation only")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("calibration plan has no jobs")
    methods = {job.get("method") for job in jobs}
    seeds = {job.get("seed") for job in jobs}
    if methods != set(required_methods):
        raise ValueError("calibration method set differs from the preregistered set")
    if seeds != set(expected_seeds):
        raise ValueError("calibration seed set differs from the preregistered set")
    counts = Counter((job.get("method"), job.get("seed")) for job in jobs)
    expected_pairs = {(method, seed) for method in required_methods
                      for seed in expected_seeds}
    if set(counts) != expected_pairs or any(count != 1 for count in counts.values()):
        raise ValueError("calibration does not contain exactly one job per method/seed")

    run_formal_study._assert_frozen(plan)
    events = _validate_events(plan_path.parent, jobs)
    audited = []
    for job in jobs:
        for stage in ("train", "evaluate"):
            run_formal_study._validate_stage_artifact(plan, job, stage)
        train_dir = _resolve(root, _option(job["train"], "--out"))
        eval_dir = _resolve(root, _option(job["evaluate"], "--out"))
        updates = _read_json_lines(train_dir / "updates.jsonl")
        _assert_finite(updates, f"{job['id']}.updates")
        last_steps = updates[-1].get("environment_steps")
        if not isinstance(last_steps, int) or last_steps > plan["steps_per_training_seed"]:
            raise ValueError(f"invalid update history for {job['id']}")
        if job.get("trainer_family") == "offpolicy-ctde":
            if max(row.get("gradient_updates", 0) for row in updates) < 1:
                raise ValueError(f"off-policy calibration performed no updates: {job['id']}")
        elif last_steps != plan["steps_per_training_seed"]:
            raise ValueError(f"on-policy calibration ended at wrong step: {job['id']}")
        checkpoint_path = train_dir / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _assert_finite(checkpoint, f"{job['id']}.checkpoint")
        summary_path = eval_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _assert_finite(summary, f"{job['id']}.summary")
        audited.append({
            "id": job["id"], "method": job["method"], "seed": job["seed"],
            "trainer_family": job.get("trainer_family"),
            "checkpoint_sha256": file_sha(checkpoint_path),
            "summary_sha256": file_sha(summary_path),
            "gradient_updates": checkpoint.get("gradient_updates"),
        })
    return {
        "schema": SCHEMA,
        "status": "passed-numerical-stability-only",
        "created_unix": time.time(),
        "calibration_plan": str(plan_path.relative_to(root)),
        "calibration_plan_sha256": file_sha(plan_path),
        "event_stream_sha256": file_sha(plan_path.parent / "events.jsonl"),
        "event_count": len(events),
        "job_count": len(audited),
        "methods": sorted(methods),
        "seeds": sorted(seeds),
        "scope_warning": (
            "Development calibration proves execution and finite outputs only. "
            "Its metrics are forbidden from confirmatory tables, figures, method "
            "selection, or claims of comparative performance."),
        "offpolicy_warning": (
            "The 5120-step calibration has only post-warmup stability coverage; "
            "it is not evidence of convergence or hyperparameter quality."),
        "jobs": audited,
    }, plan


def freeze_formal_plan(calibration_plan_path, output_name, required_methods,
                       calibration_seeds, formal_seeds, episodes_per_seed,
                       created_unix=None):
    report, calibration_plan = audit_calibration(
        calibration_plan_path, required_methods, calibration_seeds)
    calibration_path = Path(calibration_plan_path).resolve()
    gate_path = calibration_path.parent / "calibration-gate.json"
    if gate_path.exists():
        raise ValueError("refusing to overwrite existing calibration gate")
    gate_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    inventory_value = _option(calibration_plan["jobs"][0]["train"], "--inventory")
    inventory_path = _resolve(ROOT, inventory_value)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    validation_count = sum(row.get("split") == "validation"
                           for row in inventory.get("scenarios", []))
    if validation_count < 1 or episodes_per_seed % validation_count:
        raise ValueError("episodes per seed must divide evenly across validation scenarios")
    args = Namespace(
        out=output_name, inventory=str(inventory_path), transport_aux=None,
        scope="formal-paper", steps=204800, rollout=256,
        seeds=list(formal_seeds), methods=list(required_methods),
        eval_split="validation", unlock_test=False,
        episodes_per_scenario=episodes_per_seed // validation_count,
        gae_lambda=.95, device="cuda")
    formal_plan, output = run_formal_study.build_plan(args)
    if output.exists():
        raise ValueError("refusing to overwrite formal output directory")
    gate_rel = str(gate_path.relative_to(ROOT))
    calibration_rel = str(calibration_path.relative_to(ROOT))
    formal_plan["input_sha256"][calibration_rel] = file_sha(calibration_path)
    formal_plan["input_sha256"][gate_rel] = file_sha(gate_path)
    formal_plan["promotion_gate"] = {
        "schema": SCHEMA, "path": gate_rel,
        "sha256": file_sha(gate_path),
        "policy": "stability-only; no outcome-based method selection",
    }
    if created_unix is not None:
        formal_plan["created_unix"] = created_unix
    output.mkdir(parents=False)
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(formal_plan, indent=2), encoding="utf-8")
    return report, formal_plan, plan_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--required-method", action="append", required=True)
    parser.add_argument("--calibration-seed", action="append", type=int, required=True)
    parser.add_argument("--formal-seed", action="append", type=int, required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    args = parser.parse_args(argv)
    if args.episodes_per_seed < 1:
        parser.error("episodes-per-seed must be positive")
    report, plan, path = freeze_formal_plan(
        args.calibration_plan, args.out, args.required_method,
        args.calibration_seed, args.formal_seed, args.episodes_per_seed)
    print(json.dumps({
        "status": report["status"], "calibration_jobs": report["job_count"],
        "formal_jobs": len(plan["jobs"]), "formal_plan": str(path),
        "formal_plan_sha256": file_sha(path),
    }), flush=True)


if __name__ == "__main__":
    main()
