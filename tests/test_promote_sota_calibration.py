import json
from pathlib import Path
import tempfile
import unittest

import torch

from peva_sim import promote_sota_calibration as promotion


class PromoteSotaCalibrationTest(unittest.TestCase):
    def test_finite_audit_rejects_nan_nested_tensor(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            promotion._assert_finite({"model": [torch.tensor([1.0, float("nan")])]})

    def test_event_gate_requires_every_successful_stage(self):
        jobs = [{"id": "maddpg-seed101"}]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [
                {"status": "finished", "job": "maddpg-seed101",
                 "stage": "train", "returncode": 0},
                {"status": "complete"},
            ]
            (run / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lack successful events"):
                promotion._validate_events(run, jobs)

    def test_event_gate_rejects_historical_job_failure(self):
        jobs = [{"id": "maddpg-seed101"}]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [
                {"status": "job-failed", "job": "maddpg-seed101"},
                {"status": "skipped-complete", "job": "maddpg-seed101",
                 "stage": "train"},
                {"status": "skipped-complete", "job": "maddpg-seed101",
                 "stage": "evaluate"},
                {"status": "complete"},
            ]
            (run / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed job"):
                promotion._validate_events(run, jobs)

    def test_event_gate_accepts_complete_matrix(self):
        jobs = [{"id": "maddpg-seed101"}, {"id": "happo-seed101"}]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = []
            for job in jobs:
                for stage in ("train", "evaluate"):
                    rows.append({"status": "finished", "job": job["id"],
                                 "stage": stage, "returncode": 0})
            rows.append({"status": "complete"})
            (run / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertEqual(promotion._validate_events(run, jobs), rows)


if __name__ == "__main__":
    unittest.main()
