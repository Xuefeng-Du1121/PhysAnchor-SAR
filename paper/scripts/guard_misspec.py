"""Guard: eval_misspec with default settings must reproduce official per-episode records."""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json, subprocess, sys, shutil
import numpy as np
PR = str(_paths.root("peva-phys-residual"))
CP = PR + "/runs/complete-paper-v1"
PY = sys.executable


def official_records(summary, method=None):
    s = json.load(open(summary))
    if method:
        s = s["summaries"][method] if "summaries" in s else s
        recs = json.load(open(summary)).get("records", {})
        recs = recs.get(method, recs) if isinstance(recs, dict) else recs
    else:
        recs = s["records"]
    return recs


def run(args, out):
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([PY, "tools/eval_misspec.py", *args, "--split", "test", "--episodes-per-scenario", "2",
                    "--out", out], check=True, cwd=PR, stdout=subprocess.DEVNULL)
    return [json.loads(l) for l in open(out + "/episodes.jsonl")]


def compare(name, mine, recs, per_scen):
    # official order: case-major, 50 repeats per case
    off = [recs[c * per_scen + r] for c in range(2) for r in range(2)]
    a = [m["success_rate"] for m in mine]
    b = [o.get("success_rate", o.get("info", {}).get("success_rate")) for o in off]
    ok = np.allclose(a, b)
    print(f"{name}: mine {a} official {b} -> {'OK' if ok else 'MISMATCH'}")
    return ok


ok = True
s = json.load(open(f"{PR}/runs/complete-paper-v2/test/anchor-zero-residual/summary.json"))
print("official record keys:", list(s["records"][0].keys())[:12])
if "--skip-zero" not in sys.argv:
    ok &= compare("zero-residual", run(["--checkpoint", f"{PR}/runs/complete-paper-v2/train/anchor-zero-residual/checkpoint.pt"], "/tmp/gz"),
                  s["records"], 50)
c = json.load(open(f"{CP}/test/classical/summary.json"))
print("classical keys:", list(c.keys()))
recs = c.get("records")
for m in ("pvf-greedy", "cbba"):
    r = recs[m] if isinstance(recs, dict) else [x for x in recs if x.get("method") == m]
    per = c["episodes_per_method"] // len(c["scenarios"])
    print(m, "official episodes per scenario:", per)
    ok &= compare(m, run(["--classical", m], "/tmp/g" + m), r, per)
print("GUARD", "PASS" if ok else "FAIL")
