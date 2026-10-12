"""Collect every evaluation used by the revision-3 tables and figures into one JSON.

Three on-disk formats are read, nothing is typed in:
  formal   : summary.json with per-episode "records" (formal_mappo / ic3net / peva evaluators)
  misspec  : summary.json + episodes.jsonl (tools/eval_misspec.py)
  classical: complete-paper-v1/<split>/classical/summary.json, records[method]
Per method and split we keep one entry per seed: success, complete rescue, TTD, survival, episodes,
and per-scenario success (used for the OOD current-speed stratification).
Output: runs/complete-paper-v4/results_r3.json
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np

PR = str(_paths.root("peva-phys-residual"))
V5 = str(_paths.root("peva-gen-final-v5"))
V1 = PR + "/runs/campaign-v1/eval"
CP = PR + "/runs/complete-paper-v1"
E3 = PR + "/runs/complete-paper-v3/eval"
E4 = PR + "/runs/complete-paper-v4/eval"
ST3 = PR + "/runs/stage3-v1/validation"
ST2 = PR + "/runs/stage2-anchor-mappo-v1/validation"


def seed_of(d):
    b = os.path.basename(d.rstrip("/"))
    return b.rsplit("-seed", 1)[-1] if "-seed" in b else "0"


def from_records(recs):
    s = np.array([r["info"]["success_rate"] for r in recs], float)
    t = np.array([r["info"]["ttd_all"] for r in recs], float)
    v = np.array([r["info"].get("survival_score", np.nan) for r in recs], float)
    per = {}
    for r in recs:
        per.setdefault(r["scenario_id"], []).append(r["info"]["success_rate"])
    return dict(succ=float(s.mean()), all=float(np.mean(s >= 1 - 1e-9)), ttd=float(t.mean()),
                surv=float(np.nanmean(v)), n=len(recs), per={k: float(np.mean(x)) for k, x in per.items()})


def read(fmt, path):
    if fmt == "classical":
        d, m = path.split(":")
        p = Path(d) / "summary.json"
        if not p.exists():
            return None
        s = json.load(open(p))
        return from_records(s["records"][m]) if m in s.get("records", {}) else None
    p = Path(path) / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    if fmt == "formal":
        if s.get("records") and "info" in s["records"][0]:
            return from_records(s["records"])
        g = lambda k: s[k]["mean"] if isinstance(s.get(k), dict) else s.get(k)  # noqa: E731
        return dict(succ=g("success_rate"), all=None, ttd=g("ttd_all"), surv=g("survival_score"),
                    n=s.get("episodes"), per={})
    rows = [json.loads(x) for x in open(Path(path) / "episodes.jsonl")]
    per = {}
    for r in rows:
        per.setdefault(r["scenario_id"], []).append(r["success_rate"])
    return dict(succ=float(np.mean([r["success_rate"] for r in rows])),
                all=float(np.mean([r["all_found"] for r in rows])),
                ttd=float(np.mean([r["ttd_all"] for r in rows])),
                surv=float(np.mean([r["survival_score"] for r in rows])), n=len(rows),
                per={k: float(np.mean(x)) for k, x in per.items()})


def gather(fmt, pats):
    """All seeds matching the patterns; the first pattern that provides a seed wins."""
    out = {}
    if fmt == "classical":
        r = read(fmt, pats[0])
        return [dict(seed="0", **r)] if r else []
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            sd = seed_of(d)
            if sd in out:
                continue
            r = read(fmt, d)
            if r:
                out[sd] = dict(seed=sd, src=d, **r)
    return [out[k] for k in sorted(out, key=lambda x: int(x) if x.isdigit() else 0)]


def first(*alts):
    """First alternative (fmt, patterns) that yields any result."""
    for fmt, pats in alts:
        r = gather(fmt, pats)
        if r:
            return r
    return []


PLAN_OLD = ("pvf-greedy", "cbba", "bayesian-information-gain", "greedy-probability", "lawnmower-coverage",
            "oracle-target")
PLAN_NEW = ("mpc-horizon", "gradient-ascent", "expanding-square", "sector-search", "spiral-search", "pso-search",
            "aco-search", "voronoi-coverage", "levy-random-walk")
SPLITS = ("test", "validation", "ood")


def planner(p):
    r = {}
    for sp in SPLITS:
        alts = [("misspec", [f"{E4}/{sp}-{p}", f"{E3}/{sp}-{p}"])]
        if sp != "ood" and p in PLAN_OLD:
            alts.append(("classical", [f"{CP}/{sp}/classical:{p}"]))
        r[sp] = first(*alts)
    return r


def learned(test, val, ood):
    return dict(test=first(*test), validation=first(*val), ood=first(*ood))


def collect():
    """{method: {split: [per-seed dict]}} for every method of the paper (main table, ablations, PEVA ablations)."""
    M = {}
    for p in PLAN_OLD + PLAN_NEW:
        M[p] = planner(p)
    M["phys30"] = {sp: first(("misspec", [f"{E4}/{sp}-phys30"])) for sp in SPLITS}
    M["phys15"] = dict(test=first(("misspec", [f"{E3}/zero-test-ang15"])),
                       validation=first(("misspec", [f"{E3}/zero-validation-ang15"])), ood=[])
    M["anchor30"] = learned([("formal", [f"{V1}/test/anchor-ang30-seed*", f"{CP}/test/anchor-ang30-seed*"])],
                            [("formal", [f"{V1}/validation/anchor-ang30-seed*", f"{CP}/validation/anchor-ang30-seed*"])],
                            [("misspec", [f"{E3}/ood-anchor-ang30-seed*"])])
    M["anchor15"] = learned([("formal", [f"{V1}/test/anchor-mappo-seed*", f"{CP}/test/anchor-mappo-seed*"])],
                            [("formal", [f"{V1}/validation/anchor-mappo-seed*", f"{CP}/validation/anchor-mappo-seed*",
                                         f"{ST2}/anchor-mappo-seed*"])],
                            [("misspec", [f"{E3}/ood-anchor-mappo-seed*"])])
    M["mappo"] = learned([("formal", [f"{V5}/runs/campaign-v3/test/mappo-seed*"])],
                         [("formal", [f"{V5}/runs/eaai-final-v5b-mappo-seeds0-9/validation/mappo-seed*"])],
                         [("misspec", [f"{E4}/ood-mappo-seed*"])])
    M["mappo-long"] = learned([("misspec", [f"{E4}/test-longmappo-seed*"])],
                              [("misspec", [f"{E4}/validation-longmappo-seed*"])],
                              [("misspec", [f"{E4}/ood-longmappo-seed*"])])
    M["commnet"] = learned([("formal", [f"{V5}/runs/campaign-v3/test/commnet-mappo-seed*"])],
                           [("formal", [f"{V5}/runs/eaai-final-v5b-commnet-seeds0-9/validation/commnet-mappo-seed*"])],
                           [("misspec", [f"{E4}/ood-commnet-seed*"])])
    M["happo"] = learned([("formal", [f"{V1}/test/happo-seed*"])], [("formal", [f"{V1}/validation/happo-seed*"])],
                         [("formal", [f"{E4}/ood-happo-seed*"])])
    _rest(M)
    return M


def _rest(M):
    M["ic3net"] = learned([("formal", [f"{V1}/test/ic3net-seed*"])], [("formal", [f"{V1}/validation/ic3net-seed*"])],
                          [("misspec", [f"{E3}/ood-ic3net-seed*"])])
    M["ippo"] = learned([("misspec", [f"{E4}/test-ippo-seed*"])], [("misspec", [f"{E4}/validation-ippo-seed*"])],
                        [("misspec", [f"{E4}/ood-ippo-seed*"])])
    M["pvf-mappo"] = learned([("formal", [f"{V5}/runs/campaign-v3/test/pvf-mappo-seed*"])],
                             [("formal", [f"{V5}/runs/eaai-final-v5b-pvf-mappo-seeds0-9/validation/pvf-mappo-seed*"])],
                             [("misspec", [f"{E3}/ood-pvf-mappo-seed*"])])
    M["pvf-commnet"] = learned([("formal", [f"{V5}/runs/campaign-v3/test/pvf-commnet-mappo-seed*"])],
                               [("formal", [f"{V5}/runs/eaai-final-v5b-pvf-commnet-mappo-seeds0-9/validation/"
                                            "pvf-commnet-mappo-seed*"])], [("misspec", [f"{E4}/ood-pvfcommnet-seed*"])])
    M["pvf-happo"] = learned([("formal", [f"{V1}/test/happo-pvf-seed*"])],
                             [("formal", [f"{V1}/validation/happo-pvf-seed*"])],
                             [("formal", [f"{E4}/ood-happo-pvf-seed*"])])
    M["pvf-ic3net"] = learned([("formal", [f"{V1}/test/ic3net-pvf-seed*"])],
                              [("formal", [f"{V1}/validation/ic3net-pvf-seed*"])],
                              [("misspec", [f"{E4}/ood-pvfic3net-seed*"])])
    M["shaping"] = dict(test=gather("misspec", [f"{E3}/test-shape1-noanchor-seed*", f"{PR}/runs/complete-paper-v3/"
                                                "eval-prio/test-shape1-noanchor-seed*", f"{E4}/test-shape1-noanchor-seed*"]),
                        validation=gather("formal", [f"{ST3}/shape1-noanchor-seed*"])
                        + gather("misspec", [f"{E4}/validation-shape1-noanchor-seed*"]),
                        ood=gather("misspec", [f"{E3}/ood-shape1-noanchor-seed*"]))
    M["bc-finetune"] = learned([("misspec", [f"{E4}/test-bcft-seed*"])],
                               [("misspec", [f"{E4}/validation-bcft-seed*"])],
                               [("misspec", [f"{E4}/ood-bcft-seed*"])])
    M["bc-init"] = learned([("misspec", [f"{E4}/test-bcinit-seed*"])],
                           [("misspec", [f"{E4}/validation-bcinit-seed*"])],
                           [("misspec", [f"{E4}/ood-bcinit-seed*"])])
    M["peva-gen"] = learned([("formal", [f"{V5}/runs/peva-test-eval-v2/eval/peva-gen-seed*"])],
                            [("formal", [f"{V5}/runs/eaai-final-v5b-peva-gen-seeds0-9/validation/peva-gen-seed*"])],
                            [("formal", [f"{V5}/runs/peva-ood-r6/peva-gen-seed*"])])
    # campaign K: anchor direction as input only (observation as the anchored actor, no anchor term in the action)
    M["anchor-input"] = learned([("misspec", [f"{E4}/test-ainput-seed*"])],
                                [("misspec", [f"{E4}/validation-ainput-seed*"])],
                                [("misspec", [f"{E4}/ood-ainput-seed*"])])
    for k, n in (("peva-nodiff", "peva-deterministic-belief"), ("peva-norsr", "peva-no-rsr"),
                 ("peva-beta0", "peva-beta0-uniform-no-pvf-context"), ("peva-nocov", "peva-no-coverage"),
                 ("peva-nophys", "peva-no-physics-context")):
        M[k] = learned([("formal", [f"{V5}/runs/peva-test-eval-v2/eval/{n}-seed*"])],
                       [("formal", [f"{V5}/runs/eaai-final-v5b-peva-ablation-seeds0-9/validation/{n}-seed*"])], [])
    M["anchor-2upd"] = learned([("formal", [f"{PR}/runs/complete-paper-v2/test/anchor-2upd-seed*"])],
                               [("formal", [f"{PR}/runs/anchor-untrained-128-formal-eval"])],
                               [("misspec", [f"{E3}/ood-anchor-2upd-seed*"])])
    M["anchor0"] = learned([("formal", [f"{V1}/test/anchor-ang0-seed*", f"{CP}/test/anchor-ang0-seed*"])],
                           [("formal", [f"{V1}/validation/anchor-ang0-seed*", f"{CP}/validation/anchor-ang0-seed*"])], [])
    for k in ("str2", "str6", "str8"):
        M[f"anchor-{k}"] = learned([("formal", [f"{V1}/test/anchor-{k}-seed*", f"{CP}/test/anchor-{k}-seed*"])],
                                   [("formal", [f"{V1}/validation/anchor-{k}-seed*", f"{CP}/validation/anchor-{k}-seed*"])],
                                   [])
    CJ = PR + "/runs/campaign-j/eval"           # campaign J: trained anchor at theta 45/60/90 and alpha 1/12
    for k in ("ang45", "ang60", "ang90", "str1", "str12"):
        M[f"anchor-{k}"] = learned([("formal", [f"{CJ}/test/anchor-{k}-seed*"])],
                                   [("formal", [f"{CJ}/validation/anchor-{k}-seed*"])], [])
    for k in ("shape0.5-anchor", "shape1-anchor-s4", "shape2.0-anchor", "shape4.0-anchor"):
        M[k] = learned([("formal", [f"{V1}/test/{k}-seed*", f"{CP}/test/{k}-seed*"])],
                       [("formal", [f"{V1}/validation/{k}-seed*", f"{CP}/validation/{k}-seed*", f"{ST3}/{k}-seed*"])], [])


def summary_line(v):
    cells = []
    for sp in SPLITS:
        r = v.get(sp) or []
        if r:
            s = np.array([x["succ"] for x in r])
            a = np.array([x["all"] for x in r if x["all"] is not None])
            cells.append(f"{s.mean():.3f}/{(a.mean() if len(a) else float('nan')):.3f} n={len(r):2d} "
                         f"e={r[0]['n'] or 0:3d}")
        else:
            cells.append("—")
    return "".join(f"{c:>26s}" for c in cells)


if __name__ == "__main__":
    M = collect()
    out = Path(PR) / "runs/complete-paper-v4/results_r3.json"
    out.write_text(json.dumps(M, indent=1))
    print(f"{'method':26s}" + "".join(f"{sp:>26s}" for sp in SPLITS))
    for k, v in M.items():
        print(f"{k:26s}" + summary_line(v))
    print("->", out, file=sys.stderr)
