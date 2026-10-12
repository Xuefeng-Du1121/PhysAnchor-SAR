"""Main results table (LaTeX) and paired statistics, from results_r3.json and the raw episode files.

Output: runs/complete-paper-v4/tables-r3/tab_main.tex and paired_stats.json.
Paired test: every episode of the anchored policy (10 seeds) is paired with the episode of a deterministic
planner on the same scenario and seed; we report the mean paired difference in success with a 95%
bootstrap interval (2000 resamples) and the win / tie / loss shares.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
from pathlib import Path
import numpy as np
from results_r3 import collect

M = collect()
((_paths.root("peva-phys-residual") / "runs/complete-paper-v4/results_r3.json")
 .write_text(json.dumps(M, indent=1)))

PR = _paths.root("peva-phys-residual")
OUT = PR / "runs/complete-paper-v4/tables-r3"
OUT.mkdir(parents=True, exist_ok=True)
SPL = ("test", "validation", "ood")

GROUPS = [
    ("Physics written into the action (ours)", [
        ("anchor30", r"Physics anchor + residual, $\theta=30^{\circ}$"),
        ("anchor15", r"Physics anchor + residual, $\theta=15^{\circ}$"),
        ("phys30", r"Physics anchor only, $\theta=30^{\circ}$ (no network)")]),
    ("Belief-driven planners", [
        ("pvf-greedy", "PVF-greedy \\citep{zhang2024vrpm}"), ("cbba", "CBBA \\citep{cbba}"),
        ("bayesian-information-gain", "Bayesian info. gain \\citep{yao2022fusion}"),
        ("greedy-probability", "Greedy probability"),
        ("mpc-horizon", "Receding-horizon MPC \\citep{tian2026mpc}"), ("gradient-ascent", "Value-gradient ascent")]),
    ("IAMSAR and geometric search patterns", [
        ("sector-search", "Sector search \\citep{iamsar}"), ("expanding-square", "Expanding square \\citep{iamsar}"),
        ("lawnmower-coverage", "Parallel track \\citep{moshagen2025wadden}"), ("spiral-search", "Archimedean spiral")]),
    ("Swarm and stochastic search", [
        ("aco-search", "Ant colony \\citep{yao2026aco}"), ("pso-search", "Particle swarm \\citep{jiang2026rlpso}"),
        ("voronoi-coverage", "Voronoi coverage \\citep{huang2024voronoi}"),
        ("levy-random-walk", "L\\'evy walk \\citep{levy}")]),
    ("MARL, no physics", [
        ("mappo", "MAPPO \\citep{mappo}"), ("mappo-long", r"MAPPO, $5\times$ budget"), ("ippo", "IPPO \\citep{ippo}"),
        ("commnet", "CommNet-MAPPO \\citep{commnet}"), ("happo", "HAPPO \\citep{zhong2024harl}"),
        ("ic3net", "IC3Net \\citep{ic3net}")]),
    ("MARL, physics value field as input", [
        ("pvf-mappo", "PVF + MAPPO \\citep{hou2024marl}"), ("pvf-commnet", "PVF + CommNet-MAPPO"),
        ("pvf-happo", "PVF + HAPPO"), ("pvf-ic3net", "PVF + IC3Net"),
        ("anchor-input", r"Anchor direction as input, $\theta=30^{\circ}$")]),
    ("MARL, physics in the reward or the initialization", [
        ("shaping", "Potential shaping \\citep{hong2025mpfrs}"), ("bc-init", "BC of the anchor \\citep{pomerleau1991}"),
        ("bc-finetune", "BC of the anchor + PPO \\citep{jsrl}")]),
    ("MARL, physics governs learning", [("peva-gen", "PEVA-Gen \\citep{valueaware}")]),
]
ORACLE = ("oracle-target", "Oracle (true positions)")


EPKEY = {"succ": "success_rate", "all": "all_found", "ttd": "ttd_all"}


def _episode_values(src, k):
    """Per-episode values of metric k from an evaluation directory (eval_misspec or formal format)."""
    p = Path(src) / "episodes.jsonl"
    if p.exists():
        rows = [json.loads(line) for line in open(p)]
        if rows and "success_rate" in rows[0]:
            return np.array([r[EPKEY[k]] for r in rows], float)
    s = json.load(open(Path(src) / "summary.json"))
    out = []
    for r in s.get("records", []):
        info = r["info"]
        out.append(float(info["success_rate"] >= 1 - 1e-9) if k == "all" else info[EPKEY[k]])
    return np.array(out, float)


def stat(rows, k):
    """(mean, sd, n, kind): sd over seeds for learned methods, over evaluation episodes for deterministic ones."""
    v = np.array([r[k] for r in rows if r.get(k) is not None], float)
    if not len(v):
        return None
    if len(v) > 1:
        return float(v.mean()), float(v.std(ddof=1)), len(v), "seed"
    # deterministic method: no training randomness; report the standard error of the mean over the
    # evaluation episodes, which is the uncertainty of the reported number on this episode set
    ep = _episode_values(rows[0]["src"], k) if rows[0].get("src") else np.array([])
    return float(v.mean()), (float(ep.std(ddof=1) / np.sqrt(len(ep))) if len(ep) > 1 else None), 1, "episode"


def cell(s, best, second, sd=True):
    """mean with subscript sd; deterministic (episode-level) sd is marked with a dagger in the caption sense:
    it is printed in grey so seed and episode spreads are not confused."""
    if s is None:
        return "--"
    m, d, _, kind = s
    t = f"{m:.3f}"
    if best is not None and abs(m - best) < 5e-4:
        t = r"\mathbf{" + t + "}"
    elif second is not None and abs(m - second) < 5e-4:
        t = r"\underline{" + t + "}"
    if sd and d is not None:
        sdt = f"{d:.3f}".lstrip("0") if d < 1 else f"{d:.2f}"
        t += (r"_{\pm" + sdt + "}") if kind == "seed" else (r"_{\color{epsd}\pm" + sdt + "}")
    return "$" + t + "$"


keys = [k for _, g in GROUPS for k, _ in g]
cols = [(sp, k) for sp in SPL for k in ("succ", "all", "ttd")]
S = {m: {c: stat(M[m].get(c[0]) or [], c[1]) for c in cols} for m in keys + [ORACLE[0]]}
rank = {}
for c in cols:
    vals = sorted({round(S[m][c][0], 3) for m in keys if S[m][c] is not None}, reverse=(c[1] != "ttd"))
    rank[c] = (vals[0] if vals else None, vals[1] if len(vals) > 1 else None)

L = [r"\begin{tabular}{l" + "ccc" * 3 + "c}", r"\toprule",
     r" & \multicolumn{3}{c}{Test (22--23 Jan)} & \multicolumn{3}{c}{Validation (12--13 Jan)} & "
     r"\multicolumn{3}{c}{OOD (24 scenarios)} & \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
     r"Method & Succ.$\uparrow$ & All$\uparrow$ & TTD$\downarrow$ & Succ.$\uparrow$ & All$\uparrow$ & TTD$\downarrow$ & "
     r"Succ.$\uparrow$ & All$\uparrow$ & TTD$\downarrow$ & $n$ \\", r"\midrule"]
for gname, g in GROUPS:
    L.append(r"\multicolumn{11}{l}{\emph{" + gname + r"}}\\")
    for k, lab in g:
        n = max([len(M[k].get(sp) or []) for sp in SPL])
        cs = [cell(S[k][c], *rank[c]) for c in cols]
        L.append(f"{lab} & " + " & ".join(cs) + f" & {n if n > 1 else 'det.'} \\\\")
    L.append(r"\midrule")
cs = [cell(S[ORACLE[0]][c], None, None) for c in cols]
L += [f"{ORACLE[1]} & " + " & ".join(cs) + r" & det. \\", r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_main.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))


# ------------------------------------------------------------- paired statistics vs deterministic planners
def _load(src):
    """{(scenario, seed): (success, ttd)} for one evaluation directory."""
    d = {}
    p = src / "episodes.jsonl"
    if p.exists():
        rows = [json.loads(line) for line in open(p)]
        if rows and "success_rate" in rows[0]:          # eval_misspec format
            for r in rows:
                d[(r["scenario_id"], int(r["seed"]))] = (r["success_rate"], r["ttd_all"])
            return d
    s = json.load(open(src / "summary.json"))
    for r in s.get("records", []):
        d[(r["scenario_id"], int(r["seed"]))] = (r["info"]["success_rate"], r["info"]["ttd_all"])
    return d


def _classical(split, m):
    s = json.load(open(PR / f"runs/complete-paper-v1/{split}/classical/summary.json"))
    return {(r["scenario_id"], int(r["seed"])): (r["info"]["success_rate"], r["info"]["ttd_all"])
            for r in s["records"][m]}


def episodes_of(m, split):
    rows = M[m].get(split) or []
    out = []
    for r in rows:
        if "src" in r:
            out.append(_load(Path(r["src"])))
        else:
            out.append(_classical(split, m))
    return out


rng = np.random.default_rng(0)
PAIR = {}
REFS = ["sector-search", "expanding-square", "aco-search", "pvf-greedy", "cbba", "mpc-horizon", "pso-search",
        "lawnmower-coverage", "bayesian-information-gain"]
for split in SPL:
    for us, others in (("anchor30", ["phys30"] + REFS), ("phys30", REFS)):
        ours = episodes_of(us, split)
        for m in others:
            ref = episodes_of(m, split)
            if not ours or not ref:
                continue
            ds, dt, sc = [], [], []
            for o in ours:                       # every seed of `us` against every seed of `m`, same episode
                for r in ref:
                    for key, (s, t) in o.items():
                        if key in r:
                            ds.append(s - r[key][0])
                            dt.append(t - r[key][1])
                            sc.append(key[0])
            if not ds:
                continue
            ds, dt, sc = np.array(ds), np.array(dt), np.array(sc)
            # cluster bootstrap: resample scenarios (episodes of one scenario share its currents), then pool
            # all pairs of the drawn scenarios; with two scenarios per test/validation split this is coarse,
            # so those splits keep the episode bootstrap and only OOD (24 scenarios) is resampled by scenario
            ids = np.unique(sc)
            if split == "ood":
                grp = [np.flatnonzero(sc == s) for s in ids]
                bs, bt = [], []
                for _ in range(2000):
                    ix = np.concatenate([grp[j] for j in rng.integers(len(grp), size=len(grp))])
                    bs.append(ds[ix].mean())
                    bt.append(dt[ix].mean())
            else:
                bs = [rng.choice(ds, len(ds)).mean() for _ in range(2000)]
                bt = [rng.choice(dt, len(dt)).mean() for _ in range(2000)]
            per = np.array([ds[sc == s].mean() for s in ids])
            key = f"{split}:{us}-vs-{m}"
            PAIR[key] = dict(n_pairs=len(ds), d_succ=float(ds.mean()), ci_succ=np.percentile(bs, [2.5, 97.5]).tolist(),
                             d_ttd=float(dt.mean()), ci_ttd=np.percentile(bt, [2.5, 97.5]).tolist(),
                             win=float(np.mean(ds > 1e-9)), tie=float(np.mean(np.abs(ds) <= 1e-9)),
                             loss=float(np.mean(ds < -1e-9)), n_scen=len(ids),
                             scen_win=int(np.sum(per > 1e-9)), scen_tie=int(np.sum(np.abs(per) <= 1e-9)),
                             scen_loss=int(np.sum(per < -1e-9)))
            p = PAIR[key]
            print(f"{split:10s} {us:8s} vs {m:26s} pairs {p['n_pairs']:5d}  dSucc {p['d_succ']:+.3f} "
              f"[{p['ci_succ'][0]:+.3f},{p['ci_succ'][1]:+.3f}]  dTTD {p['d_ttd']:+.3f} "
              f"[{p['ci_ttd'][0]:+.3f},{p['ci_ttd'][1]:+.3f}]  W/T/L {p['win']:.2f}/{p['tie']:.2f}/{p['loss']:.2f}")
(OUT / "paired_stats.json").write_text(json.dumps(PAIR, indent=1))
