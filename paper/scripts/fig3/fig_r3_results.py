"""Results figures of revision 3: detection timing (Fig. 4), writing vs learning the physics (Fig. 7),
generalisation (Fig. 10). Every number comes from results_r3.json and the raw per-episode files."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from r3_common import PR, VIZ, note, save
from style_r3 import METHOD, TYPE_COLOR, mcolor, head, SPLITC, OURS, PATTERN, PLAN

RES = json.load(open(PR / "runs/complete-paper-v4/results_r3.json"))
CACHE = PR / "runs/complete-paper-v4/det_cache_r3.npz"
DT, T = 30.0, 10800.0
_C = dict(np.load(CACHE)) if CACHE.exists() else {}
_dirty = [False]


def _npz_det(d, prefix="episode-"):
    out = []
    for f in sorted(Path(d).glob(f"{prefix}*.npz")):
        z = np.load(f)
        found, ts = z["found"], z["time_s"]
        t = np.full(found.shape[1], np.inf)
        for g in range(found.shape[1]):
            hit = np.flatnonzero(found[:, g])
            if len(hit):
                t[g] = ts[hit[0]]
            elif z["terminal_found"][g]:
                t[g] = ts[-1] + DT
        out.append(t)
    return np.asarray(out, float)


def _jsonl_det(d):
    rows = [json.loads(x) for x in open(Path(d) / "episodes.jsonl")]
    return np.asarray([[np.inf if v is None else v for v in r["detected_at"]] for r in rows], float)


def det(method, split="test"):
    """List (one per seed) of episodes x survivors detection-time arrays (inf = never)."""
    rows = RES.get(method, {}).get(split) or []
    out = []
    for i, r in enumerate(rows):
        key = f"{method}__{split}__{i}"
        if key not in _C:
            src = r.get("src")
            if src is None:
                _C[key] = _npz_det(PR / f"runs/complete-paper-v1/{split}/classical", f"{method}-episode-")
            else:
                p = Path(src) / "episodes.jsonl"
                ok = p.exists() and "success_rate" in json.loads(open(p).readline())
                _C[key] = _jsonl_det(src) if ok else _npz_det(src)
            _dirty[0] = True
        if len(_C[key]):
            out.append(_C[key])
    return out


def vals(method, split, k="succ"):
    return np.array([r[k] for r in (RES.get(method, {}).get(split) or []) if r.get(k) is not None], float)


def lab(k):
    return METHOD[k][0] if k in METHOD else k


# =============================================================== Fig. 4 detection timing
SHORT4 = {"phys30": "Anchor only (ours)", "anchor30": "Anchor + res. (ours)", "anchor15": "Anchor + res., 15°",
          "sector-search": "Sector search", "expanding-square": "Expanding square", "aco-search": "Ant colony",
          "pso-search": "Particle swarm", "pvf-greedy": "PVF-greedy", "greedy-probability": "Greedy prob.",
          "mpc-horizon": "RH-MPC", "cbba": "CBBA", "bayesian-information-gain": "Bayes. info. gain",
          "lawnmower-coverage": "Parallel track", "levy-random-walk": "Lévy walk", "voronoi-coverage": "Voronoi",
          "spiral-search": "Spiral", "gradient-ascent": "Gradient ascent", "shaping": "Shaping only",
          "mappo-long": "MAPPO 5×", "ic3net": "IC3Net", "mappo": "MAPPO", "commnet": "CommNet",
          "pvf-ic3net": "PVF+IC3Net", "pvf-mappo": "PVF+MAPPO", "pvf-happo": "PVF+HAPPO",
          "pvf-commnet": "PVF+CommNet", "happo": "HAPPO", "peva-gen": "PEVA-Gen", "ippo": "IPPO",
          "bc-init": "BC only", "bc-finetune": "BC + PPO"}


def fig_ttd():
    fig = plt.figure(figsize=(7.0, 3.85))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    tg = np.arange(0, T + DT, DT)
    CURVES = ["anchor30", "sector-search", "expanding-square", "aco-search", "pso-search", "pvf-greedy", "cbba",
              "ic3net", "mappo", "pvf-mappo", "peva-gen"]
    for k in CURVES:
        A = det(k)
        if not A:
            continue
        per_seed = np.array([[np.mean(a <= t) for t in tg] for a in A])
        mu = per_seed.mean(0)
        _, typ, ls = METHOD[k]
        if k == "anchor30" and len(A) > 1:
            ax.fill_between(tg / 3600, per_seed.min(0), per_seed.max(0), color=OURS, alpha=0.15, lw=0, zorder=4)
        ax.plot(tg / 3600, mu, color=TYPE_COLOR[typ], ls=ls, lw=1.9 if k == "anchor30" else 1.05,
                label=lab(k).replace(" (ours)", " (ours)"), zorder=5 if k == "anchor30" else 3,
                solid_capstyle="round")
        note(f"fig4a {k:20s} detected by 15min {mu[30]:.3f} 30min {mu[60]:.3f} 1h {mu[120]:.3f} 3h {mu[-1]:.3f} "
             f"(seeds {len(A)}, episodes {sum(len(a) for a in A)})")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("Mission time (h)")
    ax.set_ylabel("Survivors detected (cumulative)")
    ax.grid(axis="y")
    h, l = ax.get_legend_handles_labels()
    l = [x.replace("Physics anchor + residual (ours)", "Anchor + res. (ours)") for x in l]
    # the curves leave an empty band between 0.16 and 0.55 after the first half hour: put the legend there
    ax.legend(h, l, ncol=2, fontsize=5.6, loc="center", bbox_to_anchor=(0.6, 0.30), handlelength=1.6,
              columnspacing=0.8, labelspacing=0.22, frameon=True, framealpha=0.95, edgecolor="none",
              facecolor="white", borderpad=0.25)
    head(ax, "a", "Detections over the 3-h mission (held-out test)", y=1.055)

    ax = fig.add_subplot(gs[1])
    ROWS = ["phys30", "anchor30", "anchor15", "sector-search", "expanding-square", "aco-search", "pso-search",
            "pvf-greedy", "greedy-probability", "mpc-horizon", "cbba", "bayesian-information-gain",
            "lawnmower-coverage", "levy-random-walk", "voronoi-coverage", "spiral-search", "gradient-ascent",
            "shaping", "mappo-long", "ic3net", "mappo", "commnet", "pvf-ic3net", "pvf-mappo", "pvf-happo",
            "pvf-commnet", "happo", "peva-gen", "ippo"]
    edges = [0, 900, 1800, 3600, T + DT]
    BLAB = ["≤ 15 min", "15–30 min", "30–60 min", "1–3 h", "not found"]
    BCOL = ["#123A6B", "#2F6BAA", "#7FA7D6", "#C9DBEC", "#EEF0F2"]     # the paper's navy family, dark = early
    data = []
    for k in ROWS:
        A = det(k)
        if not A:
            continue
        P = np.concatenate(A).ravel()
        fr = [np.mean(P <= edges[1])] + [np.mean((P > edges[i]) & (P <= edges[i + 1])) for i in range(1, 4)]
        fr.append(1 - sum(fr))
        ttd = np.mean(vals(k, "test", "ttd"))
        data.append((k, fr, ttd))
    data.sort(key=lambda d: d[2])
    for y, (k, fr, ttd) in enumerate(data):
        left = 0
        for j, f in enumerate(fr):
            ax.barh(y, f, left=left, height=0.74, color=BCOL[j], lw=0, zorder=2)
            left += f
        ax.text(1.015, y, f"{ttd:.2f}", va="center", fontsize=5.8, color="#333333", transform=ax.get_yaxis_transform())
        note(f"fig4b {k:26s} " + " ".join(f"{b}:{f:.3f}" for b, f in zip(BLAB, fr)) + f"  TTD {ttd:.3f}")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([SHORT4.get(k, lab(k)) for k, _, _ in data], fontsize=5.9)
    for t, (k, _, _) in zip(ax.get_yticklabels(), data):
        t.set_color(mcolor(k))
        if METHOD[k][1] == "ours":
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.set_xlabel("Share of survivors")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.text(1.015, -0.95, "TTD", fontsize=5.9, fontweight="bold", transform=ax.get_yaxis_transform(), va="center")
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#C3C9D0" if j == 4 else c, lw=0.4, label=b)
                       for j, (c, b) in enumerate(zip(BCOL, BLAB))], ncol=5, fontsize=5.8,
              loc="lower left", bbox_to_anchor=(-0.02, 1.035), handlelength=1.0, columnspacing=0.6,
              borderaxespad=0.0)
    ax.text(-0.36, 1.095, "b", transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    ax.text(-0.30, 1.095, "When each survivor is found (test, seeds pooled)", transform=ax.transAxes,
            fontsize=7.5, ha="left", va="bottom")
    save(fig, "figR4_detection_timing")


# =============================================================== Fig. 7 writing vs learning
SHORT = {"peva-nodiff": "− diffusion belief", "peva-norsr": "− rectification", "peva-beta0": "β = 0, no PVF ctx",
         "peva-nocov": "− coverage update", "peva-nophys": "− physics context", "anchor-2upd": "Anchor, 2 updates",
         "anchor15": "Anchor + res., 15°", "anchor30": "Anchor + res., 30°", "phys15": "Anchor only, 15°",
         "phys30": "Anchor only, 30°", "mappo-long": "MAPPO, 5× steps", "bc-finetune": "BC + PPO",
         "bc-init": "BC only", "shaping": "Shaping only", "peva-gen": "PEVA-Gen (full)"}
GROUPS7 = [("No physics", "marl", ["mappo", "mappo-long", "ippo", "commnet", "happo", "ic3net"]),
           ("Physics\nas input", "marl_in", ["pvf-mappo", "pvf-commnet", "pvf-happo", "pvf-ic3net", "anchor-input"]),
           ("Physics governs\nlearning (PEVA-Gen)", "peva", ["peva-gen", "peva-nodiff", "peva-norsr", "peva-beta0",
                                                            "peva-nocov", "peva-nophys"]),
           ("Physics in\nreward / init.", "shape", ["shaping", "bc-init", "bc-finetune"]),
           ("Physics written into\nthe action (ours)", "ours", ["anchor-2upd", "anchor15", "anchor30", "phys15",
                                                               "phys30"])]


TICK7 = {"mappo": "MAPPO", "mappo-long": "MAPPO 5×", "ippo": "IPPO", "commnet": "CommNet", "happo": "HAPPO",
         "ic3net": "IC3Net", "pvf-mappo": "MAPPO", "pvf-commnet": "CommNet", "pvf-happo": "HAPPO",
         "pvf-ic3net": "IC3Net", "anchor-input": "Anchor dir.", "peva-gen": "Full", "peva-nodiff": "−Diff.", "peva-norsr": "−RSR",
         "peva-beta0": "β=0", "peva-nocov": "−Cov.", "peva-nophys": "−Phys.", "shaping": "Shaping",
         "bc-init": "BC", "bc-finetune": "BC+PPO", "anchor-2upd": "2 upd.", "anchor15": "Res. 15°",
         "anchor30": "Res. 30°", "phys15": "Only 15°", "phys30": "Only 30°"}
GROUP_TITLE7 = ["No physics", "Physics as input", "PEVA-Gen", "Reward / init.", "Written (ours)"]
# one muted fill per family, the same family colours as every other figure, lighter for non-ours
FILL7 = {"marl": "#B9C0C8", "marl_in": "#EDB98A", "peva": "#D9959B", "shape": "#C2AE93", "ours": OURS}


def fig_written():
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    x, ticks, names = 0.0, [], []
    rng = np.random.default_rng(1)
    for gi, (gname, typ, keys) in enumerate(GROUPS7):
        x0 = x
        for k in keys:
            te, va = vals(k, "test"), vals(k, "validation")
            if not len(te) and not len(va):
                continue
            m = te.mean() if len(te) else np.nan
            if len(te):
                ax.bar(x, m, 0.7, color=FILL7[typ], edgecolor=TYPE_COLOR[typ], lw=0.6, zorder=2)
                if len(te) > 1:
                    ax.errorbar(x, m, yerr=te.std(ddof=1), color="#333333", lw=0.6, capsize=1.5, capthick=0.5, zorder=4)
                    ax.scatter(x + rng.uniform(-0.18, 0.18, len(te)), te, s=2.6, color="#333333", lw=0, alpha=0.7,
                               zorder=5)
            if len(va):
                ax.scatter(x, va.mean(), marker="D", s=9, facecolor="white", edgecolor="#1A1A1A", lw=0.7, zorder=6)
            ticks.append(x)
            names.append(TICK7.get(k, lab(k)))
            note(f"fig7 {k:16s} test {m:.3f} (n={len(te)})  val {va.mean() if len(va) else float('nan'):.3f} "
                 f"(n={len(va)})")
            x += 1
        if gi % 2 == 0:
            ax.axvspan(x0 - 0.5, x - 0.5, color="#F4F6F8", zorder=0, lw=0)
        ax.text((x0 + x - 1) / 2, 1.02, GROUP_TITLE7[gi], ha="center", va="bottom", fontsize=6.6,
                color=TYPE_COLOR[typ], fontweight="bold", transform=ax.get_xaxis_transform())
        x += 0.6
    v = vals("sector-search", "test").mean()
    ax.axhline(v, color="#6B747D", ls=(0, (4, 2)), lw=0.7, zorder=1)
    ax.text(x - 0.5, v + 0.012, "sector search", ha="right", va="bottom", fontsize=5.8, color="#6B747D")
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, rotation=40, ha="right", rotation_mode="anchor", fontsize=6.1)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.7, x - 0.4)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Survivors detected (test)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", mfc="white", mec="#1A1A1A", mew=0.7, ms=3.6,
                              label="validation mean"),
                       Line2D([], [], marker="o", ls="", color="#333333", ms=2.2, alpha=0.7, label="seeds")],
              loc="upper left", fontsize=6.0, bbox_to_anchor=(0.0, 0.62), handletextpad=0.3)
    save(fig, "figR7_written_vs_learned")


# =============================================================== Fig. 10 generalisation
def box_speeds():
    bc = np.load(VIZ / "box_currents.npz")
    out = {}
    for sid in [str(s) for s in bc["ids"]]:
        sp = np.concatenate([np.hypot(*np.moveaxis(bc[sid + k], -1, 0)).ravel() for k in ("__t0", "__t3")])
        out[sid] = float(np.median(sp))
    return out


SUCC_CMAP = None


def succ_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("succ", ["#F7F9FB", "#C9DBEC", "#7FA7D6", "#2F6BAA", "#123A6B"])


def heat(ax, M, rows, cols, fmt="{:.2f}", vmin=0.0, vmax=1.0, cmap=None, bold_rows=(), tcol=0.62):
    """Square-cell annotated heat map (cell aspect 1), shared by Figs. 11 and 12."""
    ax.imshow(M, cmap=cmap or succ_cmap(), vmin=vmin, vmax=vmax, aspect="equal")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                hi = (M[i, j] - vmin) / (vmax - vmin) > tcol
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=5.6,
                        color="white" if hi else "#1A1A1A", fontweight="bold" if i in bold_rows else None)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)


GEN_ROWS = ["phys30", "anchor30", "anchor15", "aco-search", "sector-search", "expanding-square", "pso-search",
            "pvf-greedy", "cbba", "mpc-horizon", "bayesian-information-gain", "levy-random-walk",
            "voronoi-coverage", "lawnmower-coverage", "spiral-search", "bc-init", "bc-finetune", "shaping",
            "mappo-long", "ic3net", "mappo", "pvf-mappo", "ippo"]


def fig_gen():
    cand = [k for k in GEN_ROWS if k in RES and all(len(vals(k, sp)) for sp in ("validation", "test", "ood"))]
    cand.sort(key=lambda k: -vals(k, "ood").mean())
    cell = 0.235                                        # inches, square cells
    fig = plt.figure(figsize=(7.0, 4.15))
    fw, fh = fig.get_size_inches()
    # (a) heat map: evaluation set x method (methods as columns, sorted by OOD success)
    M = np.array([[vals(k, sp).mean() for k in cand] for sp in ("validation", "test", "ood")])
    x0, top = 0.62, 0.32
    ax = fig.add_axes([x0 / fw, (fh - top - 3 * cell) / fh, len(cand) * cell / fw, 3 * cell / fh])
    ax.imshow(M, cmap=succ_cmap(), vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j, k in enumerate(cand):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < 1 else "1.0", ha="center", va="center",
                    fontsize=5.2, color="white" if M[i, j] > 0.62 else "#1A1A1A",
                    fontweight="bold" if METHOD[k][1] == "ours" else None)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Val.", "Test", "OOD"], fontsize=6.2)
    ax.set_xticks(range(len(cand)))
    ax.set_xticklabels([SHORT4.get(k, lab(k)) for k in cand], rotation=50, ha="right", rotation_mode="anchor",
                       fontsize=6.0)
    for t, k in zip(ax.get_xticklabels(), cand):
        t.set_color(mcolor(k))
        if METHOD[k][1] == "ours":
            t.set_fontweight("bold")
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cand), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for k in cand:
        v = {sp: vals(k, sp).mean() for sp in ("test", "validation", "ood")}
        note(f"fig10a {k:24s} test {v['test']:.3f} val {v['validation']:.3f} ood {v['ood']:.3f}")
    ax.text(0.0, 1.06, "a", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    ax.text(0.025, 1.06, "Success on the three evaluation sets (methods sorted by OOD success)",
            transform=ax.transAxes, fontsize=7.5, va="bottom")

    # (b) OOD success by current quartile: the strongest methods plus two learned baselines, direct labels
    ax = fig.add_axes([0.62 / fw, 0.42 / fh, 4.6 / fw, 1.75 / fh])
    spd = box_speeds()
    ood_ids = sorted({s for r in RES["anchor30"]["ood"] for s in r["per"]})
    q = np.quantile([spd[s] for s in ood_ids], [0, 0.25, 0.5, 0.75, 1])
    bins = [[s for s in ood_ids if q[i] <= spd[s] <= q[i + 1] and (i == 3 or spd[s] < q[i + 1])] for i in range(4)]
    xq = np.arange(1, 5)
    LINES = [("anchor30", "Anchor + res. (ours)"), ("phys30", "Anchor only (ours)"), ("aco-search", "Ant colony"),
             ("sector-search", "Sector search"), ("expanding-square", "Expanding square"),
             ("pvf-greedy", "PVF-greedy"), ("shaping", "Shaping only"), ("ic3net", "IC3Net")]
    ends = []
    for k, name in LINES:
        rows = RES.get(k, {}).get("ood") or []
        if not rows:
            continue
        per = {s: np.mean([r["per"][s] for r in rows if s in r["per"]]) for s in ood_ids}
        yv = [np.mean([per[s] for s in b]) for b in bins]
        _, typ, ls = METHOD[k]
        ours = typ == "ours"
        ax.plot(xq, yv, color=TYPE_COLOR[typ], ls=ls, lw=1.8 if ours else 1.1, marker="o", ms=3.4 if ours else 2.8,
                mec="white", mew=0.4, zorder=5 if ours else 3)
        ends.append([yv[-1], name, TYPE_COLOR[typ], ours])
        note(f"fig10b {k:22s} OOD success by current quartile " + " ".join(f"{a:.3f}" for a in yv)
             + f"  (quartile median speeds {np.round([np.median([spd[s] for s in b]) for b in bins], 3).tolist()})")
    ends.sort(key=lambda e: e[0])
    yl = [e[0] for e in ends]
    for i in range(1, len(yl)):
        yl[i] = max(yl[i], yl[i - 1] + 0.045)
    for (v, name, col, ours), yy in zip(ends, yl):
        ax.text(4.14, yy, name, color=col, fontsize=6.0, va="center", fontweight="bold" if ours else None,
                clip_on=False)
    ax.set_ylim(0.0, 1.02)
    med = [np.median([spd[s] for s in b]) for b in bins]
    ax.set_xticks(xq)
    ax.set_xticklabels([f"{m:.2f}" for m in med], fontsize=6.2)
    ax.set_xlim(0.85, 4.1)
    ax.set_xlabel("Median box current per OOD quartile (m s$^{-1}$)")
    ax.set_ylabel("Survivors detected (OOD)")
    ax.grid(axis="y")
    head(ax, "b", "OOD success by current strength")
    save(fig, "figR10_generalisation")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["ttd", "written", "gen"]
    for name in todo:
        globals()["fig_" + name]()
    if _dirty[0]:
        np.savez_compressed(CACHE, **_C)
