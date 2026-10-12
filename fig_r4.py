"""Revision-5 results figures: learning curves (Fig. 9), ablation (Fig. 10), generalisation (Fig. 11),
operational stress (Fig. 12) and survivor count / team size (Fig. 13).

Heat maps (Figs. 11-13) share one cell size (CELL inches), one colour scale, one font and one short-name set,
with methods always as columns; they are included in the manuscript at the same scale so cells print equal.
"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from r3_common import PR, E3, E4, note, save
from style_r3 import METHOD, mcolor, OURS
import fig_r3_results2 as R
import fig_r3_results2b as B
import fig_r3_results as F

CELL = 0.205                              # inches; figures are included at natural size, so cells print equal
SUCC, DIV = B.SUCC, B.DIV
SN = {"phys30": "Anchor only", "anchor30": "Anchor+res.", "anchor15": "Anchor+res. 15°",
      "sector-search": "Sector", "expanding-square": "Exp. square", "lawnmower-coverage": "Par. track",
      "spiral-search": "Spiral", "aco-search": "Ant colony", "pso-search": "PSO", "voronoi-coverage": "Voronoi",
      "levy-random-walk": "Lévy walk", "pvf-greedy": "PVF-greedy", "cbba": "CBBA",
      "bayesian-information-gain": "Info. gain", "greedy-probability": "Greedy", "mpc-horizon": "MPC",
      "mappo": "MAPPO", "mappo-long": "MAPPO 5×", "ippo": "IPPO", "ic3net": "IC3Net", "pvf-mappo": "PVF+MAPPO",
      "shaping": "Shaping", "bc-init": "BC", "bc-finetune": "BC+PPO", "commnet": "CommNet", "happo": "HAPPO"}


def ours(k):
    return METHOD.get(k, ("", ""))[1] == "ours"


# ------------------------------------------------------------------ shared heat-map helpers
def place(fig, x_in, top_in, nrows, ncols, ch=None):
    """Axes with cells CELL wide and ch (default CELL) high, top-left corner at (x_in, top_in) inches."""
    ch = ch or CELL
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x_in / fw, (fh - top_in - nrows * ch) / fh, ncols * CELL / fw, nrows * ch / fh])


def hm(ax, M, ylab, xkeys=None, xlab=None, cmap=SUCC, norm=None, fmt="{:.2f}", show_x=True, show_y=True):
    """Annotated heat map; columns are methods (xkeys) or plain labels (xlab). Cell shape follows the axes."""
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 1,
              aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            lvl = norm(v) if norm else v
            dark = abs(lvl - 0.5) > 0.33 if norm else lvl > 0.6
            s = fmt(v) if callable(fmt) else fmt.format(v)
            ax.text(j, i, s, ha="center", va="center", fontsize=5.4, color="white" if dark else "#1A1A1A",
                    fontweight="bold" if (xkeys and ours(xkeys[j])) else None)
    ax.xaxis.tick_top()
    ax.set_xticks(range(M.shape[1]))
    if show_x:
        ax.set_xticklabels([SN.get(k, k) for k in xkeys] if xkeys else xlab, fontsize=6.0,
                           rotation=90 if xkeys else 0)
        if xkeys:
            for t, k in zip(ax.get_xticklabels(), xkeys):
                t.set_color(mcolor(k))
                if ours(k):
                    t.set_fontweight("bold")
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(ylab if show_y else [], fontsize=6.0)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.9)
    ax.tick_params(which="minor", length=0)


def ftitle(fig, x_in, top_in, letter, text):
    fw, fh = fig.get_size_inches()
    fig.text(x_in / fw, 1 - top_in / fh, letter, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    fig.text((x_in + 0.14) / fw, 1 - top_in / fh, text, fontsize=7.5, ha="left", va="bottom")


def cbar(fig, x_in, top_in, h_in, label, cmap=SUCC, norm=None):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    fw, fh = fig.get_size_inches()
    cax = fig.add_axes([x_in / fw, (fh - top_in - h_in) / fh, 0.09 / fw, h_in / fh])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm or Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label(label, fontsize=6.2)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_linewidth(0.4)


# =============================================================== Fig. 11 generalisation
def fig_gen():
    cand = [k for k in F.GEN_ROWS if k in F.RES and all(len(F.vals(k, sp)) for sp in ("validation", "test", "ood"))]
    cand.sort(key=lambda k: -F.vals(k, "ood").mean())
    A = np.array([[F.vals(k, sp).mean() for k in cand] for sp in ("validation", "test", "ood")])
    spd = F.box_speeds()
    ood_ids = sorted({s for r in F.RES["anchor30"]["ood"] for s in r["per"]})
    q = np.quantile([spd[s] for s in ood_ids], [0, 0.25, 0.5, 0.75, 1])
    bins = [[s for s in ood_ids if q[i] <= spd[s] <= q[i + 1] and (i == 3 or spd[s] < q[i + 1])] for i in range(4)]
    Q = np.full((4, len(cand)), np.nan)
    for j, k in enumerate(cand):
        rows = F.RES[k]["ood"]
        per = {s: np.mean([r["per"][s] for r in rows if s in r["per"]]) for s in ood_ids}
        Q[:, j] = [np.mean([per[s] for s in b]) for b in bins]
        note(f"fig11 {k:24s} val {A[0, j]:.3f} test {A[1, j]:.3f} ood {A[2, j]:.3f} | OOD quartiles "
             + " ".join(f"{v:.3f}" for v in Q[:, j]))
    med = [np.median([spd[s] for s in b]) for b in bins]
    note(f"fig11 quartile median speeds {np.round(med, 3).tolist()} n={[len(b) for b in bins]}")
    x0, lab_h = 0.62, 0.62
    w = x0 + len(cand) * CELL + 0.55
    h = 0.2 + lab_h + 3 * CELL + 0.32 + 4 * CELL + 0.03
    fig = plt.figure(figsize=(w, h))
    top_a = 0.2 + lab_h
    ax = place(fig, x0, top_a, 3, len(cand))
    hm(ax, A, ["Val.", "Test", "OOD"], xkeys=cand)
    ftitle(fig, 0.0, 0.17, "a", "Success on the three evaluation sets (sorted by OOD success)")
    top_b = top_a + 3 * CELL + 0.32
    ax = place(fig, x0, top_b, 4, len(cand))
    hm(ax, Q, [f"Q{i + 1}  {m:.2f}" for i, m in enumerate(med)], xkeys=cand, show_x=False)
    ftitle(fig, 0.0, top_b - 0.04, "b", "OOD success by current quartile (median box current, m s$^{-1}$)")
    cbar(fig, x0 + len(cand) * CELL + 0.12, top_a, top_b + 4 * CELL - top_a, "Survivors detected")
    save(fig, "figR10_generalisation")


# =============================================================== Fig. 12 operational stress
CLAB = ["Nominal", "$p_d$ 0.6", "Drift 0.3", "Drift 0.5", "Loss 0.5", "Combined"]


def fig_robust():
    R.fig_robust()                        # writes tab_robust.tex (Table 5); its figure is overwritten below
    rows, M = R.stress_matrix()
    keep = [i for i in range(len(rows)) if np.isfinite(M[i, 1:]).any()]
    rows, M = [rows[i] for i in keep], M[keep]
    S = M.T                               # conditions x methods
    D = (S[1:] - S[:1]) * 100
    global CELL
    x0, lab_h = 0.55, 0.62
    nm = len(rows)
    cb = CELL * 6 / 5                     # (b) has 5 rows, (a) 6: enlarge the square cells of (b) to equal height
    w = x0 + nm * CELL + 0.62 + nm * cb + 0.05
    h = 0.2 + lab_h + 6 * CELL + 0.05
    fig = plt.figure(figsize=(w, h))
    top = 0.2 + lab_h
    ax = place(fig, x0, top, 6, nm)
    hm(ax, S, CLAB, xkeys=rows)
    ftitle(fig, 0.0, 0.17, "a", "Success on held-out test")
    x1 = x0 + nm * CELL + 0.62
    c0, CELL = CELL, cb
    ax2 = place(fig, x1, top, 5, nm)
    CELL = c0
    lim = np.nanmax(np.abs(D))
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    hm(ax2, D, CLAB[1:], xkeys=rows, cmap=DIV, norm=norm, fmt=lambda v: f"{int(round(v)):+d}" if round(v) else "0")
    ftitle(fig, x1 - 0.55, 0.17, "b", "Change from nominal (points)")
    for k, r in zip(rows, M):
        note(f"fig12 {k:26s} " + " ".join(f"{c}:{v:.3f}" for (c, _), v in zip(R.CONDS, r)))
    save(fig, "figR11_robustness")


# =============================================================== Fig. 13 survivor count / team size
def fig_count():
    C = B.count_matrix().T                # G x methods
    T, NS = B.team_matrix()
    T = T.T                               # N x methods
    global CELL
    x0, lab_h = 0.42, 0.62
    gap = 0.62
    ca = CELL * T.shape[0] / C.shape[0]   # (a) has 6 rows, (b) 7: enlarge the square cells of (a) to equal height
    w = x0 + len(B.CNT) * ca + gap + len(B.TEAM) * CELL + 0.6
    h = 0.2 + lab_h + T.shape[0] * CELL + 0.05
    fig = plt.figure(figsize=(w, h))
    top = 0.2 + lab_h
    c0, CELL = CELL, ca
    ax = place(fig, x0, top, C.shape[0], len(B.CNT))
    CELL = c0
    hm(ax, C, [f"$G$={g}" for g in range(1, 7)], xkeys=B.CNT)
    ftitle(fig, 0.0, 0.17, "a", "Survivors $G$ ($N=6$)")
    x1 = x0 + len(B.CNT) * ca + gap
    ax2 = place(fig, x1, top, T.shape[0], len(B.TEAM))
    hm(ax2, T, [f"$N$={n}" for n in NS], xkeys=B.TEAM)
    ftitle(fig, x1 - 0.42, 0.17, "b", "UAVs $N$ ($G=3$, network-free)")
    cbar(fig, x1 + len(B.TEAM) * CELL + 0.12, top, T.shape[0] * CELL, "Survivors detected")
    for k, r in zip(B.CNT, C.T):
        note(f"fig13a {k:22s} " + ", ".join(f"G={g + 1}:{v:.3f}" for g, v in enumerate(r) if np.isfinite(v)))
    for k, r in zip(B.TEAM, T.T):
        note(f"fig13b {k:22s} " + ", ".join(f"N={n}:{v:.3f}" for n, v in zip(NS, r) if np.isfinite(v)))
    save(fig, "figR12_count_team")


# =============================================================== Fig. 9 learning curves
def lc_points(pattern, split_at="-step"):
    pts = {}
    for p in glob.glob(str(pattern)):
        st = int(Path(p).name.split(split_at)[1].lstrip("-").split("-")[0])
        r = R.summ(p)
        if r:
            pts.setdefault(st, []).append(r[0])
    return {s: np.array(v) for s, v in pts.items()}


def fig_curves():
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    series = []
    a30 = lc_points(E4 / "lc-anchor30-step*-seed*")
    a30.setdefault(204800, np.array([]))
    a30[204800] = np.concatenate([a30[204800], R.res("anchor30", "validation")])
    series.append((a30, "Anchor + residual (ours)", OURS, "o", 1.8))
    pv = lc_points(R.CP / "learning-curve-eval/pvf-plain-seed*-step-*", "-step-")
    series.append((pv, "PVF + MAPPO", mcolor("pvf-mappo"), "s", 1.2))
    sh = lc_points(E4 / "lc-shapeonly-step*-seed*")
    if len(R.res("shaping", "validation")):
        sh[204800] = R.res("shaping", "validation")
    series.append((sh, "Potential shaping only", mcolor("shaping"), "D", 1.2))
    mp = lc_points(E4 / "lc-mappo-step*-seed*")         # standard-budget MAPPO (the 5x run is in Table 3)
    if len(R.res("mappo", "validation")):
        mp[204800] = R.res("mappo", "validation")
    series.append((mp, "MAPPO", mcolor("mappo"), "^", 1.2))
    for pts, name, col, mk, lw in series:
        xs = sorted(pts)
        mu = np.array([pts[x].mean() for x in xs])
        sd = np.array([pts[x].std(ddof=1) if len(pts[x]) > 1 else 0 for x in xs])
        xm = np.array(xs) / 1e5
        ax.fill_between(xm, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.15, lw=0)
        ax.plot(xm, mu, color=col, lw=lw, marker=mk, ms=3.0, mec="white", mew=0.4, label=name, zorder=4)
        note(f"fig9 {name:28s} " + ", ".join(f"{int(x)}:{m:.3f}(n={len(pts[x])})" for x, m in zip(xs, mu)))
    v = R.res("phys30", "validation")
    if len(v):
        ax.axhline(v.mean(), color=OURS, ls=(0, (4, 2)), lw=0.8, zorder=2,
                   label="Anchor only, no training (ours)")
        note(f"fig9 anchor only validation {v.mean():.3f}")
    ax.set_xlim(0, 2.12)
    ax.set_xticks([0, 0.512, 1.024, 1.536, 2.048])
    ax.set_xticklabels(["0", "0.51", "1.02", "1.54", "2.05"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Environment steps ($\times10^{5}$)")
    ax.set_ylabel("Survivors detected (val.)")
    ax.grid(axis="y")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=6.2, handlelength=2.0)
    save(fig, "figR8_learning_curves")


# =============================================================== Fig. 10 ablation
NF = "#2A9D8F"                            # network-free anchor (teal), trained anchor + residual in OURS navy


def ep_se(d):
    p = Path(d) / "episodes.jsonl"
    if not p.exists():
        return np.nan
    v = np.array([json.loads(l)["success_rate"] for l in open(p)], float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def nf_line(ax, pts):
    """Network-free anchor: line with a band of +-1 standard error over the evaluation episodes."""
    xs = sorted(pts)
    mu = np.array([pts[x][0] for x in xs])
    se = np.array([pts[x][1] for x in xs])
    ax.fill_between(xs, mu - se, mu + se, color=NF, alpha=0.22, lw=0, zorder=1)
    ax.plot(xs, mu, color=NF, lw=1.3, marker="o", ms=2.6, mec="white", mew=0.3, zorder=2,
            label="Anchor only (± s.e.)")


def tr_pts(ax, data):
    """Trained anchor + residual: mean with +-1 s.d. over seeds, same marker and bar at every setting."""
    xs = [x for x in sorted(data) if len(data[x])]
    mu = np.array([data[x].mean() for x in xs])
    sd = np.array([data[x].std(ddof=1) for x in xs])
    ax.errorbar(xs, mu, yerr=sd, fmt="o", color=OURS, ms=3.6, mec="white", mew=0.5, elinewidth=1.0, capsize=2.2,
                capthick=0.8, zorder=5, label="Anchor + residual (± s.d.)")
    return xs, mu, sd


def fig_ablation():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.1), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    pts = {}
    for th in (0, 15, 30, 45, 60, 75, 90):
        d = E3 / f"zero-test-ang{th}"
        if R.summ(d):
            pts[th] = (R.summ(d)[0], ep_se(d))
    for th in (5, 10, 20, 25, 35, 40, 50):
        d = E4 / f"sens-test-theta{th}"
        if R.summ(d):
            pts[th] = (R.summ(d)[0], ep_se(d))
    nf_line(ax, pts)
    tr = {0: R.res("anchor0"), 15: R.res("anchor15"), 30: R.res("anchor30")}
    for th in (45, 60, 90):
        tr[th] = R.res(f"anchor-ang{th}")
    xs, mu, sd = tr_pts(ax, tr)
    note("fig10a anchor only: " + ", ".join(f"{t}:{pts[t][0]:.3f}±{pts[t][1]:.3f}" for t in sorted(pts)))
    note("fig10a trained: " + ", ".join(f"{x}:{m:.3f}±{s:.3f}(n={len(tr[x])})" for x, m, s in zip(xs, mu, sd)))
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlim(-5, 95)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel(r"Rotation angle $\theta$ (°)")
    ax.set_ylabel("Survivors detected (test)")
    ax.legend(loc="lower left", fontsize=6.0)
    B.title(ax, "a", r"Angle ($\alpha=4$)")
    ax = axs[1]
    pts = {}
    for al in (1, 2, 3, 4, 6, 8, 12):
        d = E4 / f"sens15-test-alpha{al}"
        if R.summ(d):
            pts[al] = (R.summ(d)[0], ep_se(d))
    nf_line(ax, pts)
    tr = {1: R.res("anchor-str1"), 2: R.res("anchor-str2"), 4: R.res("anchor15"), 6: R.res("anchor-str6"),
          8: R.res("anchor-str8"), 12: R.res("anchor-str12")}
    xs, mu, sd = tr_pts(ax, tr)
    note("fig10b anchor only 15: " + ", ".join(f"{a}:{pts[a][0]:.3f}±{pts[a][1]:.3f}" for a in sorted(pts)))
    note("fig10b trained: " + ", ".join(f"{x}:{m:.3f}±{s:.3f}(n={len(tr[x])})" for x, m, s in zip(xs, mu, sd)))
    ax.set_xticks([1, 2, 4, 6, 8, 12])
    ax.set_xlim(0.3, 12.7)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel(r"Anchor strength $\alpha$")
    B.title(ax, "b", r"Strength ($\theta=15^{\circ}$)")
    ax = axs[2]
    keys = [("anchor15", 0), ("shape0.5-anchor", 0.5), ("shape1-anchor-s4", 1), ("shape2.0-anchor", 2),
            ("shape4.0-anchor", 4)]
    d15 = E3 / "zero-test-ang15"
    m15, s15 = R.summ(d15)[0], ep_se(d15)
    ax.axhspan(m15 - s15, m15 + s15, color=NF, alpha=0.22, lw=0, zorder=1)
    ax.axhline(m15, color=NF, lw=1.3, zorder=2)
    tr = {i: R.res(k) for i, (k, _) in enumerate(keys)}
    xs, mu, sd = tr_pts(ax, tr)
    note(f"fig10c anchor only 15: {m15:.3f}±{s15:.3f}; trained: "
         + ", ".join(f"beta={keys[x][1]}:{m:.3f}±{s:.3f}(n={len(tr[x])})" for x, m, s in zip(xs, mu, sd)))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(b) for _, b in keys])
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel(r"Shaping coefficient $\beta$")
    B.title(ax, "c", r"Shaping ($\theta=15^{\circ}$)")
    for a in axs:
        a.grid(axis="y")
    save(fig, "figR9_ablation")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["curves", "ablation", "gen", "robust", "count"]):
        globals()["fig_" + name]()
