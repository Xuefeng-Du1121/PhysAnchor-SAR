"""Revision-4 versions of Fig. 9 (ablation), Fig. 11 (stress) and Fig. 12 (survivor count / team size).

Same data and loaders as fig_r3_results2.py; only the presentation changes:
  * Fig. 9  : three line panels with seed dots and s.d. error bars, one hue family, no bars, no on-plot prose
  * Fig. 11 : (a) square-cell success heat map, (b) square-cell change heat map (diverging), equal heights
  * Fig. 12 : two square-cell heat maps (method x G, method x N) on one colour scale
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from r3_common import E3, E4, note, save
from style_r3 import METHOD, mcolor, head, OURS, OURS_M, OURS_L
import fig_r3_results2 as R

SUCC = LinearSegmentedColormap.from_list("succ", ["#F7F9FB", "#C9DBEC", "#7FA7D6", "#2F6BAA", "#123A6B"])
DIV = LinearSegmentedColormap.from_list("div", ["#B2182B", "#EF8A62", "#FDDBC7", "#F7F7F7", "#D1E5F0", "#67A9CF",
                                                "#2166AC"])
SHORT = {"phys30": "Anchor only (ours)", "anchor30": "Anchor + res. (ours)", "sector-search": "Sector search",
         "expanding-square": "Exp. square", "aco-search": "Ant colony", "pso-search": "Particle swarm",
         "mpc-horizon": "RH-MPC", "pvf-greedy": "PVF-greedy", "cbba": "CBBA",
         "bayesian-information-gain": "Bayes. info. gain", "lawnmower-coverage": "Parallel track",
         "ic3net": "IC3Net", "pvf-mappo": "PVF+MAPPO", "ippo": "IPPO"}


def place(fig, x_in, top_in, nrows, ncols, cell_in):
    """Axes whose data cells are exactly cell_in x cell_in inches, top-left corner at (x_in, top_in)."""
    fw, fh = fig.get_size_inches()
    w, h = ncols * cell_in, nrows * cell_in
    return fig.add_axes([x_in / fw, (fh - top_in - h) / fh, w / fw, h / fh])


def heat(ax, M, rows, cols, cmap, norm=None, fmt="{:.2f}", tcol=0.6):
    """Square-cell annotated heat map."""
    im = ax.imshow(M, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 1, aspect="equal")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=5.6, color="#9AA5B1")
                continue
            lvl = norm(v) if norm else v
            dark = (lvl > tcol) if not norm else (abs(lvl - 0.5) > 0.33)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5, color="white" if dark else "#1A1A1A",
                    fontweight="bold" if METHOD.get(rows[i], ("", ""))[1] == "ours" else None)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(k, R.lab(k)) for k in rows], fontsize=6.0)
    for t, k in zip(ax.get_yticklabels(), rows):
        t.set_color(mcolor(k))
        if METHOD[k][1] == "ours":
            t.set_fontweight("bold")
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.9)
    ax.tick_params(which="minor", length=0)
    return im


def title(ax, letter, text, y=1.0):
    ax.text(0.0, y, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    ax.text(0.0, y, "    " + text, transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom")


# =============================================================== Fig. 9 ablation (line panels)
def seeds(ax, x, v, col, mk="o", ms=3.4, lw=1.4, label=None):
    """Mean marker + s.d. bar + jittered seed dots at one x."""
    if not len(v):
        return
    rng = np.random.default_rng(int(x * 100) % 997)
    ax.scatter(x + rng.normal(0, 0.012, len(v)) * (ax.get_xlim()[1] - ax.get_xlim()[0]), v, s=3.0, color=col,
               alpha=0.35, lw=0, zorder=3)
    if len(v) > 1:
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=col, lw=0.9, capsize=1.8, capthick=0.7, zorder=4)
    ax.plot(x, v.mean(), mk, color=col, ms=ms, mec="white", mew=0.5, zorder=5, label=label)


def fig_ablation():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw=dict(wspace=0.28, width_ratios=[1.3, 1, 0.85]))
    # (a) angle: network-free sweep on test (line) + trained seeds (points)
    ax = axs[0]
    ax.set_xlim(-6, 96)
    pts = {}
    for th in (0, 15, 30, 45, 60, 75, 90):
        r = R.summ(E3 / f"zero-test-ang{th}")
        if r:
            pts[th] = r[0]
    for th in (5, 10, 20, 25, 35, 40, 50):
        r = R.summ(E4 / f"sens-test-theta{th}")
        if r:
            pts[th] = r[0]
    xs = sorted(pts)
    ax.plot(xs, [pts[x] for x in xs], color=OURS_L, lw=1.6, zorder=2, label="anchor only")
    ax.plot(xs, [pts[x] for x in xs], "o", color=OURS_L, ms=2.4, mec="white", mew=0.3, zorder=2)
    learned = {0: R.res("anchor0"), 15: R.res("anchor15"), 30: R.res("anchor30")}
    for th, v in learned.items():
        seeds(ax, th, v, OURS, label="anchor + residual" if th == 0 else None)
        note(f"fig9a learned theta={th}: {v.mean():.3f}±{v.std(ddof=1):.3f} (n={len(v)})")
    note("fig9a physics-only test: " + ", ".join(f"{t}:{pts[t]:.3f}" for t in xs))
    ax.axvspan(25, 60, color="#EEF3F9", zorder=0, lw=0)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel(r"Rotation angle $\theta$ (°)")
    ax.set_ylabel("Survivors detected (test)")
    ax.set_ylim(0.2, 1.02)
    ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=5.9, handlelength=1.4)
    title(ax, "a", "Rotation angle")
    # (b) strength
    ax = axs[1]
    ax.set_xlim(0.3, 12.7)
    for pre, col, name in (("sens15-test-alpha", OURS_L, r"anchor only, 15°"), ("sens-test-alpha", OURS_M,
                                                                                 r"anchor only, 30°")):
        p = {}
        for al in (1, 2, 3, 4, 6, 8, 12):
            r = R.summ(E4 / f"{pre}{al}")
            if r:
                p[al] = r[0]
        if "15" not in name and 4 not in p and len(R.res("phys30")):
            p[4] = R.res("phys30").mean()
        if "15" in name and 4 not in p and len(R.res("phys15")):
            p[4] = R.res("phys15").mean()
        xs = sorted(p)
        ax.plot(xs, [p[x] for x in xs], color=col, lw=1.4, marker="o", ms=2.4, mec="white", mew=0.3, zorder=2,
                label=name)
        note(f"fig9b {name}: " + ", ".join(f"{a}:{p[a]:.3f}" for a in xs))
    for al, v in {2: R.res("anchor-str2"), 4: R.res("anchor15"), 6: R.res("anchor-str6"),
                  8: R.res("anchor-str8")}.items():
        seeds(ax, al, v, OURS, label="anchor + res., 15°" if al == 2 else None)
    ax.set_xticks([1, 2, 4, 6, 8, 12])
    ax.set_xlabel(r"Anchor strength $\alpha$")
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.4, labelspacing=0.25)
    title(ax, "b", "Anchor strength")
    # (c) shaping coefficient
    ax = axs[2]
    keys = [("anchor15", 0), ("shape0.5-anchor", 0.5), ("shape1-anchor-s4", 1), ("shape2.0-anchor", 2),
            ("shape4.0-anchor", 4)]
    ax.set_xlim(-0.6, len(keys) - 0.4)
    p15 = R.res("phys15")
    if len(p15):
        ax.axhline(p15.mean(), color=OURS_L, lw=1.4, zorder=1, label="anchor only, 15°")
    for i, (k, b) in enumerate(keys):
        v = R.res(k)
        seeds(ax, i, v, OURS, label="anchor + res., 15°" if i == 0 else None)
        note(f"fig9c shaping beta={b}: {v.mean():.3f}±{(v.std(ddof=1) if len(v) > 1 else 0):.3f} (n={len(v)})")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(b) for _, b in keys])
    ax.set_xlabel(r"Shaping coefficient $\beta$")
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.4, labelspacing=0.25)
    title(ax, "c", "Potential shaping")
    save(fig, "figR9_ablation")


# =============================================================== Fig. 11 stress (two square heat maps)
def fig_robust():
    R.fig_robust()                        # writes tab_robust.tex (Table 5); its figure is overwritten below
    rows, M = R.stress_matrix()
    keep = [i for i in range(len(rows)) if np.isfinite(M[i, 1:]).any()]
    rows, M = [rows[i] for i in keep], M[keep]
    cols = ["Nom.", "$p_d$\n0.6", "Drift\n0.3", "Drift\n0.5", "Loss\n0.5", "All"]
    D = (M[:, 1:] - M[:, :1]) * 100
    cell = 0.205                                        # inches; square cells, both panels the same height
    fig = plt.figure(figsize=(7.0, 0.62 + cell * len(rows) + 0.08))
    x0 = 1.25
    ax = place(fig, x0, 0.62, len(rows), M.shape[1], cell)
    heat(ax, M, rows, cols, SUCC)
    ax2 = place(fig, x0 + cell * M.shape[1] + 0.4, 0.62, len(rows), D.shape[1], cell)
    lim = np.nanmax(np.abs(D))
    heat(ax2, D, rows, cols[1:], DIV, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), fmt="{:+.1f}")
    ax2.set_yticklabels([])
    for a in (ax, ax2):
        a.tick_params(axis="x", labelsize=5.8)
        for t in a.get_xticklabels():
            t.set_linespacing(0.95)
    title(ax, "a", "Success on held-out test", y=1.0 + 2.25 / len(rows))
    title(ax2, "b", "Change from nominal (points)", y=1.0 + 2.25 / len(rows))
    for k, r in zip(rows, M):
        note(f"fig11 {k:26s} " + " ".join(f"{c}:{v:.3f}" for (c, _), v in zip(R.CONDS, r)))
    save(fig, "figR11_robustness")


# =============================================================== Fig. 12 survivor count / team size
CNT = ["phys30", "anchor30", "sector-search", "expanding-square", "aco-search", "pso-search", "mpc-horizon",
       "pvf-greedy", "cbba", "ic3net"]
TEAM = ["phys30", "sector-search", "expanding-square", "aco-search", "pvf-greedy", "cbba", "lawnmower-coverage"]


def count_matrix():
    M = np.full((len(CNT), 6), np.nan)
    for a, k in enumerate(CNT):
        for G in range(1, 7):
            if G == 3:
                v = R.res(k)[:1] if k == "ic3net" else R.res(k)
            elif k == "anchor30":
                v = np.concatenate([R.many(E4 / f"count-G{G}-anchor30-seed*"), R.many(E3 / f"mis-G{G}K{G}-ang30-seed*")])
            elif k == "ic3net":
                v = R.many(E4 / f"count-G{G}-ic3net-seed0")
            elif k == "phys30":
                r = R.summ(E4 / f"count-G{G}-phys30") or R.summ(E3 / f"mis-G{G}K{G}-zero30")
                v = np.array([r[0]]) if r else np.array([])
            else:
                r = R.summ(E4 / f"count-G{G}-{k}") or R.summ(E3 / f"mis-G{G}-{k}")
                v = np.array([r[0]]) if r else np.array([])
            if len(v):
                M[a, G - 1] = v.mean()
    return M


def team_matrix():
    NS = (2, 3, 4, 6, 8, 10, 12)
    M = np.full((len(TEAM), len(NS)), np.nan)
    for a, k in enumerate(TEAM):
        for b, N in enumerate(NS):
            if N == 6:
                v = R.res(k)
                M[a, b] = v.mean() if len(v) else np.nan
            else:
                r = R.summ(E4 / f"team{N}-{k}")
                if r:
                    M[a, b] = r[0]
    return M, NS


def fig_count():
    C = count_matrix()
    T, NS = team_matrix()
    cell = 0.235
    fig = plt.figure(figsize=(7.0, 0.5 + cell * len(CNT) + 0.08))
    ax = place(fig, 1.3, 0.5, len(CNT), C.shape[1], cell)
    im = heat(ax, C, CNT, [str(g) for g in range(1, 7)], SUCC)
    title(ax, "a", "Survivors $G$ ($N=6$)", y=1.0 + 1.6 / len(CNT))
    ax2 = place(fig, 1.3 + cell * 6 + 1.3, 0.5, len(TEAM), T.shape[1], cell)
    heat(ax2, T, TEAM, [str(n) for n in NS], SUCC)
    title(ax2, "b", "UAVs $N$ ($G=3$, network-free)", y=1.0 + 1.6 / len(TEAM))
    fw, fh = fig.get_size_inches()
    cax = fig.add_axes([(1.3 + cell * 6 + 1.3 + cell * len(NS) + 0.25) / fw, (fh - 0.5 - cell * len(CNT)) / fh,
                        0.1 / fw, cell * len(CNT) / fh])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Survivors detected", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_linewidth(0.4)
    for k, r in zip(CNT, C):
        note(f"fig12a {k:22s} " + ", ".join(f"G={g + 1}:{v:.3f}" for g, v in enumerate(r) if np.isfinite(v)))
    for k, r in zip(TEAM, T):
        note(f"fig12b {k:22s} " + ", ".join(f"N={n}:{v:.3f}" for n, v in zip(NS, r) if np.isfinite(v)))
    save(fig, "figR12_count_team")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["ablation", "robust", "count"]):
        globals()["fig_" + name]()
