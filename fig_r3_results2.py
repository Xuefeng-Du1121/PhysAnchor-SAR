"""Results figures of revision 3, part 2: learning curves (Fig. 8), ablation (Fig. 9), operational stress
(Fig. 11 + Table 5) and survivor count / team size (Fig. 12). Reads summary.json / episodes.jsonl only."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from r3_common import PR, E3, E4, note, save
from style_r3 import METHOD, TYPE_COLOR, mcolor, head, SPLITC, OURS, OURS_M, OURS_L, HALO

RES = json.load(open(PR / "runs/complete-paper-v4/results_r3.json"))
CP = PR / "runs/complete-paper-v1"


def summ(d):
    """(success, complete rescue) of one evaluation directory, any format; None if missing."""
    p = Path(d) / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    g = lambda k: s[k]["mean"] if isinstance(s.get(k), dict) else s.get(k)  # noqa: E731
    allf = s.get("complete_rescue")
    if allf is None and s.get("records") and "info" in s["records"][0]:
        allf = float(np.mean([r["info"]["success_rate"] >= 1 - 1e-9 for r in s["records"]]))
    return g("success_rate"), allf


def many(pattern):
    v = [summ(d) for d in sorted(glob.glob(str(pattern)))]
    return np.array([x[0] for x in v if x], float)


def res(k, split="test", key="succ"):
    return np.array([r[key] for r in (RES.get(k, {}).get(split) or []) if r.get(key) is not None], float)


def lab(k):
    return METHOD[k][0] if k in METHOD else k


# =============================================================== Fig. 8 learning curves
def fig_curves():
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    END = []                                            # (y at last point, label, colour, bold) for direct labels

    def curve(points, k, lw=1.2, mk="o", label=None, ls=None):
        xs = sorted(points)
        mu = np.array([points[x].mean() for x in xs])
        sd = np.array([points[x].std(ddof=1) if len(points[x]) > 1 else 0 for x in xs])
        col = mcolor(k)
        ax.fill_between(xs, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.13, lw=0)
        ax.plot(xs, mu, color=col, lw=lw, ls="-", marker=mk, ms=2.8, mec="white", mew=0.4, zorder=4)
        END.append([mu[-1], xs[-1], label or lab(k), col, METHOD[k][1] == "ours"])
        note(f"fig8 {label or k:28s} " + ", ".join(f"{int(x)}:{m:.3f}(n={len(points[x])})" for x, m in zip(xs, mu)))

    lc = CP / "learning-curve-eval"
    for key, k, lb in (("anchor-mappo", "anchor15", "Anchor + res., 15° (ours)"),
                       ("pvf-plain", "pvf-mappo", "PVF + MAPPO")):
        pts = {}
        for p in glob.glob(str(lc / f"{key}-seed*-step-*/summary.json")):
            st = int(Path(p).parent.name.rsplit("-step-", 1)[1])
            pts.setdefault(st, []).append(summ(Path(p).parent)[0])
        pts = {s: np.array(v) for s, v in pts.items()}
        if pts:
            curve(pts, k, label=lb, lw=1.3 if k == "anchor15" else 1.1)
    pts = {}
    for p in glob.glob(str(E4 / "lc-anchor30-step*-seed*")):
        st = int(Path(p).name.split("-step")[1].split("-")[0])
        r = summ(p)
        if r:
            pts.setdefault(st, []).append(r[0])
    a30 = res("anchor30", "validation")
    if len(a30):
        pts.setdefault(204800, []).extend(a30[:3].tolist())
    if pts:
        curve({s: np.array(v) for s, v in pts.items()}, "anchor30", lw=1.8, label="Anchor + res., 30° (ours)")
    pts = {}
    for p in glob.glob(str(E4 / "lc-longmappo-step*-seed*")):
        st = int(Path(p).name.split("-step")[1].split("-")[0])
        r = summ(p)
        if r:
            pts.setdefault(st, []).append(r[0])
    lm = res("mappo-long", "validation")
    if len(lm):
        pts[1024000] = lm
    if pts:
        curve({s: np.array(v) for s, v in pts.items()}, "mappo-long", mk="s", label="MAPPO, 5× budget")
    pts = {}
    for p in glob.glob(str(E4 / "lc-mappo-step*-seed*")):
        st = int(Path(p).name.split("-step")[1].split("-")[0])
        r = summ(p)
        if r:
            pts.setdefault(st, []).append(r[0])
    m = res("mappo", "validation")
    if len(m):
        pts[204800] = m
    if pts:
        curve({s: np.array(v) for s, v in pts.items()}, "mappo", mk="^", label="MAPPO")
    pts = {}
    for p in glob.glob(str(E4 / "lc-shapeonly-step*-seed*")):
        st = int(Path(p).name.split("-step")[1].split("-")[0])
        r = summ(p)
        if r:
            pts.setdefault(st, []).append(r[0])
    sh = res("shaping", "validation")
    if len(sh):
        pts[204800] = sh
    if pts:
        curve({s: np.array(v) for s, v in pts.items()}, "shaping", mk="D", label="Shaping only")
    two = res("anchor-2upd", "validation")
    if len(two):
        ax.scatter([128], [two.mean()], marker="*", s=48, color=OURS, edgecolor="white", lw=0.5, zorder=6)
        ax.text(150, two.mean() - 0.075, "after 2 updates", fontsize=5.8, color=OURS, va="top")
        note(f"fig8 anchor-2upd at 128 steps: {two.mean():.3f}")
    v = res("phys30", "validation")
    if len(v):                                          # one reference line only: the anchor with no training
        ax.axhline(v.mean(), color=OURS, ls=(0, (4, 2)), lw=0.7, zorder=1)
        ax.text(130, v.mean() + 0.012, "anchor only, no training", fontsize=5.8, color=OURS, va="bottom")
    # direct labels at the right end of each curve (no legend box)
    END.sort(key=lambda e: e[0])
    ys = [e[0] for e in END]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.055)
    for (y0, x0, t, c, ours), y in zip(END, ys):
        ax.text(1.5e6, y, t, fontsize=5.8, color=c, va="center", fontweight="bold" if ours else None, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(90, 1.35e6)
    ax.set_ylim(0, 1.03)
    ax.axvline(204800, color="#C3C9D0", lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.text(215000, 0.5, "standard\nbudget", fontsize=5.6, color="#6B747D", va="center", ha="left")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Survivors detected (val.)")
    ax.grid(axis="y")
    save(fig, "figR8_learning_curves")


# =============================================================== Fig. 9 ablation
def bars(ax, xs, groups, col, width):
    rng = np.random.default_rng(3)
    for x, v in zip(xs, groups):
        if not len(v):
            continue
        ax.bar(x, v.mean(), width, color=col, alpha=0.85, lw=0, zorder=2)
        if len(v) > 1:
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color="#222222", lw=0.7, capsize=1.8, capthick=0.6, zorder=4)
        ax.scatter(x + rng.uniform(-0.28, 0.28, len(v)) * width, v, s=4, color="white", edgecolor="#1A1A1A",
                   lw=0.35, zorder=5)


def physline(ax, pts, col, ls, label):
    xs = sorted(pts)
    if xs:
        ax.plot(xs, [pts[x] for x in xs], color=col, ls=ls, lw=1.1, marker="o", ms=2.6, mec="white", mew=0.4,
                label=label, zorder=6)


def fig_ablation():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.32, width_ratios=[1.35, 1, 0.9]))
    # (a) angle
    ax = axs[0]
    learned = {0: res("anchor0"), 15: res("anchor15"), 30: res("anchor30")}
    bars(ax, list(learned), list(learned.values()), OURS_M, 6.0)
    for split, col, ls in (("test", OURS, "-"), ("validation", SPLITC["validation"], (0, (3, 1.2)))):
        pts = {}
        for th in (0, 15, 30, 45, 60, 75, 90):
            r = summ(E3 / f"zero-{split}-ang{th}")
            if r:
                pts[th] = r[0]
        for th in (5, 10, 20, 25, 35, 40, 50):
            r = summ(E4 / f"sens-{split}-theta{th}")
            if r:
                pts[th] = r[0]
        note(f"fig9a physics-only {split}: " + ", ".join(f"{t}:{pts[t]:.3f}" for t in sorted(pts)))
        physline(ax, pts, col, ls, f"anchor only ({split})")
        if 90 in pts:
            ax.text(87, pts[90] + (0.04 if split == "validation" else -0.06), f"{pts[90]:.2f}", fontsize=5.7,
                    color=col, ha="right")
    for th, v in learned.items():
        note(f"fig9a learned theta={th}: {v.mean():.3f}±{v.std(ddof=1):.3f} (n={len(v)})")
    ax.set_xlabel(r"Rotation angle $\theta$ (deg)")
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlim(-6, 94)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("Survivors detected")
    ax.grid(axis="y")
    ax.legend(handles=[Patch(color=OURS_M, label="anchor + residual (test, seeds)"),
                       Line2D([], [], color=OURS, marker="o", ms=2.6, label="anchor only (test)"),
                       Line2D([], [], color=SPLITC["validation"], ls=(0, (3, 1.2)), marker="o", ms=2.6,
                              label="anchor only (validation)")], fontsize=5.7, loc="lower left")
    head(ax, "a", "Rotation angle")
    # (b) strength
    ax = axs[1]
    learned = {2: res("anchor-str2"), 4: res("anchor15"), 6: res("anchor-str6"), 8: res("anchor-str8")}
    bars(ax, list(learned), list(learned.values()), OURS_M, 1.1)
    for th, pre, col, ls in ((15, "sens15-test-alpha", OURS, "-"), (30, "sens-test-alpha", OURS_L, (0, (3, 1.2)))):
        pts = {}
        for al in (1, 2, 3, 4, 6, 8, 12):
            r = summ(E4 / f"{pre}{al}")
            if r:
                pts[al] = r[0]
        if th == 30 and 4 not in pts and len(res("phys30")):
            pts[4] = res("phys30").mean()
        physline(ax, pts, col if th == 15 else "#7FA7D6", ls, rf"anchor only, $\theta={th}^{{\circ}}$")
        note(f"fig9b physics-only theta={th}: " + ", ".join(f"{a}:{pts[a]:.3f}" for a in sorted(pts)))
    ax.set_xlabel(r"Anchor strength $\alpha$ (bars: $\theta=15^{\circ}$)")
    ax.set_xticks([1, 2, 4, 6, 8, 12])
    ax.set_xlim(0.2, 12.8)
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y")
    ax.legend(fontsize=5.7, loc="lower right", framealpha=0.95)
    head(ax, "b", "Anchor strength")
    # (c) shaping
    ax = axs[2]
    keys = [("anchor15", 0), ("shape0.5-anchor", 0.5), ("shape1-anchor-s4", 1), ("shape2.0-anchor", 2),
            ("shape4.0-anchor", 4)]
    vs = [res(k) for k, _ in keys]
    bars(ax, range(len(keys)), vs, OURS_M, 0.7)
    p15 = res("phys15")
    if len(p15):
        ax.axhline(p15.mean(), color=OURS, lw=0.9, ls=(0, (3, 1.2)), label=r"anchor only, $\theta=15^{\circ}$")
        ax.legend(fontsize=5.7, loc="lower right", framealpha=0.95)
    for (k, b), v in zip(keys, vs):
        note(f"fig9c shaping beta={b}: {v.mean():.3f}±{(v.std(ddof=1) if len(v) > 1 else 0):.3f} (n={len(v)})")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(b) for _, b in keys])
    ax.set_xlabel(r"Shaping coefficient $\beta$")
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y")
    head(ax, "c", "Potential shaping")
    save(fig, "figR9_ablation")


# =============================================================== Fig. 11 + Table 5 operational stress
CONDS = [("nominal", "Nominal"), ("det06", r"$p_d=0.6$"), ("drift03", "drift 0.3 m/s"), ("drift05", "drift 0.5 m/s"),
         ("drop05", "packet loss 0.5"), ("hard", "combined")]
CONDS_FIG = ["Nominal", "$p_d=0.6$", "drift\n0.3 m/s", "drift\n0.5 m/s", "packet\nloss 0.5", "combined"]
SROWS = [("phys30", "phys30", None), ("anchor30", "anchor30-seed{}", 3), ("sector-search", "sector-search", None),
         ("expanding-square", "expanding-square", None), ("aco-search", "aco-search", None),
         ("pso-search", "pso-search", None), ("mpc-horizon", "mpc-horizon", None), ("pvf-greedy", "pvf-greedy", None),
         ("cbba", "cbba", None), ("bayesian-information-gain", "bayesian-information-gain", None),
         ("lawnmower-coverage", "lawnmower-coverage", None), ("ic3net", "ic3net-seed0", None),
         ("pvf-mappo", "pvfmappo-seed0", None), ("ippo", "ippo-seed0", None)]


def stress_matrix():
    rows, M = [], []
    for k, name, nseed in SROWS:
        r = []
        for c, _ in CONDS:
            if c == "nominal":
                v = res(k)
                if nseed:
                    v = v[:nseed]
                elif k in ("ic3net", "pvf-mappo", "ippo"):
                    v = v[:1]
                r.append(v.mean() if len(v) else np.nan)
            else:
                if nseed:
                    v = [summ(E4 / f"stress-{c}-{name.format(s)}") for s in range(nseed)]
                    v = [x[0] for x in v if x]
                    r.append(np.mean(v) if v else np.nan)
                else:
                    x = summ(E4 / f"stress-{c}-{name}")
                    r.append(x[0] if x else np.nan)
        rows.append(k)
        M.append(r)
    return rows, np.array(M, float)


def fig_robust():
    rows, M = stress_matrix()
    keep = [i for i in range(len(rows)) if np.isfinite(M[i, 1:]).any()]
    rows, M = [rows[i] for i in keep], M[keep]
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.55, 1.0], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("succ", ["#F7F9FB", "#C9DBEC", "#7FA7D6", "#2F6BAA", "#123A6B"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.9,
                        color="white" if M[i, j] > 0.62 else "#1A1A1A",
                        fontweight="bold" if METHOD[rows[i]][1] == "ours" else None)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(CONDS_FIG, fontsize=6.3, linespacing=0.95)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab(k) for k in rows], fontsize=6.3)
    for t, k in zip(ax.get_yticklabels(), rows):
        t.set_color(mcolor(k))
        if METHOD[k][1] == "ours":
            t.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.02, 1.17, "a", transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="right")
    ax.text(0.0, 1.17, "Survivors detected on the held-out test split, no retraining", transform=ax.transAxes,
            fontsize=7.5)
    ax = fig.add_subplot(gs[1])
    drop = M[:, -1] - M[:, 0]
    ys = np.arange(len(rows))
    ax.barh(ys, drop, 0.62, color=[mcolor(k) for k in rows], alpha=0.85, lw=0)
    for y, d in zip(ys, drop):
        if np.isfinite(d):
            ax.text(d - 0.005 if d < 0 else d + 0.005, y, f"{d * 100:+.1f}", va="center",
                    ha="right" if d < 0 else "left", fontsize=5.8)
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_yticks([])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    lo = np.nanmin(drop)
    ax.set_xlim(min(lo * 1.35, -0.05), 0.04)
    ax.set_xlabel("Combined stress − nominal (labels in points)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    head(ax, "b", "Loss under combined stress", y=1.17)
    for k, r in zip(rows, M):
        note(f"fig11 {k:26s} " + " ".join(f"{c}:{v:.3f}" for (c, _), v in zip(CONDS, r)))
    save(fig, "figR11_robustness")
    # Table 5 (LaTeX), same numbers
    L = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "c" * len(CONDS) + "}", r"\toprule",
         "Method & " + " & ".join(c[1] for c in CONDS) + r" \\", r"\midrule"]
    best = np.nanmax(M, axis=0)
    for k, r in zip(rows, M):
        cells = [("--" if not np.isfinite(v) else (r"$\mathbf{" + f"{v:.3f}" + "}$" if abs(v - b) < 5e-4 else f"${v:.3f}$"))
                 for v, b in zip(r, best)]
        L.append(f"{lab(k)} & " + " & ".join(cells) + r" \\")
    L[-1] = L[-1]
    L += [r"\bottomrule", r"\end{tabular*}"]
    out = PR / "runs/complete-paper-v4/tables-r3"
    out.mkdir(exist_ok=True)
    (out / "tab_robust.tex").write_text("\n".join(L) + "\n")
    return
    L += [r"\bottomrule", r"\end{tabular}"]
    out = PR / "runs/complete-paper-v4/tables-r3"
    out.mkdir(exist_ok=True)
    (out / "tab_robust.tex").write_text("\n".join(L) + "\n")


# =============================================================== Fig. 12 survivor count and team size
CNT = ["phys30", "anchor30", "sector-search", "expanding-square", "aco-search", "pso-search", "mpc-horizon",
       "pvf-greedy", "cbba", "ic3net"]
TEAM = ["phys30", "sector-search", "expanding-square", "aco-search", "pvf-greedy", "cbba", "lawnmower-coverage"]


def fig_count():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]
    for k in CNT:
        pts = {}
        for G in range(1, 7):
            if G == 3:
                v = res(k)[:1] if k == "ic3net" else res(k)
            elif k == "anchor30":
                v = np.concatenate([many(E4 / f"count-G{G}-anchor30-seed*"),
                                    many(E3 / f"mis-G{G}K{G}-ang30-seed*")])
            elif k == "ic3net":
                v = many(E4 / f"count-G{G}-ic3net-seed0")
            elif k == "phys30":
                r = summ(E4 / f"count-G{G}-phys30") or summ(E3 / f"mis-G{G}K{G}-zero30")
                v = np.array([r[0]]) if r else np.array([])
            else:
                r = summ(E4 / f"count-G{G}-{k}") or summ(E3 / f"mis-G{G}-{k}")
                v = np.array([r[0]]) if r else np.array([])
            if len(v):
                pts[G] = v.mean()
        _, typ, ls = METHOD[k]
        xs = sorted(pts)
        if xs:
            ax.plot(xs, [pts[x] for x in xs], color=TYPE_COLOR[typ], ls=ls, lw=1.8 if typ == "ours" else 1.0,
                    marker="o" if typ == "ours" else ".", ms=3.6, mec="white", mew=0.4, label=lab(k),
                    zorder=5 if typ == "ours" else 3)
            note(f"fig12a {k:22s} " + ", ".join(f"G={x}:{pts[x]:.3f}" for x in xs))
    ax.set_xlabel("Number of survivors $G$ (anchor extracts $K=G$ peaks)")
    ax.set_ylabel("Survivors detected (test)")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y")
    ax.legend(fontsize=5.4, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 0.225), handlelength=1.7,
              columnspacing=0.6, labelspacing=0.26)
    head(ax, "a", "Survivor count")
    ax = axs[1]
    for k in TEAM:
        pts = {}
        for N in (2, 3, 4, 6, 8, 10, 12):
            if N == 6:
                v = res(k)
                r = (v.mean(),) if len(v) else None
            else:
                r = summ(E4 / f"team{N}-{k}")
            if r:
                pts[N] = r[0]
        _, typ, ls = METHOD[k]
        xs = sorted(pts)
        if xs:
            ax.plot(xs, [pts[x] for x in xs], color=TYPE_COLOR[typ], ls=ls, lw=1.8 if typ == "ours" else 1.0,
                    marker="o" if typ == "ours" else ".", ms=3.6, mec="white", mew=0.4, label=lab(k),
                    zorder=5 if typ == "ours" else 3)
            note(f"fig12b {k:22s} " + ", ".join(f"N={x}:{pts[x]:.3f}" for x in xs))
    ax.set_xlabel("Number of UAVs $N$ (3 survivors)")
    ax.set_xticks([2, 3, 4, 6, 8, 10, 12])
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y")
    ax.legend(fontsize=5.6, ncol=2, loc="lower right", handlelength=1.7, columnspacing=0.6, labelspacing=0.26)
    head(ax, "b", "Team size (network-free policies)")
    save(fig, "figR12_count_team")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["curves", "ablation", "robust", "count"]):
        globals()["fig_" + name]()
