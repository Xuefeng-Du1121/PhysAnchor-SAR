"""Fig. 5 (mission storyboard) and Fig. 6 (methods on the same episode), identical size and style.

Scenario: ood-jan08-area5, the OOD box with the strongest current (median 0.85 m/s), chosen by that
criterion only; canonical seed (repeat 0). Recorded with tools/eval_rec.py (--record-full).
Fig. 5 top row: currents of the whole 40 km box with the belief particles and the survivor drift;
       bottom row: the zoomed search with the orange physics value field, at the same four moments
       (start and the three detections).
Fig. 6: the same episode for eight methods; each panel reports this episode and, in grey, the mean
       over the 10 evaluation episodes of this scenario (from the OOD evaluation).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
from r3_common import E3, E4, ocean_local, smooth_field, scalebar, note, save
from style_r3 import (VALUE, TRACK, TRACK_L, HALO, OURS, PLAN, PATTERN, SWARM, MARL, MARL_IN, FOUND, OCEAN,
                      OCEAN_SOFT, head, frame)

SID = "ood-jan08-area5"
km = lambda x: np.asarray(x) / 1000.0  # noqa: E731
UCOL = [TRACK[0], TRACK[1], TRACK[2], TRACK_L[0], TRACK_L[1], TRACK_L[2]]   # pair = same hue, two tones


def rec(name):
    return np.load(E4 / name / f"traj-{SID}-rep0.npz")


def fade(ax, xy, color, lw=0.9, a0=0.18, z=5, recent=None):
    """Track whose opacity grows with time (old -> faint, recent -> solid).
    With recent=n only the last n steps are drawn solid and everything older stays at a0."""
    if len(xy) < 2:
        return
    seg = np.stack([xy[:-1], xy[1:]], 1)
    rgba = np.tile(to_rgba(color), (len(seg), 1))
    if recent is None:
        rgba[:, 3] = np.linspace(a0, 1.0, len(seg))
    else:
        rgba[:, 3] = a0
        n = min(recent, len(seg))
        rgba[-n:, 3] = np.linspace(a0, 1.0, n)
    ax.add_collection(LineCollection(seg, colors=rgba, linewidths=lw, capstyle="round", zorder=z))


def survivors(ax, z, k, ms=4.2):
    det = z["detected_at"]
    for g in range(z["targets"].shape[1]):
        found_k = bool(z["terminal_found"][g]) and np.isfinite(det[g]) and det[g] <= z["t"][k] + 30.0
        p = km(z["targets"][k, g])
        if found_k:
            ax.plot(*p, "o", ms=ms, color=FOUND, mec="white", mew=0.6, zorder=9)
        else:
            ax.plot(*p, "X", ms=ms, color="#1A1A1A", mec="white", mew=0.45, zorder=9)


# ================================================================ Fig. 5 storyboard
z = rec("rec-anchor30-s3")
t = z["t"]
nT = len(t)
det = np.sort(z["detected_at"][np.isfinite(z["detected_at"])])
ev = [int(np.searchsorted(t, d)) for d in det]
ks = [0] + [min(e, nT - 1) for e in ev]
while len(ks) < 4:
    ks.append(nT - 1)
ks = sorted(set(ks))[:4]
while len(ks) < 4:
    ks.insert(-1, (ks[-2] + ks[-1]) // 2)
side = int(round(np.sqrt(z["grid"].shape[0])))
gx = z["grid"][:side, 0]
v0 = z["pvf"][0].max()
U = z["uavs"]
P = np.concatenate([U.reshape(-1, 2), z["particles"][ks].reshape(-1, 2), z["targets"].reshape(-1, 2)])
lo, hi = np.percentile(P, 0.5, axis=0) - 1200, np.percentile(P, 99.5, axis=0) + 1200
span = (hi - lo).max()
c = (lo + hi) / 2
ext = [km(c[0] - span / 2), km(c[0] + span / 2), km(c[1] - span / 2), km(c[1] + span / 2)]
fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.75), gridspec_kw=dict(wspace=0.06, hspace=0.1))
for j, k in enumerate(ks):
    # top: whole box, currents + belief particles + drift so far
    ax = axs[0, j]
    m = ocean_local(ax, z["flow"][k], z["flow_axis"], vmax=1.2, cmap=OCEAN, density=0.9, lw=0.4)
    for g in range(z["particles"].shape[1]):
        if not (z["terminal_found"][g] and z["detected_at"][g] <= t[k] + 30.0):
            pp = z["particles"][k, g]
            ax.scatter(km(pp[:, 0]), km(pp[:, 1]), s=0.5, color="#F2A65A", alpha=0.75, lw=0, zorder=3,
                       rasterized=True)
        ax.plot(km(z["targets"][:k + 1, g, 0]), km(z["targets"][:k + 1, g, 1]), color="#1A1A1A", lw=0.8, zorder=4)
    for i in range(U.shape[1]):
        ax.plot(*km(U[k, i]), "o", ms=2.2, color=UCOL[i], mec="white", mew=0.35, zorder=6)
    survivors(ax, z, k, ms=3.4)
    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, ec="white", lw=0.8,
                           ls=(0, (2, 1.2)), zorder=7))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect("equal")
    frame(ax)
    lab = "start" if j == 0 else f"survivor {j} found"
    ax.text(0.04, 0.95, f"t = {t[k] / 60:.0f} min", transform=ax.transAxes, fontsize=6.8, fontweight="bold",
            va="top", path_effects=HALO, zorder=10)
    ax.text(0.96, 0.95, lab, transform=ax.transAxes, fontsize=6.2, ha="right", va="top", path_effects=HALO,
            zorder=10)
    if j == 0:
        scalebar(ax, 38, 1.6, km=10)
    # bottom: zoomed search with the value field
    ax = axs[1, j]
    ocean_local(ax, z["flow"][k], z["flow_axis"], vmax=1.2, cmap=OCEAN_SOFT, stream=False)
    vk = max(z["pvf"][k].max(), 1e-30)      # per-panel scale: the shape of what is left, not its size
    X, Y, val = smooth_field(km(gx), km(gx), z["pvf"][k].reshape(side, side) / vk, k=4, order=3)
    ax.contourf(X, Y, np.clip(val, 0, 1), levels=[0.03, 0.12, 0.25, 0.45, 0.65, 0.85, 1.01], cmap=VALUE, vmin=0,
                vmax=1, alpha=0.93, zorder=2)
    ax.text(0.96, 0.95, f"max value ×{z['pvf'][k].max() / v0:.3f}", transform=ax.transAxes, fontsize=6.0,
            ha="right", va="top", path_effects=HALO, zorder=10)
    for i in range(U.shape[1]):
        fade(ax, km(U[:k + 1, i]), UCOL[i], lw=1.0, a0=0.12, recent=50)
        ax.add_patch(Circle(km(U[k, i]), km(z["sensing_m"]), fc=UCOL[i], ec="none", alpha=0.18, zorder=5))
        ax.plot(*km(U[k, i]), "o", ms=3.0, color=UCOL[i], mec="white", mew=0.5, zorder=8)
    for g in range(z["targets"].shape[1]):
        ax.plot(km(z["targets"][:k + 1, g, 0]), km(z["targets"][:k + 1, g, 1]), color="#1A1A1A", lw=0.6,
                ls=(0, (1, 1)), zorder=6)
    survivors(ax, z, k)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    frame(ax)
    if j == 0:
        scalebar(ax, ext[1] - 0.5, ext[2] + 0.6, km=2)
    note(f"fig5 k={k} t={t[k] / 60:.1f}min field max {z['pvf'][k].max() / v0:.3f}")
for j in range(4):
    head(axs[0, j], "abcd"[j], "")
    head(axs[1, j], "efgh"[j], "")
axs[0, 0].set_ylabel("Ocean and belief\n(40 km box)", fontsize=7.0, labelpad=3)
axs[1, 0].set_ylabel("Search\n(dashed window)", fontsize=7.0, labelpad=3)
note(f"fig5 {SID}: detections at {np.round(det / 60, 1).tolist()} min, snapshots {ks}, "
     f"survivor drift {np.round(np.linalg.norm(z['targets'][-1] - z['targets'][0], axis=1) / 1000, 2).tolist()} km "
     f"over {t[-1] / 60:.0f} min, median current {np.median(np.hypot(*np.moveaxis(z['flow'][0], -1, 0))):.3f} m/s")
save(fig, "figR5_mission")

# ================================================================ Fig. 6 methods on the same episode
METHODS = [("rec-anchor30-s3", "Physics anchor (ours)", OURS, [E3 / f"ood-anchor-ang30-seed{s}" for s in range(10)]),
           ("rec-pvf-greedy", "PVF-greedy", PLAN, [E4 / "ood-pvf-greedy"]),
           ("rec-cbba", "CBBA", PLAN, [E4 / "ood-cbba"]),
           ("rec-sector-search", "Sector search", PATTERN, [E4 / "ood-sector-search"]),
           ("rec-expanding-square", "Expanding square", PATTERN, [E4 / "ood-expanding-square"]),
           ("rec-aco-search", "Ant colony", SWARM, [E4 / "ood-aco-search"]),
           ("rec-mappo-s0", "MAPPO", MARL, [E4 / f"ood-mappo-seed{s}" for s in range(5)]),
           ("rec-pvfmappo-s0", "PVF + MAPPO", MARL_IN, [E3 / f"ood-pvf-mappo-seed{s}" for s in range(10)])]


def scen_mean(dirs):
    v = []
    for d in dirs:
        p = d / "summary.json"
        if p.exists():
            ps = json.load(open(p)).get("per_scenario", {}).get(SID)
            if ps:
                v.append((ps["success_rate"], ps["complete_rescue"]))
    return (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), len(v)) if v else None


Z = {n: rec(n) for n, *_ in METHODS}
# common window: where the belief-driven / pattern / swarm methods fly (learned baselines that wander off
# the prior leave it; their tracks are clipped at the frame)
allxy = np.concatenate([Z[n]["uavs"].reshape(-1, 2) for n, _, col, _ in METHODS if col not in (MARL, MARL_IN)])
tg = np.concatenate([Z[n]["targets"].reshape(-1, 2) for n in Z])          # every survivor drift stays visible
lo = np.clip(np.minimum(np.percentile(allxy, 1, axis=0), tg.min(0)) - 1200, 0, 4e4)
hi = np.clip(np.maximum(np.percentile(allxy, 99, axis=0), tg.max(0)) + 1200, 0, 4e4)
span = (hi - lo).max()
c = np.clip((lo + hi) / 2, span / 2, 4e4 - span / 2)
ex6 = [km(c[0] - span / 2), km(c[0] + span / 2), km(c[1] - span / 2), km(c[1] + span / 2)]
z0 = Z[METHODS[0][0]]
X, Y, val0 = smooth_field(km(gx), km(gx), z0["pvf"][0].reshape(side, side) / z0["pvf"][0].max(), k=4, order=3)
fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.75), gridspec_kw=dict(wspace=0.06, hspace=0.16))
for q, (n, lab, col, dirs) in enumerate(METHODS):
    ax = axs.ravel()[q]
    zz = Z[n]
    ocean_local(ax, zz["flow"][0], zz["flow_axis"], vmax=1.2, cmap=OCEAN_SOFT, density=0.7, lw=0.35)
    ax.contourf(X, Y, np.clip(val0, 0, 1), levels=[0.03, 1.01], colors=["white"], alpha=0.55, zorder=1.8)
    ax.contourf(X, Y, np.clip(val0, 0, 1), levels=[0.03, 0.12, 0.25, 0.45, 0.65, 0.85, 1.01], cmap=VALUE, vmin=0,
                vmax=1, alpha=0.6, zorder=2)
    for i in range(zz["uavs"].shape[1]):
        fade(ax, km(zz["uavs"][:, i]), UCOL[i], lw=0.85, a0=0.25)
    for g in range(zz["targets"].shape[1]):
        ax.plot(km(zz["targets"][:, g, 0]), km(zz["targets"][:, g, 1]), color="#1A1A1A", lw=0.6, ls=(0, (1, 1)),
                zorder=6)
        d = zz["detected_at"][g]
        if np.isfinite(d) and zz["terminal_found"][g]:
            kk = min(int(np.searchsorted(zz["t"], d)), len(zz["t"]) - 1)
            ax.plot(*km(zz["targets"][kk, g]), "o", ms=4.0, color=FOUND, mec="white", mew=0.6, zorder=9)
        else:
            ax.plot(*km(zz["targets"][-1, g]), "X", ms=4.2, color="#1A1A1A", mec="white", mew=0.45, zorder=9)
    ax.set_xlim(ex6[0], ex6[1])
    ax.set_ylim(ex6[2], ex6[3])
    ax.set_aspect("equal")
    frame(ax, color=col, lw=1.1 if q == 0 else 0.6)
    nf = int(zz["terminal_found"].sum())
    ttd = np.where(zz["terminal_found"], zz["detected_at"], 10800.0).mean() / 10800.0
    ax.text(0.04, 0.95, lab, transform=ax.transAxes, fontsize=6.8, fontweight="bold", color=col, va="top",
            path_effects=HALO, zorder=10)
    sm = scen_mean(dirs)
    ax.text(0.04, 0.155, f"{nf}/3 found · TTD {ttd:.2f}", transform=ax.transAxes, fontsize=6.4,
            path_effects=HALO, zorder=10)
    ax.text(0.04, 0.05, f"scenario mean {sm[0]:.2f}" if sm else "scenario mean –",
            transform=ax.transAxes, fontsize=6.0, color="#4A5560", path_effects=HALO, zorder=10)
    note(f"fig6 {lab:22s} this episode found {nf}/3 ttd {ttd:.3f}; scenario mean "
         f"{'n/a' if not sm else f'succ {sm[0]:.3f} all {sm[1]:.3f} over {sm[2]} runs'}")
    head(ax, "abcdefgh"[q], "")
scalebar(axs[0, 0], ex6[1] - 0.6, ex6[3] - 0.18 * (ex6[3] - ex6[2]), km=2)
save(fig, "figR6_methods_same_episode")
