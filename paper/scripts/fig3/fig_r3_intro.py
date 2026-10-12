"""Introduction figure: why maritime SAR is a physics problem with a sparse reward.

(a) Copernicus currents around a training scenario and its 40x40 km search box.
(b-d) Passive evolution of the public belief over 3 h with no search: particles advected by the
      currents, the physics value field (orange) and the true survivor tracks.
(e) The sparse reward: fraction of held-out test episodes in which the swarm has received its first
    reward (first detection) by time t, against the survival value exp(-t/tau) of an undetected survivor.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from r3_common import (VIZ, ocean_region, ocean_local, lonlat_ticks, scalebar, box_corners, note, save,
                       smooth_field, TAU_S, DT, T)
from style_r3 import (VALUE, OURS, PLAN, MARL, MARL_IN, PEVA, HALO, head, frame, SPLITC, OCEAN)

reg = np.load(VIZ / "region_train-jan03-area1.npz")
pz = np.load(VIZ / "passive_train-jan03-area1.npz")
fig = plt.figure(figsize=(7.0, 3.3))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.32, 3.0], wspace=0.2)
right = outer[0, 1].subgridspec(2, 3, height_ratios=[1.0, 0.86], hspace=0.42, wspace=0.07)

# (a) regional map --------------------------------------------------------------------------------
ax = fig.add_subplot(outer[0, 0])
lon, lat = reg["lon"], reg["lat"]
m = ocean_region(ax, lon, lat, reg["u"], reg["v"], vmax=1.2, density=1.7)
o = reg["origin"]
c = box_corners(o)
ax.plot(c[:, 0], c[:, 1], color=SPLITC["test"], lw=1.3, zorder=8)
ax.annotate("search box\n40×40 km", (c[1, 0], c[1, 1]), xytext=(14, -26), textcoords="offset points",
            fontsize=6.4, color=SPLITC["test"], fontweight="bold", path_effects=HALO, zorder=9,
            arrowprops=dict(arrowstyle="-", color=SPLITC["test"], lw=0.6))
ax.set_xlim(max(lon.min(), o[0] - 4.3), min(lon.max(), o[0] + 4.3))
ax.set_ylim(max(lat.min(), o[1] - 9.6), min(lat.max(), o[1] + 1.9))
lonlat_ticks(ax, range(118, 140, 3), range(18, 36, 3))
cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.04, pad=0.075, aspect=24,
                  ticks=[0, 0.3, 0.6, 0.9, 1.2])
cb.set_label("Surface current (m s$^{-1}$)", fontsize=6.6, labelpad=1.5)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
head(ax, "a", "Copernicus currents, 3 Jan 2018")
note(f"intro a: origin {o.tolist()}, start {reg['start_utc']}")

# (b-d) belief snapshots ---------------------------------------------------------------------------
km = lambda x: np.asarray(x) / 1000.0  # noqa: E731
t_all, tr = pz["t"], pz["tracks"]
ks = [int(np.argmin(np.abs(t_all - s))) for s in (0, 3600, t_all[-1])]
side = int(round(np.sqrt(len(pz["grid"]))))
gx = pz["grid"][:side, 0]
v0 = pz["pvf"][ks[0]].max()
P = np.concatenate([pz["particles"][k].reshape(-1, 2) for k in ks] + [tr.reshape(-1, 2)])
lo, hi = np.percentile(P, 0.5, axis=0) - 1800, np.percentile(P, 99.5, axis=0) + 1800
span = (hi - lo).max()
ctr = (lo + hi) / 2
ext = (km(ctr[0] - span / 2), km(ctr[0] + span / 2), km(ctr[1] - span / 2), km(ctr[1] + span / 2))
for j, k in enumerate(ks):
    ax = fig.add_subplot(right[0, j])
    ocean_local(ax, pz["flow"][k], pz["flow_axis"], vmax=1.2, cmap=OCEAN, density=1.0, lw=0.4)
    X, Y, val = smooth_field(km(gx), km(gx), pz["pvf"][k].reshape(side, side) / v0, k=4, order=3)
    ax.contourf(X, Y, np.clip(val, 0, 1), levels=[0.04, 0.15, 0.3, 0.5, 0.7, 0.9, 1.01], cmap=VALUE,
                vmin=0, vmax=1, alpha=0.95, zorder=2)
    for g in range(pz["particles"].shape[1]):
        p = pz["particles"][k, g]
        ax.scatter(km(p[:, 0]), km(p[:, 1]), s=0.25, color="#7A3E12", alpha=0.35, lw=0, zorder=3, rasterized=True)
    step = min(int(round(t_all[k] / DT)), len(tr) - 1)
    for g in range(tr.shape[1]):
        ax.plot(km(tr[:step + 1, g, 0]), km(tr[:step + 1, g, 1]), color="#1A1A1A", lw=0.85, zorder=5)
        ax.plot(*km(tr[0, g]), "o", ms=2.1, mfc="white", mec="#1A1A1A", mew=0.55, zorder=6)
        ax.plot(*km(tr[step, g]), "X", ms=4.0, color="#1A1A1A", mec="white", mew=0.4, zorder=7)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    frame(ax)
    h = t_all[k] / 3600
    ax.text(0.04, 0.95, f"t = {h:.0f} h", transform=ax.transAxes, fontsize=6.8, fontweight="bold", va="top",
            path_effects=HALO, zorder=10)
    ax.text(0.96, 0.95, f"value ×{np.exp(-t_all[k] / TAU_S):.2f}", transform=ax.transAxes, fontsize=6.2,
            ha="right", va="top", path_effects=HALO, zorder=10)
    scalebar(ax, ext[1] - 0.8, ext[2] + 0.8, km=5)
    if j == 0:
        head(ax, "b", "Belief without search: drift and spread over the 3-h mission")
    note(f"intro b{j + 1}: t={h:.2f}h survivor displacement "
         f"{np.round(np.linalg.norm(tr[step] - tr[0], axis=1) / 1000, 2).tolist()} km, "
         f"field max {pz['pvf'][k].max() / v0:.3f} of t0")

# (c) sparse reward: share of test episodes with a first detection by time t --------------------------
# Kept inside the axes: short labels at the right end of each curve, no rotated annotation.
ax = fig.add_subplot(right[1, :])
z = np.load(VIZ / "test_traces.npz")
tg = np.arange(0, T + DT, DT)
spec = [("anchor30", "Anchor (ours)", OURS, "-", 1.7), ("cl_pvf-greedy", "PVF-greedy", PLAN, "-", 1.1),
        ("ic3net", "IC3Net", MARL, "-", 1.0), ("mappo", "MAPPO", MARL, (0, (3, 1.3)), 1.0),
        ("pvfmappo", "PVF+MAPPO", MARL_IN, "-", 1.0), ("pevagen", "PEVA-Gen", PEVA, "-", 1.0)]
for key, lab, col, ls, lw in spec:
    first = np.min(z["det_" + key], axis=2)
    mu = np.array([[np.mean(f <= t) for t in tg] for f in first]).mean(0)
    ax.plot(tg / 3600, mu, color=col, ls=ls, lw=lw, solid_capstyle="round", zorder=4, label=lab)
    note(f"intro e {lab:22s} P(first detection by 0.5h) {mu[60]:.3f}, 1h {mu[120]:.3f}, 3h {mu[-1]:.3f} "
         f"(seeds={len(first)}, episodes/seed={first.shape[1]})")
surv = np.exp(-tg / TAU_S)
ax.fill_between(tg / 3600, 0, surv, color="#DDE3E9", alpha=0.5, lw=0, zorder=1)
ax.plot(tg / 3600, surv, color="#8A939C", lw=0.7, ls=(0, (1, 1)), zorder=2, label=r"survival $e^{-t/\tau}$")
# legend in the empty band between the anchored curves (~1) and the learned ones (<0.4), clear of the
# survival curve, which passes this band before 1.6 h
ax.legend(ncol=4, fontsize=5.8, loc="center", bbox_to_anchor=(0.69, 0.63), handlelength=1.5, columnspacing=0.8,
          labelspacing=0.25, frameon=False, borderpad=0.25)
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.04)
ax.set_xlabel("Mission time (h)", labelpad=1.5)
ax.set_ylabel("Episodes", labelpad=2)
ax.grid(axis="y")
head(ax, "c", "First detection (first non-zero reward), test episodes")
save(fig, "figR1_intro")
