"""Method figure from real data: from currents to anchored actions, one decision step (1 x 4 panels).

Held-out test episode (22 Jan, seed 820000), theta = 30 deg policy, step k = 30 (t = 15 min):
(a) currents, UAV tracks so far, particle belief and hidden survivors, (b) physics value field,
(c) NMS peaks with suppression radius, (d) role assignment and rotated anchors.
The heading-residual histogram of the whole episode is saved separately (figR2_residual) and used as a
data inset of the architecture figure (Fig. 2c).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from r3_common import VIZ, ocean_local, smooth_field, scalebar, note, save
from style_r3 import VALUE, TRACK, HALO, OURS, OCEAN_SOFT, head, frame

z = np.load(VIZ / "replay_anchor30_test_ep0.npz")
K = 30
t, U, X, PV, PK, PT, AD, A = (z["t"], z["uavs"], z["targets"], z["pvf"], z["peaks"], z["particles"],
                               z["anchor_dir"], z["action"])
det = np.nan_to_num(z["detected_at"].astype(float), nan=np.inf)
km = lambda x: np.asarray(x) / 1000.0  # noqa: E731
side = int(round(np.sqrt(len(z["grid"]))))
gx = z["grid"][:side, 0]
pk_ok = np.all(np.isfinite(PK[K]), axis=1)
alive = ~(det <= t[K])
pts = np.concatenate([U[:K + 1].reshape(-1, 2), PT[K][alive].reshape(-1, 2), PK[K][pk_ok]])
lo = np.minimum(np.percentile(pts, 1, axis=0), U[:K + 1].reshape(-1, 2).min(0)) - 1500
hi = np.maximum(np.percentile(pts, 99, axis=0), U[:K + 1].reshape(-1, 2).max(0)) + 1500
span = (hi - lo).max()
c = (lo + hi) / 2
ext = (km(c[0] - span / 2), km(c[0] + span / 2), km(c[1] - span / 2), km(c[1] + span / 2))
n_u, n_g = U.shape[1], X.shape[1]
peak_of = np.arange(n_u) % n_g
theta = np.deg2rad(30.0)
sign = np.where((np.arange(n_u) // n_g) % 2 == 0, -1.0, 1.0)
ALPHA, VMAX = 4.0, 20.0


def rot(d, a):
    return np.array([np.cos(a) * d[0] - np.sin(a) * d[1], np.sin(a) * d[0] + np.cos(a) * d[1]])


fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.98), gridspec_kw=dict(wspace=0.05))
Xs, Ys, val = smooth_field(km(gx), km(gx), PV[K].reshape(side, side), k=4, order=3)
val = np.clip(val / PV[K].max(), 0, 1)
spd = float(np.median(np.hypot(*np.moveaxis(z["flow"][K], -1, 0))))
titles = ["Belief and tracks", "Value field $V$", "NMS peaks", "Rotated anchors"]
for p, ax in enumerate(axs):
    ocean_local(ax, z["flow"][K], z["flow_axis"], vmax=0.3, cmap=OCEAN_SOFT, density=1.0, lw=0.45,
                stream=(p == 0))
    if p >= 1:
        ax.contourf(Xs, Ys, val, levels=[0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.01], cmap=VALUE, vmin=0, vmax=1,
                    alpha=0.93, zorder=2)
    if p == 0:
        for g in np.flatnonzero(alive):
            ax.scatter(km(PT[K, g, :, 0]), km(PT[K, g, :, 1]), s=0.8, color=["#8C510A", "#BF6F1F", "#D9A15B"][g],
                       alpha=0.45, lw=0, zorder=3, rasterized=True)
            ax.plot(*km(X[K, g]), "X", ms=4.6, color="#1A1A1A", mec="white", mew=0.5, zorder=8)
        for i in range(n_u):
            ax.plot(km(U[:K + 1, i, 0]), km(U[:K + 1, i, 1]), color=TRACK[peak_of[i]], lw=0.8,
                    ls="-" if sign[i] < 0 else (0, (2.5, 1.2)), zorder=4)
        ax.text(0.5, 0.05, "× hidden survivor", transform=ax.transAxes, ha="center", fontsize=6.0,
                path_effects=HALO, zorder=10)
    if p >= 2:
        for j in np.flatnonzero(pk_ok):
            ax.plot(*km(PK[K, j]), "*", ms=8, color=TRACK[j], mec="white", mew=0.6, zorder=8)
            if p == 2:
                ax.add_patch(Circle(km(PK[K, j]), 2.0, fill=False, ec=TRACK[j], lw=0.7, ls=(0, (2, 1.5)), zorder=7))
                ax.text(*km(PK[K, j] + 800), str(j + 1), fontsize=6.5, color=TRACK[j], fontweight="bold",
                        path_effects=HALO, zorder=9)
    if p == 3:
        for i in range(n_u):
            col = TRACK[peak_of[i]]
            if pk_ok[peak_of[i]]:
                ax.plot(km([U[K, i, 0], PK[K, peak_of[i], 0]]), km([U[K, i, 1], PK[K, peak_of[i], 1]]),
                        color=col, lw=0.55, ls=(0, (1, 1.2)), alpha=0.85, zorder=4)
            ax.annotate("", km(U[K, i] + 2400 * rot(AD[K, i], sign[i] * theta)), km(U[K, i]), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7))
    for i in range(n_u):
        if p in (0, 3):
            ax.plot(*km(U[K, i]), "o", ms=3.3 if p == 3 else 2.8, color=TRACK[peak_of[i]], mec="white", mew=0.5,
                    zorder=7)
        elif p == 1:
            ax.plot(*km(U[K, i]), "o", ms=2.8, color="#33415C", mec="white", mew=0.4, zorder=7)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    frame(ax)
    head(ax, "abcd"[p], titles[p], size=8.0)
scalebar(axs[0], ext[1] - 0.6, ext[3] - 1.4, km=2)
axs[1].text(0.5, 0.05, r"$V(\mathbf{x})=p_d\,e^{-t/\tau}\,\hat p_t(\mathbf{x})$", transform=axs[1].transAxes,
            ha="center", fontsize=6.2, path_effects=HALO, zorder=10)
axs[2].text(0.5, 0.05, "suppression radius 2 km", transform=axs[2].transAxes, ha="center",
            fontsize=6.0, path_effects=HALO, zorder=10)
axs[3].text(0.5, 0.05, r"peak $=i$ mod $G$, rotation $\pm30^\circ$", transform=axs[3].transAxes,
            ha="center", fontsize=6.0, path_effects=HALO, zorder=10)
note(f"pipeline (1x4): step {K} t={t[K] / 60:.1f} min, alive survivors {int(alive.sum())}, median current {spd:.3f}")
save(fig, "figR2_pipeline")

# ---- heading residual of every executed action of the episode (data inset for Fig. 2c) ------------
devs, mags = [], []
for k in range(len(t)):
    for i in range(n_u):
        d = AD[k, i]
        if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-9:
            continue
        za = rot(d, sign[i] * theta)
        a = A[k, i] / VMAX
        cross, dot = za[0] * a[1] - za[1] * a[0], za @ a
        devs.append(-sign[i] * np.degrees(np.arctan2(cross, dot)))   # + leans towards the assigned peak
        mags.append(np.linalg.norm(a))
devs, mags = np.array(devs), np.array(mags)
fig = plt.figure(figsize=(1.9, 1.45))
ax = fig.add_subplot(111, projection="polar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(-60)
ax.set_thetamax(60)
bins = np.arange(-60, 61, 4)
h, _ = np.histogram(np.clip(devs, -59.9, 59.9), bins)
h = h / h.sum()
ax.bar(np.deg2rad(bins[:-1] + 2), h, width=np.deg2rad(4) * 0.92, color=OURS, alpha=0.75, lw=0, zorder=3)
rmax = h.max() * 1.18
ax.plot([0, 0], [0, rmax], color=OURS, lw=1.4, zorder=4)
ax.plot([np.deg2rad(30)] * 2, [0, rmax], color=TRACK[0], lw=0.9, ls=(0, (2, 1.2)), zorder=4)
ax.text(0, rmax * 1.05, "anchor", ha="center", va="bottom", fontsize=6.0, color=OURS, fontweight="bold")
ax.text(np.deg2rad(36), rmax * 1.02, "peak", ha="left", va="bottom", fontsize=6.0, color=TRACK[0])
ax.set_ylim(0, rmax)
ax.set_yticks([])
ax.set_xticks(np.deg2rad([-60, -30, 0, 30, 60]))
ax.set_xticklabels(["−60°", "−30°", "", "+30°", "+60°"], fontsize=5.6)
ax.tick_params(pad=0.5)
ax.grid(color="#E1E5EA", lw=0.5)
ax.spines["polar"].set_color("#9AA5B1")
ax.spines["polar"].set_linewidth(0.6)
q90 = np.percentile(np.abs(devs), 90)
note(f"residual inset: n={len(devs)}, |dev| median {np.median(np.abs(devs)):.2f}, p90 {q90:.2f} deg, "
     f"towards-peak share {np.mean(devs > 0):.3f}, |a| median {np.median(mags):.3f} vmax")
save(fig, "figR2_residual")
