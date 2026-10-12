"""Study area: regional currents with every scenario box, local currents in representative boxes,
and the current-speed regime of all 40 scenarios (12 training, 2 validation, 2 test, 24 OOD)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from r3_common import VIZ, ocean_region, ocean_local, lonlat_ticks, box_corners, note, save
from style_r3 import SPLITC, HALO, head, frame, OCEAN

sa = np.load(VIZ / "study_area.npz")
bc = np.load(VIZ / "box_currents.npz")
ids, splits, origins = [str(x) for x in bc["ids"]], [str(x) for x in bc["splits"]], bc["origins"]
spl = lambda s: "ood" if s == "ood_test" else s  # noqa: E731
fig = plt.figure(figsize=(7.0, 4.25))
outer = gridspec.GridSpec(2, 2, width_ratios=[1.38, 1], height_ratios=[2.35, 0.72], hspace=0.36, wspace=0.07)

# (a) regional map ---------------------------------------------------------------------------------
ax = fig.add_subplot(outer[0, 0])
lon, lat = sa["lon"], sa["lat"]
m = ocean_region(ax, lon, lat, sa["u_test"], sa["v_test"], vmax=1.2, density=1.6)
seen = set()
for sid, sp, o in zip(ids, splits, origins):
    s = spl(sp)
    cc = box_corners(o)
    ax.plot(cc[:, 0], cc[:, 1], color=SPLITC[s], lw=1.1 if s != "ood" else 0.85, zorder=8)
    if s == "ood" and "jan15" in sid:
        ax.text(cc[:, 0].max() + 0.18, cc[:, 1].mean(), "A" + sid[-1], fontsize=6.3, color=SPLITC["ood"],
                fontweight="bold", va="center", path_effects=HALO, zorder=9)
    if s not in seen and s != "ood":
        seen.add(s)
        lab = {"train": "Training\n1–6 Jan", "validation": "Validation\n12–13 Jan", "test": "Test\n22–23 Jan"}[s]
        off = {"train": (-30, 16), "validation": (-44, -40), "test": (6, -40)}[s]
        ax.annotate(lab, (cc[:, 0].mean(), cc[:, 1].mean()), xytext=off, textcoords="offset points", fontsize=6.4,
                    color=SPLITC[s], fontweight="bold", path_effects=HALO, zorder=9, ha="center",
                    arrowprops=dict(arrowstyle="-", color=SPLITC[s], lw=0.55, shrinkA=0, shrinkB=3))
ax.text(0.98, 0.03, "OOD areas A0–A5\n8, 15, 19, 27 Jan", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.3, color=SPLITC["ood"], fontweight="bold", path_effects=HALO, zorder=9)
ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
lonlat_ticks(ax, [120, 125, 130, 135], [20, 24, 28, 32])
from datetime import datetime, timezone  # noqa: E402
day = datetime.fromtimestamp(int(sa["test_start_utc"]), tz=timezone.utc).strftime("%-d %b %Y")
head(ax, "a", f"Surface currents ({day}) and the 40 scenario boxes")

# (b) local currents in representative boxes ------------------------------------------------------
pick = [("train-jan01-area0", "Train 1 Jan"), ("train-jan03-area1", "Train 3 Jan"), ("train-jan06-area0", "Train 6 Jan"),
        ("validation-jan12", "Val 12 Jan"), ("test-jan22", "Test 22 Jan"), ("test-jan23", "Test 23 Jan"),
        ("ood-jan15-area0", "OOD A0"), ("ood-jan15-area2", "OOD A2"), ("ood-jan15-area5", "OOD A5")]
inner = outer[0, 1].subgridspec(3, 3, wspace=0.05, hspace=0.05)
med = {}
for sid in ids:
    sp = np.concatenate([np.hypot(*np.moveaxis(bc[sid + k], -1, 0)).ravel() for k in ("__t0", "__t3")])
    med[sid] = (float(np.median(sp)), float(np.percentile(sp, 90)))
for q, (sid, lab) in enumerate(pick):
    ax = fig.add_subplot(inner[q // 3, q % 3])
    ocean_local(ax, bc[sid + "__t0"], bc["axis_m"], vmax=1.2, cmap=OCEAN, density=0.7, lw=0.35)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect("equal")
    col = SPLITC[spl(splits[ids.index(sid)])]
    frame(ax, color=col, lw=1.1)
    ax.text(0.05, 0.94, lab, transform=ax.transAxes, fontsize=6.2, color=col, fontweight="bold", va="top",
            path_effects=HALO)
    ax.text(0.95, 0.05, f"{med[sid][0]:.2f} m/s", transform=ax.transAxes, ha="right", fontsize=6.0,
            path_effects=HALO)
    if q == 0:
        head(ax, "b", "40×40 km boxes, same scale as (a)")
cb = fig.colorbar(m, ax=fig.axes[1:], orientation="horizontal", fraction=0.035, pad=0.03, aspect=30,
                  ticks=[0, 0.3, 0.6, 0.9, 1.2])
cb.set_label("Current speed (m s$^{-1}$)", fontsize=6.6, labelpad=1)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)

# (c) speed regime of all 40 scenarios ----------------------------------------------------------
# smaller canvas: the panel starts at the left edge of map (a), so its "Speed" label lines up with the
# latitude labels of (a), and it is lower than before; (a) and (b) are untouched
ax = fig.add_subplot(outer[1, :])
fig.canvas.draw()
pa = fig.axes[0].get_position()             # drawn box of map (a) (equal aspect)
pb = fig.axes[9].get_position()             # last small panel of (b): right edge of the (b) grid
pc = ax.get_position()
ax.set_position([pa.x0, pc.y0, pb.x1 - pa.x0, pc.height * 0.8])
order = sorted(ids, key=lambda s: med[s][0])
for i, sid in enumerate(order):
    s = spl(splits[ids.index(sid)])
    ax.bar(i, med[sid][0], 0.7, color=SPLITC[s], alpha=0.88 if s != "ood" else 0.62, lw=0, zorder=2)
    ax.plot([i, i], [med[sid][0], med[sid][1]], color=SPLITC[s], lw=0.6, zorder=2)
ax.set_xticks([])
ax.set_xlim(-0.8, len(order) - 0.2)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Speed (m s$^{-1}$)", fontsize=6.8)
ax.set_xlabel("40 scenarios, sorted by median box current", fontsize=6.8, labelpad=2)
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=SPLITC[k], alpha=a, label=v) for k, v, a in
                   (("train", "Training (12)", .88), ("validation", "Validation (2)", .88), ("test", "Test (2)", .88),
                    ("ood", "OOD (24)", .62))], ncol=4, loc="upper left", fontsize=6.3)
head(ax, "c", "Median (bar) and 90th-percentile (line) current over the 3-h mission in every box")
for sp in ("train", "validation", "test", "ood_test"):
    v = [med[s][0] for s, x in zip(ids, splits) if x == sp]
    note(f"study {sp:10s} n={len(v)} median box speed {np.median(v):.3f} m/s (range {min(v):.3f}-{max(v):.3f})")
save(fig, "figR3_study_area")
