"""Shared data access and drawing helpers for the revision-3 figures (server side, v5 venv).

Every plotted number is read from summary.json / episodes.jsonl / recorded .npz files;
nothing is typed in. Each figure appends what it plotted to FIGURE_DATA_r3.md.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from style_r3 import *  # noqa: F401,F403  (palette + rcParams)
from style_r3 import OCEAN, OCEAN_SOFT, LAND, LAND_EDGE

PR = _paths.root("peva-phys-residual")
V5 = _paths.root("peva-gen-final-v5")
VIZ = PR / "runs/complete-paper-v1/viz-data"
E3 = PR / "runs/complete-paper-v3/eval"
E4 = PR / "runs/complete-paper-v4/eval"
OUT = PR / "runs/complete-paper-v4/figures-r3"
OUT.mkdir(parents=True, exist_ok=True)
R_M, EARTH, TAU_S, DT, T = 40000.0, 6371008.8, 9000.0, 30.0, 10800.0
LOG = []


def note(s):
    LOG.append(s)
    print(s, flush=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    note(f"wrote {name}")
    with open(OUT / "FIGURE_DATA_r3.md", "a") as f:
        f.write("\n".join(LOG) + "\n")
    LOG.clear()


def box_corners(origin):
    """Corners (lon, lat) of a 40x40 km local frame centred on origin (same maths as real_env.LocalFrame)."""
    sc = EARTH * np.pi / 180 * np.array([np.cos(np.deg2rad(origin[1])), 1.0])
    xy = np.array([[0, 0], [R_M, 0], [R_M, R_M], [0, R_M], [0, 0]], float)
    return (xy - R_M / 2) / sc + np.asarray(origin, float)


def ocean_local(ax, uv, axis_m, vmax=None, cmap=OCEAN_SOFT, density=0.9, lw=0.45, stream=True, z=0):
    """Local current panel in km: speed colour + thin white streamlines. uv[row=y, col=x, 2]."""
    km = np.asarray(axis_m) / 1000.0
    u, v = uv[..., 0], uv[..., 1]
    s = np.hypot(u, v)
    vmax = vmax or max(float(np.nanmax(s)), 1e-6)
    m = ax.pcolormesh(km, km, s, cmap=cmap, vmin=0, vmax=vmax, shading="gouraud", rasterized=True, zorder=z)
    if stream:
        ax.streamplot(km, km, u, v, density=density, color="white", linewidth=lw, arrowsize=0.45,
                      zorder=z + 0.5)
    return m


def _rast(cs):
    try:
        cs.set_rasterized(True)
    except Exception:  # matplotlib < 3.8
        for c in getattr(cs, "collections", []):
            c.set_rasterized(True)


def fill_nearest(a):
    """Replace non-finite cells by their nearest finite neighbour (only so contours meet the coast)."""
    from scipy import ndimage
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def smooth_field(x, y, f, k=4, order=3):
    """Upsample a gridded field k-fold (display only) so filled contours are not blocky."""
    from scipy import ndimage
    return (np.linspace(x[0], x[-1], len(x) * k), np.linspace(y[0], y[-1], len(y) * k),
            ndimage.zoom(f, k, order=order))


def ocean_region(ax, lon, lat, u, v, vmax=1.2, density=1.4):
    """Regional current map: smooth speed colour, warm-grey land, white streamlines.
    Land cells are filled by nearest ocean value before contouring and then covered by the land polygon."""
    s = np.hypot(u, v)
    land = ~np.isfinite(s)
    X, Y, sf = smooth_field(lon, lat, fill_nearest(s), k=4, order=1)
    cs = ax.contourf(X, Y, np.clip(sf, 0, vmax), levels=np.linspace(0, vmax, 31), cmap=OCEAN, zorder=1)
    _rast(cs)
    X, Y, lf = smooth_field(lon, lat, land.astype(float), k=4, order=1)
    cl = ax.contourf(X, Y, lf, levels=[0.5, 2], colors=[LAND], zorder=2)
    _rast(cl)
    ax.contour(X, Y, lf, levels=[0.5], colors=LAND_EDGE, linewidths=0.45, zorder=3)
    m = cs
    uu = np.ma.masked_where(land, np.nan_to_num(u))
    vv = np.ma.masked_where(land, np.nan_to_num(v))
    ax.streamplot(lon, lat, uu, vv, density=density, color="white", linewidth=0.4, arrowsize=0.4, zorder=4)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    return m


def lonlat_ticks(ax, xt, yt):
    """Degree ticks; call after set_xlim/set_ylim (ticks outside the view are dropped)."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xt = [x for x in xt if x0 <= x <= x1]
    yt = [y for y in yt if y0 <= y <= y1]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{x:g}°E" for x in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:g}°N" for y in yt])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)


def scalebar(ax, x1, y0, km=5, txt=None):
    from style_r3 import HALO
    ax.plot([x1 - km, x1], [y0, y0], color="#222222", lw=1.6, solid_capstyle="butt", zorder=20)
    ax.text(x1 - km / 2, y0 + 0.35, txt or f"{km:g} km", ha="center", va="bottom", fontsize=6,
            zorder=20, path_effects=HALO)


# ---------------------------------------------------------------- results
def load_summary(path):
    s = json.load(open(path))
    g = lambda k: (s[k]["mean"] if isinstance(s.get(k), dict) else s.get(k))  # noqa: E731
    out = dict(succ=g("success_rate"), ttd=g("ttd_all"), surv=g("survival_score"), n_ep=s.get("episodes"))
    if "complete_rescue" in s:
        out["all"] = s["complete_rescue"]
    if "per_scenario" in s:
        out["per_scenario"] = s["per_scenario"]
    return out


def ev(name, root=E4):
    p = Path(root) / name / "summary.json"
    return load_summary(p) if p.exists() else None


def ev_seeds(prefix, root=E4, seeds=range(10)):
    return [r for r in (ev(f"{prefix}-seed{s}", root) for s in seeds) if r]


def episodes(name, root=E4):
    p = Path(root) / name / "episodes.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


def det_matrix(name, root=E4):
    """episodes x survivors detection times (inf = missed) of an eval_misspec run."""
    return np.array([[np.inf if d is None else d for d in r["detected_at"]] for r in episodes(name, root)], float)


def S(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return dict(mean=float(v.mean()) if len(v) else np.nan, sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                n=len(v), vals=v)
