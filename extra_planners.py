"""Additional non-learning SAR baselines for the common evaluator (revision round 3).

All planners use only the information available to the learned actors: their
own positions, elapsed time and the public value field ``value_density`` on the
shared ``prior_grid_m`` (survival-, detection- and coverage-aware posterior).
None uses survivor positions. Every planner replans at every 30-s step and
returns constant-speed velocity commands; the simulator geofence applies.

Patterns follow the IAMSAR Manual (expanding square, sector search); the
optimisation and coordination baselines follow the cited references in the
manuscript (PSO, Voronoi/Lloyd coverage, receding-horizon MPC, ant-colony
pheromone search, random-walk and probability-gradient ascent).
"""
import numpy as np


def _unit(delta, speed):
    delta = np.asarray(delta, dtype=np.float64)
    n = np.linalg.norm(delta, axis=-1, keepdims=True)
    return (delta / np.maximum(n, 1e-12) * speed).astype(np.float64)


def _datum(obs):
    """Value-weighted centroid of the public field (the 'datum' a SAR coordinator would plot)."""
    pts = np.asarray(obs["prior_grid_m"], np.float64)
    v = np.asarray(obs["value_density"], np.float64)
    if v.sum() <= 0:
        return pts.mean(0)
    return (pts * v[:, None]).sum(0) / v.sum()


class _Waypoints:
    """Shared machinery: each UAV follows its own waypoint list, cycling."""

    def _follow(self, obs, cfg):
        pos = np.asarray(obs["positions_m"], np.float64)
        thr = cfg.speed_mps * cfg.dt_s * 1.5
        tgt = []
        for i, p in enumerate(pos):
            path = self.paths[i]
            k = int(self.idx[i])
            while np.linalg.norm(path[k] - p) <= thr:
                k = (k + 1) % len(path)
                if k == self.idx[i]:
                    break
            self.idx[i] = k
            tgt.append(path[k])
        return _unit(np.asarray(tgt) - pos, cfg.speed_mps)


class ExpandingSquare(_Waypoints):
    """IAMSAR expanding-square search centred on the datum; UAVs start at staggered legs."""
    name = "expanding-square"

    def reset(self, obs):
        self.paths, self.idx = None, None

    def act(self, obs, cfg, state=None):
        if self.paths is None:
            c = _datum(obs)
            s = 2.0 * cfg.sensing_m * 0.9  # track spacing slightly below sweep width
            n = len(obs["positions_m"])
            dirs = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], float)
            self.paths = []
            for i in range(n):
                pts, p, leg = [], c + s * i * np.array([0.5, 0.5]) / n, 1
                rot = i % 4
                for k in range(160):
                    d = dirs[(k + rot) % 4]
                    p = p + d * s * leg
                    pts.append(np.clip(p, 0, cfg.region_m))
                    if k % 2 == 1:
                        leg += 1
                self.paths.append(np.asarray(pts))
            self.idx = np.zeros(n, int)
        return self._follow(obs, cfg)


class SectorSearch(_Waypoints):
    """IAMSAR sector search: radial legs through the datum, sectors split among UAVs."""
    name = "sector-search"

    def reset(self, obs):
        self.paths, self.idx = None, None

    def act(self, obs, cfg, state=None):
        if self.paths is None:
            c = _datum(obs)
            n = len(obs["positions_m"])
            R = 3.0 * cfg.prior_sigma_m
            self.paths = []
            for i in range(n):
                pts = []
                for k in range(12):
                    th = 2 * np.pi * (i / n + k / (12 * n)) + (k % 2) * np.pi
                    pts += [c, np.clip(c + R * np.array([np.cos(th), np.sin(th)]), 0, cfg.region_m)]
                self.paths.append(np.asarray(pts))
            self.idx = np.zeros(n, int)
        return self._follow(obs, cfg)


class SpiralSearch(_Waypoints):
    """Archimedean spirals around the datum with phase offsets per UAV."""
    name = "spiral-search"

    def reset(self, obs):
        self.paths, self.idx = None, None

    def act(self, obs, cfg, state=None):
        if self.paths is None:
            c = _datum(obs)
            n = len(obs["positions_m"])
            b = 2.0 * cfg.sensing_m * n / (2 * np.pi)
            self.paths = []
            for i in range(n):
                th = np.linspace(0, 30 * np.pi, 600) + 2 * np.pi * i / n
                r = b * (th - 2 * np.pi * i / n) + 2.0 * cfg.sensing_m * i
                pts = c + np.stack([r * np.cos(th), r * np.sin(th)], -1)
                self.paths.append(np.clip(pts, 0, cfg.region_m))
            self.idx = np.zeros(n, int)
        return self._follow(obs, cfg)


class GradientAscent:
    """Probability-gradient ascent on the smoothed public field (no coordination)."""
    name = "gradient-ascent"

    def reset(self, obs):
        return None

    def act(self, obs, cfg, state=None):
        pts = np.asarray(obs["prior_grid_m"], np.float64)
        v = np.asarray(obs["value_density"], np.float64)
        pos = np.asarray(obs["positions_m"], np.float64)
        h = 2.0 * cfg.region_m / cfg.grid_side
        d = pts[None] - pos[:, None]
        w = v[None] * np.exp(-np.sum(d ** 2, -1) / (2 * h ** 2))
        g = (w[..., None] * d).sum(1)
        far = np.linalg.norm(g, axis=1) < 1e-30
        if far.any():  # flat neighbourhood: head for the global maximum
            g[far] = pts[int(np.argmax(v))] - pos[far]
        return _unit(g, cfg.speed_mps)


class VoronoiCoverage:
    """Lloyd/Voronoi coverage: each UAV moves to the value-weighted centroid of its cell."""
    name = "voronoi-coverage"

    def reset(self, obs):
        return None

    def act(self, obs, cfg, state=None):
        pts = np.asarray(obs["prior_grid_m"], np.float64)
        v = np.asarray(obs["value_density"], np.float64) + 1e-300
        pos = np.asarray(obs["positions_m"], np.float64)
        owner = np.argmin(np.linalg.norm(pts[:, None] - pos[None], axis=-1), axis=1)
        tgt = pos.copy()
        for i in range(len(pos)):
            m = owner == i
            if m.any() and v[m].sum() > 0:
                tgt[i] = (pts[m] * v[m, None]).sum(0) / v[m].sum()
        return _unit(tgt - pos, cfg.speed_mps)


class PSOSearch:
    """Particle-swarm search: UAVs are particles, fitness = public field at their position."""
    name = "pso-search"

    def __init__(self, w=0.6, c1=1.2, c2=1.6, seed=0):
        self.w, self.c1, self.c2, self.rng = w, c1, c2, np.random.default_rng(seed)

    def reset(self, obs):
        pos = np.asarray(obs["positions_m"], np.float64)
        self.vel = np.zeros_like(pos)
        self.pbest = pos.copy()
        self.pval = np.full(len(pos), -np.inf)
        self.rng = np.random.default_rng(12345)

    def _f(self, obs, x):
        pts = np.asarray(obs["prior_grid_m"], np.float64)
        v = np.asarray(obs["value_density"], np.float64)
        return v[np.argmin(np.linalg.norm(pts[None] - x[:, None], axis=-1), axis=1)]

    def act(self, obs, cfg, state=None):
        pos = np.asarray(obs["positions_m"], np.float64)
        f = self._f(obs, pos)
        self.pval = self._f(obs, self.pbest)  # field is non-stationary: re-score memories
        better = f > self.pval
        self.pbest[better], self.pval[better] = pos[better], f[better]
        pts = np.asarray(obs["prior_grid_m"], np.float64)
        gbest = pts[int(np.argmax(np.asarray(obs["value_density"])))]
        r1, r2 = self.rng.random((2, len(pos), 1))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pbest - pos) + self.c2 * r2 * (gbest - pos)
        self.vel += self.rng.normal(0, cfg.sensing_m, self.vel.shape)  # keep exploring
        return _unit(self.vel, cfg.speed_mps)


class ACOSearch:
    """Ant-colony search with repulsive pheromone on visited cells (stigmergic coordination)."""
    name = "aco-search"

    def reset(self, obs):
        self.tau = None
        self.rng = np.random.default_rng(4242)

    def act(self, obs, cfg, state=None):
        pts = np.asarray(obs["prior_grid_m"], np.float64)
        v = np.asarray(obs["value_density"], np.float64)
        pos = np.asarray(obs["positions_m"], np.float64)
        if self.tau is None:
            self.tau = np.zeros(len(pts))
        self.tau *= 0.97
        near = np.argmin(np.linalg.norm(pts[None] - pos[:, None], axis=-1), axis=1)
        self.tau[near] += 1.0
        vn = v / max(v.max(), 1e-300)
        tgt = []
        for p in pos:
            dist = np.linalg.norm(pts - p, axis=1)
            score = (vn + 1e-6) ** 1.0 * np.exp(-self.tau) / (1 + dist / 5000.0) ** 2
            prob = score / score.sum()
            tgt.append(pts[self.rng.choice(len(pts), p=prob)])
        return _unit(np.asarray(tgt) - pos, cfg.speed_mps)


class RecedingHorizonMPC:
    """Receding-horizon planner: each UAV picks the heading that maximises the value
    collected along an H-step straight-line rollout, sequentially with coverage
    discount for earlier UAVs' rollouts (a decoupled MPC baseline)."""
    name = "mpc-horizon"

    def __init__(self, horizon_steps=10, headings=16):
        self.H, self.K = horizon_steps, headings

    def reset(self, obs):
        return None

    def act(self, obs, cfg, state=None):
        pts = np.asarray(obs["prior_grid_m"], np.float64)
        v = np.asarray(obs["value_density"], np.float64).copy()
        pos = np.asarray(obs["positions_m"], np.float64)
        ang = np.linspace(0, 2 * np.pi, self.K, endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
        step = cfg.speed_mps * cfg.dt_s
        r2 = (1.5 * cfg.sensing_m) ** 2
        out = []
        for p in pos:
            best, bestd, bestmask = -1.0, dirs[0], None
            for d in dirs:
                traj = np.clip(p + np.arange(1, self.H + 1)[:, None] * step * d, 0, cfg.region_m)
                mask = (np.sum((pts[None] - traj[:, None]) ** 2, -1) <= r2 * 4).any(0)
                gain = v[mask].sum()
                if gain > best:
                    best, bestd, bestmask = gain, d, mask
            if best <= 0:  # nothing in reach: go to the global maximum
                bestd = pts[int(np.argmax(v))] - p
            elif bestmask is not None:
                v[bestmask] *= 0.2
            out.append(bestd)
        return _unit(np.asarray(out), cfg.speed_mps)


class RandomWalk:
    """Lévy-flight random search, a lower bound that ignores the belief."""
    name = "levy-random-walk"

    def reset(self, obs):
        self.rng = np.random.default_rng(777)
        self.goal = None

    def act(self, obs, cfg, state=None):
        pos = np.asarray(obs["positions_m"], np.float64)
        if self.goal is None:
            self.goal = pos.copy()
        for i, p in enumerate(pos):
            if np.linalg.norm(self.goal[i] - p) < cfg.speed_mps * cfg.dt_s:
                th = self.rng.uniform(0, 2 * np.pi)
                L = min(cfg.region_m / 2, 500.0 * self.rng.pareto(1.5) + 500.0)
                self.goal[i] = np.clip(p + L * np.array([np.cos(th), np.sin(th)]), 0, cfg.region_m)
        return _unit(self.goal - pos, cfg.speed_mps)


EXTRA = {c.name: c for c in (ExpandingSquare, SectorSearch, SpiralSearch, GradientAscent, VoronoiCoverage,
                             PSOSearch, ACOSearch, RecedingHorizonMPC, RandomWalk)}
