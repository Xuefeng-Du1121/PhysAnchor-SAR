"""Held-out evaluation in which the public belief is built on a WRONG number of survivors.

eval_misspec.py initializes the belief with the true number G and only lets the anchor extract K != G peaks.
Here the operator's assumed count Kb enters the belief itself: the particle filter carries Kb survivor
components while the ocean contains G survivors.  The anchor (and every planner) only sees this belief.

* --belief-count Kb (required): number of belief components; pass --anchor-peaks Kb as well.
* On a confirmed detection the belief component that best explains the detected position (largest
  particle-weighted Gaussian likelihood, 1-km scale) is removed; if Kb < G and every component has been
  removed, later detections remove nothing.  With Kb == G the identity association of the benchmark is used
  and the belief is not redrawn, so Kb == G reproduces eval_misspec exactly (checked by the guard).
* Coverage evidence updates every remaining component exactly as in env.MaritimeSAR.step.
All other arguments are those of eval_misspec.py.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_misspec as E                                      # noqa: E402
from peva_sim.ocean_forced_env import OceanForcedSAR         # noqa: E402
from peva_sim.env import reflect, segment_distance           # noqa: E402

KB = None


class WrongBeliefSAR(OceanForcedSAR):
    def reset(self, seed=0):
        self.bfound = None
        obs, info = super().reset(seed)
        c = self.cfg
        if KB != c.n_targets:                                 # redraw the belief with Kb components
            self.particles = reflect(self.datum + self.belief_rng.normal(0, c.prior_sigma_m, (KB, c.particles, 2)),
                                     c.region_m)
            self.weights = np.full((KB, c.particles), 1 / c.particles)
        self.bfound = np.zeros(len(self.particles), dtype=bool)
        return self.observation(), self.info()

    def pvf(self):
        if self.bfound is None or getattr(self, "_skip", False):
            return super().pvf() if self.bfound is None else np.zeros(len(self.grid))
        c = self.cfg
        h = max(c.sensing_m, c.prior_sigma_m * c.particles ** (-1 / 6))
        value = np.zeros(len(self.grid))
        for b in range(len(self.particles)):
            if self.bfound[b]:
                continue
            for j in range(0, len(self.grid), 32):
                d = (self.grid[j:j + 32, None] - self.particles[b, None]) / h
                value[j:j + 32] += np.exp(-.5 * np.sum(d * d, -1)) @ self.weights[b] / (2 * np.pi * h * h)
        return value * c.detection_p * np.exp(-self.t / c.survival_tau_s)

    def step(self, actions, messages=None):
        c = self.cfg
        start, before = self.uavs.copy(), self.found.copy()
        W = self.weights
        self.weights = np.zeros((c.n_targets, c.particles))    # parent loop indexes by true survivor
        self.coverage_update_enabled, self._skip = False, True
        _, reward, term, trunc, info = super().step(actions, messages)
        self.weights, self._skip = W, False
        for j in np.flatnonzero(self.found & ~before):        # remove the component that explains the detection
            if len(self.particles) == c.n_targets:
                b = j
            else:
                free = np.flatnonzero(~self.bfound)
                if not len(free):
                    continue
                d2 = np.sum((self.particles[free] - self.targets[j]) ** 2, -1)
                b = free[int(np.argmax(np.sum(self.weights[free] * np.exp(-d2 / (2 * 1000.0 ** 2)), 1)))]
            self.bfound[b] = True
        for b in range(len(self.particles)):
            if self.bfound[b]:
                self.weights[b].fill(0)
                continue
            d = segment_distance(self.particles[b], start, self.uavs)
            miss = np.prod(1 - c.detection_p * np.maximum(0, 1 - (d / c.sensing_m) ** 2), axis=1)
            w = self.weights[b] * miss
            total = w.sum()
            if c.coverage_mode == "intensity":
                self.weights[b] = w
            else:
                self.weights[b] = w / total if total > 1e-300 else np.full(c.particles, 1 / c.particles)
        return self.observation(), reward, term, trunc, info


if __name__ == "__main__":
    i = sys.argv.index("--belief-count")
    KB = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    E.OceanForcedSAR = WrongBeliefSAR
    E.main()
