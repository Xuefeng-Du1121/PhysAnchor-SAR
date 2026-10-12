"""Shared visual system for all revision-3 figures.

One palette, one font, one set of map conventions, so every figure reads the same:
  * ocean panels: deep-to-light blue speed map + thin white streamlines, land in warm grey
  * belief / value: warm orange-red (kept from v2, the user liked it)
  * our method: one navy family; baselines: a muted categorical family by method type
  * UAV tracks: three peak-colours from a cool harmonious set (teal / indigo / violet),
    the two UAVs of a pair share a colour (solid / dashed)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "axes.edgecolor": "#3A3A3A", "axes.labelcolor": "#1A1A1A", "xtick.color": "#3A3A3A",
    "ytick.color": "#3A3A3A", "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.4,
    "ytick.major.size": 2.4, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#E9E9E9", "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.1, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlepad": 3.5, "hatch.linewidth": 0.5,
})

# ---- method-type palette (used identically in every chart) -----------------------------------
OURS = "#1B4F8C"        # physics written into the policy (ours)
OURS_M = "#5B8BC7"      # our variants
OURS_L = "#B8CEE8"
PLAN = "#3E8E5E"        # belief-driven planners
PLAN_L = "#9FCDB0"
PATTERN = "#B8892B"     # IAMSAR / geometric patterns
PATTERN_L = "#E3C98F"
SWARM = "#7A5C99"       # bio-inspired / optimisation (PSO, ACO, Levy)
SWARM_L = "#C3B1D6"
MARL = "#7C8794"        # learned, no physics
MARL_IN = "#D9822B"     # learned, physics as input
PEVA = "#B5414B"        # physics governs a generative learner
ORACLE = "#2A2A2A"
REF = "#9AA5B1"
BG = "#F4F6F8"
TYPE_COLOR = {"ours": OURS, "plan": PLAN, "pattern": PATTERN, "swarm": SWARM, "marl": MARL,
              "marl_in": MARL_IN, "peva": PEVA, "oracle": ORACLE}
TYPE_LABEL = {"ours": "Physics written into the policy (ours)", "plan": "Belief-driven planner",
              "pattern": "IAMSAR / geometric pattern", "swarm": "Swarm / stochastic search",
              "marl": "MARL, no physics", "marl_in": "MARL, physics as input",
              "peva": "MARL, physics governs learning", "oracle": "Oracle (true positions)"}

# ---- maps ---------------------------------------------------------------------------------------
OCEAN = LinearSegmentedColormap.from_list("ocean", ["#F2F7FB", "#D3E5F1", "#9EC6E0", "#5E9CC8", "#2B6FA8",
                                                    "#124A7E", "#08294D"])
OCEAN_SOFT = LinearSegmentedColormap.from_list("ocean_soft", ["#F7FAFC", "#E2EEF6", "#C4DCEC", "#9DC3DE",
                                                              "#6FA3CC"])
VALUE = LinearSegmentedColormap.from_list("value", [(1, 1, 1, 0), "#FDE7C8", "#F8B97D", "#EE8748",
                                                    "#CF532D", "#9C2A24"])
LAND = "#E6E2D9"
LAND_EDGE = "#9C9585"
TRACK = ["#127A86", "#3F51B5", "#8E44AD"]      # teal / indigo / violet, one per value peak
TRACK_L = ["#7CC4CB", "#9AA6E0", "#C9A4DB"]
FOUND = "#1E8449"
MISSED = "#C0392B"
SPLITC = {"train": "#4D4D4D", "validation": "#127A86", "test": "#C0392B", "ood": "#8E44AD"}

HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
HALO_S = [pe.withStroke(linewidth=1.2, foreground="white")]


# ---- one name / type / line style per method, shared by every results figure ----------------------
METHOD = {  # key: (label, type, linestyle)
    "anchor30": ("Physics anchor + residual (ours)", "ours", "-"),
    "phys30": ("Physics anchor only (ours)", "ours", (0, (3, 1.2))),
    "anchor15": (r"Anchor + residual, $\theta=15^{\circ}$", "ours", (0, (1, 1))),
    "phys15": (r"Anchor only, $\theta=15^{\circ}$", "ours", (0, (1, 1))),
    "anchor-2upd": ("Anchor, two updates", "ours", (0, (1, 1))),
    "sector-search": ("Sector search", "pattern", "-"),
    "expanding-square": ("Expanding square", "pattern", (0, (3, 1.2))),
    "lawnmower-coverage": ("Parallel track", "pattern", (0, (1, 1))),
    "spiral-search": ("Spiral", "pattern", (0, (5, 1.5, 1, 1.5))),
    "aco-search": ("Ant colony", "swarm", "-"),
    "pso-search": ("Particle swarm", "swarm", (0, (3, 1.2))),
    "voronoi-coverage": ("Voronoi coverage", "swarm", (0, (1, 1))),
    "levy-random-walk": ("Lévy walk", "swarm", (0, (5, 1.5, 1, 1.5))),
    "pvf-greedy": ("PVF-greedy", "plan", "-"),
    "cbba": ("CBBA", "plan", (0, (3, 1.2))),
    "bayesian-information-gain": ("Bayesian info. gain", "plan", (0, (1, 1))),
    "greedy-probability": ("Greedy probability", "plan", (0, (1, 1))),
    "mpc-horizon": ("Receding-horizon MPC", "plan", (0, (5, 1.5, 1, 1.5))),
    "gradient-ascent": ("Value-gradient ascent", "plan", (0, (1, 2))),
    "mappo": ("MAPPO", "marl", (0, (3, 1.2))),
    "mappo-long": (r"MAPPO, 5$\times$ budget", "marl", (0, (1, 1))),
    "ippo": ("IPPO", "marl", (0, (1, 1))),
    "commnet": ("CommNet-MAPPO", "marl", (0, (1, 1))),
    "happo": ("HAPPO", "marl", (0, (1, 1))),
    "ic3net": ("IC3Net", "marl", "-"),
    "pvf-mappo": ("PVF + MAPPO", "marl_in", "-"),
    "pvf-commnet": ("PVF + CommNet", "marl_in", (0, (3, 1.2))),
    "pvf-happo": ("PVF + HAPPO", "marl_in", (0, (1, 1))),
    "pvf-ic3net": ("PVF + IC3Net", "marl_in", (0, (1, 1))),
    "shaping": ("Potential shaping only", "shape", "-"),
    "bc-finetune": ("Behaviour cloning + PPO", "shape", (0, (3, 1.2))),
    "bc-init": ("Behaviour cloning only", "shape", (0, (1, 1))),
    "peva-gen": ("PEVA-Gen", "peva", "-"),
    "oracle-target": ("Oracle (true positions)", "oracle", (0, (1, 1))),
}
SHAPE = "#8C6D46"       # physics in the reward / initialization
TYPE_COLOR["shape"] = SHAPE
TYPE_LABEL["shape"] = "MARL, physics in reward / init"


def mcolor(k):
    return TYPE_COLOR[METHOD[k][1]]


def head(ax, letter, text="", x=0.0, y=1.02, size=8.5):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontweight="bold", ha="left", va="bottom")
    if text:
        ax.annotate(text, (x, y), xycoords="axes fraction", xytext=(10, 0), textcoords="offset points",
                    fontsize=7.5, ha="left", va="bottom")


def frame(ax, color="#9A9A9A", lw=0.5):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)
    ax.set_xticks([])
    ax.set_yticks([])
