import numpy as np

rng = np.random.default_rng()  # random generator (add a seed for reproducible results)
initial = np.random.default_rng(187)

N = 23 # number of values per method

mu = -0.4
sigma = 0.5

res = initial.normal(mu, sigma, N)  # true mean reward of each arm

epsilons = [0.0, 0.01, 0.1, 0.3]  # all epsilons to compare
k = N  # number of arms must match the number of true means
steps = 100000

best_arm = np.argmax(res)    # arm with the highest true mean (same for all runs)


def action_selection(Qa, epsilon):
    x = rng.random()  # uniform in [0, 1) -> P(x < epsilon) = epsilon
    if x < epsilon:
        return rng.integers(k)  # explore: random arm 0..k-1
    else:
        return np.argmax(Qa)    # exploit: best estimated arm


def run_bandit(epsilon):
    """Run one complete bandit simulation for a single epsilon and return its results."""
    RealReward, IdealReward = 0.0, 0.0

    Qa = np.zeros(k)             # estimated value per arm
    Na = np.zeros(k, dtype=int)  # separate array: how often each arm was pulled

    optimal_picks = 0

    # history per step (needed for the plots)
    reward_history = []   # reward actually received
    ideal_history = []    # reward the best arm would have given
    optimal_history = []  # True if the best arm was chosen

    for i in range(steps):
        rewards = rng.normal(res, sigma)  # one noisy reward per arm for this step
        A = action_selection(Qa, epsilon)
        R = rewards[A]
        Na[A] +=1
        Qa[A] = Qa[A] + 1/Na[A] * (R - Qa[A])
        RealReward += R
        IdealReward += rewards[best_arm]  # what always playing the best arm would have earned
        optimal_picks += (A == best_arm)
        reward_history.append(R)
        ideal_history.append(rewards[best_arm])
        optimal_history.append(A == best_arm)

    return {
        "epsilon": epsilon,
        "Qa": Qa,
        "Na": Na,
        "RealReward": RealReward,
        "IdealReward": IdealReward,
        "optimal_picks": optimal_picks,
        "reward_history": np.array(reward_history),
        "ideal_history": np.array(ideal_history),
        "optimal_history": np.array(optimal_history),
    }


# run one simulation per epsilon
results = [run_bandit(eps) for eps in epsilons]


# ---------------------------------------------------------------------------
# Pretty output of the results
# ---------------------------------------------------------------------------
for r in results:
    epsilon, Qa, Na = r["epsilon"], r["Qa"], r["Na"]
    RealReward, IdealReward = r["RealReward"], r["IdealReward"]
    optimal_picks = r["optimal_picks"]

    print(f"\n{k}-armed bandit | epsilon={epsilon} | steps={steps}\n")
    print(f"{'arm':>4} {'true mean':>10} {'estimate Q':>11} {'pulls':>6}  share")
    print("-" * 60)
    for a in np.argsort(res)[::-1]:  # sorted by true mean, best arm first
        share = Na[a] / steps
        bar = "#" * int(round(share * 30))
        marker = "  <- best" if a == best_arm else ""
        est = f"{Qa[a]:11.3f}" if Na[a] > 0 else f"{'-':>11}"
        print(f"{a:>4} {res[a]:10.3f} {est} {Na[a]:6d}  {bar}{marker}")
    print("-" * 60)

    regret = IdealReward - RealReward
    print(f"real reward      : {RealReward:9.2f}")
    print(f"ideal reward     : {IdealReward:9.2f}  (always best arm)")
    print(f"regret           : {regret:9.2f}")
    print(f"efficiency       : {RealReward / IdealReward * 100:8.1f} %" if IdealReward > 0 else "")
    print(f"optimal arm used : {optimal_picks / steps * 100:8.1f} % of steps")

# compact comparison of all epsilons
print(f"\n{'epsilon':>8} {'regret':>10} {'optimal arm %':>14}")
print("-" * 34)
for r in results:
    regret = r["IdealReward"] - r["RealReward"]
    print(f"{r['epsilon']:8.3f} {regret:10.2f} {r['optimal_picks'] / steps * 100:14.1f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# colors (light theme)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# one color per epsilon (extend the list if you compare more epsilons)
LINE_COLORS = [BLUE, ORANGE, "#2f9e6e", "#8e5bd0", "#d4a017", "#c2417a", "#3fa7b5"]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "font.size": 10,
})

t = np.arange(1, steps + 1)

window = 50  # moving-average window to smooth the noisy rewards
def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle(f"{k}-armed bandit, {len(epsilons)} epsilons, {steps} steps",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)


# 2) share of optimal actions over time (one curve per epsilon)
for r, color in zip(results, LINE_COLORS):
    epsilon = r["epsilon"]
    optimal_history = r["optimal_history"]
    ax.plot(t, np.cumsum(optimal_history) / t * 100, color=color, lw=2,
            label=f"epsilon = {epsilon}")
    max_share = (1 - epsilon + epsilon / k) * 100  # upper limit with fixed epsilon
    ax.axhline(max_share, color=color, lw=1, ls="--", alpha=0.7)
ax.set_ylim(0, 100)
ax.set_title("Best arm chosen (dashed: maximum possible per epsilon)")
ax.set_xlabel("step")
ax.set_ylabel("% of steps")
ax.legend(loc="lower right")

fig.tight_layout()
plt.show()