import numpy as np

rng = np.random.default_rng()  # random generator (add a seed for reproducible results)

N = 23 # number of values per method

mu = -0.4
sigma = 0.5

res = rng.normal(mu, sigma, N)  # true mean reward of each arm

epsilon = 0.1
k = N  # number of arms must match the number of true means
steps = 1000

RealReward, IdealReward = 0.0, 0.0

Qa = np.zeros(k)             # estimated value per arm
Na = np.zeros(k, dtype=int)  # separate array: how often each arm was pulled

best_arm = np.argmax(res)    # arm with the highest true mean
optimal_picks = 0

# history per step (needed for the plots)
reward_history = []   # reward actually received
ideal_history = []    # reward the best arm would have given
optimal_history = []  # True if the best arm was chosen

def action_selection():
    x = rng.random()  # uniform in [0, 1) -> P(x < epsilon) = epsilon
    if x < epsilon:
        return rng.integers(k)  # explore: random arm 0..k-1
    else:
        return np.argmax(Qa)    # exploit: best estimated arm

for i in range(steps):
    rewards = rng.normal(res, sigma)  # one noisy reward per arm for this step
    A = action_selection()
    R = rewards[A]
    Na[A] +=1
    Qa[A] = Qa[A] + 1/Na[A] * (R - Qa[A])
    RealReward += R
    IdealReward += rewards[best_arm]  # what always playing the best arm would have earned
    optimal_picks += (A == best_arm)
    reward_history.append(R)
    ideal_history.append(rewards[best_arm])
    optimal_history.append(A == best_arm)


# ---------------------------------------------------------------------------
# Pretty output of the results
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Epsilon comparison
# ---------------------------------------------------------------------------
# A single run is very noisy, so every epsilon is simulated N_RUNS times on the
# SAME arms (res) and the results are averaged. Each run starts with fresh
# estimates Qa = 0 and Na = 0.
EPSILONS = [0.0, 0.01, 0.1, 0.3]  # exploration rates to compare
N_RUNS = 200                     # independent runs per epsilon

def optimal_share_curve(eps, n_runs=N_RUNS):
    """Return the % of runs that chose the best arm, for every step."""
    run_rng = np.random.default_rng(0)  # same seed for every epsilon -> fair comparison
    optimal = np.zeros(steps)
    for _ in range(n_runs):
        Q = np.zeros(k)             # estimated value per arm (fresh per run)
        Nr = np.zeros(k, dtype=int) # pull counter per arm (fresh per run)
        for i in range(steps):
            if run_rng.random() < eps:
                A = run_rng.integers(k)   # explore: random arm 0..k-1
            else:
                A = np.argmax(Q)          # exploit: best estimated arm
            R = run_rng.normal(res[A], sigma)  # only the chosen arm's reward is needed
            Nr[A] += 1
            Q[A] = Q[A] + 1/Nr[A] * (R - Q[A])
            optimal[i] += (A == best_arm)
    return optimal / n_runs * 100


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# colors (light theme)
BLUE, ORANGE = "#2a78d6", "#eb6834"
EPS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # one fixed color per epsilon
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

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
reward_history = np.array(reward_history)
ideal_history = np.array(ideal_history)
optimal_history = np.array(optimal_history)

window = 50  # moving-average window to smooth the noisy rewards
def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
fig.suptitle(f"{k}-armed bandit, epsilon = {epsilon}, {steps} steps",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)

# 1) average reward over time
ax = axes[0, 0]
ax.plot(t[window - 1:], moving_average(reward_history, window), color=BLUE, lw=2,
        label=f"received reward ({window}-step avg)")
ax.axhline(res[best_arm], color=INK_2, lw=1.2, ls="--",
           label=f"best arm mean = {res[best_arm]:.3f}")
ax.set_title("Reward per step")
ax.set_xlabel("step")
ax.set_ylabel("reward")
ax.margins(y=0.25)  # headroom so the legend does not cover the line
ax.legend(loc="upper center", ncol=2)

# 2) share of optimal actions over time, one line per epsilon
ax = axes[0, 1]
for eps, color in zip(EPSILONS, EPS_COLORS):
    curve = optimal_share_curve(eps)
    smooth = 20  # light smoothing, the per-step mean over runs is still noisy
    ax.plot(t[smooth - 1:], moving_average(curve, smooth), color=color, lw=2,
            label=f"epsilon = {eps}")
ax.set_ylim(0, 100)
ax.set_title(f"Best arm chosen (mean of {N_RUNS} runs)")
ax.set_xlabel("step")
ax.set_ylabel("% of runs")
ax.legend(loc="lower right")

# 3) cumulative regret over time
ax = axes[1, 0]
cum_regret = np.cumsum(ideal_history - reward_history)
ax.plot(t, cum_regret, color=BLUE, lw=2)
ax.annotate(f"{cum_regret[-1]:.1f}", (t[-1], cum_regret[-1]), xytext=(-4, 6),
            textcoords="offset points", ha="right", color=INK, fontweight="bold")
ax.set_title("Cumulative regret (lower is better)")
ax.set_xlabel("step")
ax.set_ylabel("ideal reward - real reward")

# 4) true mean vs. estimate per arm, sorted by true mean
ax = axes[1, 1]
order = np.argsort(res)[::-1]
x = np.arange(k)
pulled = Na[order] > 0
ax.vlines(x[pulled], res[order][pulled], Qa[order][pulled], color=GRID, lw=2, zorder=1)
ax.scatter(x, res[order], s=60, facecolor="#fcfcfb", edgecolor=BLUE, lw=2,
           zorder=2, label="true mean")
ax.scatter(x[pulled], Qa[order][pulled], s=60, color=ORANGE, edgecolor="#fcfcfb",
           lw=1.5, zorder=3, label="estimate Q")
for xi, a in zip(x, order):  # pull count above each arm
    ax.annotate(str(Na[a]), (xi, max(res[a], Qa[a])), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8,
                color=INK if a == best_arm else MUTED,
                fontweight="bold" if a == best_arm else "normal")
ax.set_xticks(x, [str(a) for a in order], fontsize=8)
ax.set_title("Arms: true mean vs. estimate (numbers = pulls)")
ax.set_xlabel("arm (sorted by true mean)")
ax.set_ylabel("reward")
ax.margins(y=0.15)
ax.grid(axis="x", visible=False)  # 23 vertical gridlines would be noise
ax.legend(loc="lower left")

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("bandit_plots.png", dpi=130)
plt.show()