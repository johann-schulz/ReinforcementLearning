"""
Demonstration of the three standard random number generators in NumPy:

    1. Normal / Gaussian   -> Generator.normal(loc, scale, size)
    2. Continuous uniform  -> Generator.uniform(low, high, size)
    3. Discrete uniform    -> Generator.integers(low, high, size, endpoint=False)

The script uses the modern Generator API (np.random.default_rng), which is the
recommended way since NumPy 1.17. The legacy module-level functions
(np.random.normal, np.random.uniform, np.random.randint) still work, but they
share one hidden global state and use the older Mersenne Twister algorithm.

Run:  python random_demo.py
Output: printed statistics + a figure "random_demo.png"
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render without a display (writes PNG files)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 0. Create a generator
# ---------------------------------------------------------------------------
# A fixed seed makes every run reproducible: same seed -> same numbers.
# Use default_rng() without a seed for different numbers on every run.
SEED = 42
rng = np.random.default_rng(SEED)

N = 100_000  # number of samples per experiment


def describe(name, samples, expected_mean, expected_std):
    """Print empirical vs. theoretical statistics of a sample array."""
    print(f"  {name}")
    print(f"    dtype            : {samples.dtype}")
    print(f"    min / max        : {samples.min():.4f} / {samples.max():.4f}")
    print(f"    mean (empirical) : {samples.mean():.4f}   (theory: {expected_mean:.4f})")
    print(f"    std  (empirical) : {samples.std():.4f}   (theory: {expected_std:.4f})")


# ---------------------------------------------------------------------------
# 1. NORMAL / GAUSSIAN DISTRIBUTION
# ---------------------------------------------------------------------------
# rng.normal(loc=0.0, scale=1.0, size=None)
#   loc   : mean (mu)              -> center of the bell curve
#   scale : standard deviation (sigma), must be >= 0 -> width of the curve
#   size  : output shape (int or tuple), None returns a single float
#
# Properties:
#   - continuous values (float64)
#   - UNBOUNDED: any real number is possible, extreme values are just rare
#   - ~68 % within mu +- 1 sigma, ~95 % within +- 2 sigma, ~99.7 % within +- 3 sigma
print("=" * 70)
print("1) NORMAL DISTRIBUTION  rng.normal(loc, scale, size)")
print("=" * 70)

single_value = rng.normal()  # one sample from the standard normal N(0, 1)
print(f"  single value, standard normal  : {single_value:.4f}")

matrix = rng.normal(loc=10.0, scale=2.0, size=(2, 3))  # size can be a shape tuple
print(f"  2x3 matrix, N(10, 2):\n{np.round(matrix, 3)}")

mu, sigma = 170.0, 8.0  # e.g. body height in cm
normal_samples = rng.normal(loc=mu, scale=sigma, size=N)
describe(f"N(mu={mu}, sigma={sigma}), n={N}", normal_samples, mu, sigma)

# Check the 68-95-99.7 rule empirically
for k in (1, 2, 3):
    share = np.mean(np.abs(normal_samples - mu) <= k * sigma)
    print(f"    share within +-{k} sigma : {share * 100:6.2f} %")

# scale=0 is allowed and degenerates to a constant
print(f"  scale=0 -> {rng.normal(loc=5.0, scale=0.0, size=3)}")

# loc/scale can be arrays: element-wise parameters ("broadcasting")
print(f"  array params  -> {np.round(rng.normal(loc=[0, 100, 1000], scale=[1, 1, 1]), 3)}")


# ---------------------------------------------------------------------------
# 2. CONTINUOUS UNIFORM DISTRIBUTION
# ---------------------------------------------------------------------------
# rng.uniform(low=0.0, high=1.0, size=None)
#   low  : lower bound, INCLUDED
#   high : upper bound, EXCLUDED  -> half-open interval [low, high)
#   size : output shape
#
# Properties:
#   - continuous values (float64)
#   - BOUNDED: never leaves [low, high)
#   - every sub-interval of equal length has the same probability (flat histogram)
#   - mean = (low + high) / 2,  std = (high - low) / sqrt(12)
print()
print("=" * 70)
print("2) CONTINUOUS UNIFORM  rng.uniform(low, high, size)")
print("=" * 70)

print(f"  default [0, 1)        : {np.round(rng.uniform(size=5), 4)}")
print(f"  [-1, 1)               : {np.round(rng.uniform(-1.0, 1.0, size=5), 4)}")

low, high = 2.0, 8.0
uniform_samples = rng.uniform(low=low, high=high, size=N)
describe(
    f"U(low={low}, high={high}), n={N}",
    uniform_samples,
    expected_mean=(low + high) / 2,
    expected_std=(high - low) / np.sqrt(12),
)

# Floats are effectively never equal to a specific value -> duplicates are rare
print(f"    unique values    : {np.unique(uniform_samples).size} of {N}")

# Equivalent alternative: rng.random() gives [0, 1); scale and shift it yourself
manual = low + (high - low) * rng.random(5)
print(f"  low + (high-low)*rng.random() : {np.round(manual, 4)}")


# ---------------------------------------------------------------------------
# 3. DISCRETE UNIFORM DISTRIBUTION (integers)
# ---------------------------------------------------------------------------
# rng.integers(low, high=None, size=None, dtype=np.int64, endpoint=False)
#   low      : smallest possible integer (included)
#   high     : upper bound, EXCLUDED by default
#              if high is None, the range is [0, low)
#   endpoint : True -> high is INCLUDED, i.e. [low, high]
#   dtype    : integer type of the result (int64 default, e.g. np.uint8 possible)
#
# Properties:
#   - discrete values (integers), each with probability 1 / (number of values)
#   - mean = (a + b) / 2,  std = sqrt((n^2 - 1) / 12) with a,b = smallest/largest
#     possible value and n = b - a + 1 (number of possible values)
print()
print("=" * 70)
print("3) DISCRETE UNIFORM  rng.integers(low, high, size, endpoint)")
print("=" * 70)

# Classic pitfall: a die is NOT integers(1, 6) - that only gives 1..5!
print(f"  integers(1, 6)                : {rng.integers(1, 6, size=15)}  <- no 6!")
print(f"  integers(1, 7)                : {rng.integers(1, 7, size=15)}")
print(f"  integers(1, 6, endpoint=True) : {rng.integers(1, 6, size=15, endpoint=True)}")
print(f"  integers(10)  -> [0, 10)      : {rng.integers(10, size=15)}")
print(f"  dtype=np.uint8                : {rng.integers(0, 256, size=5, dtype=np.uint8).dtype}")

die = rng.integers(1, 6, size=N, endpoint=True)  # fair die: 1..6
a, b = 1, 6
n_values = b - a + 1
describe(
    f"die 1..6, n={N}",
    die,
    expected_mean=(a + b) / 2,
    expected_std=np.sqrt((n_values**2 - 1) / 12),
)
values, counts = np.unique(die, return_counts=True)
for v, c in zip(values, counts):
    print(f"    P({v}) = {c / N:.4f}   (theory: {1 / n_values:.4f})")


# ---------------------------------------------------------------------------
# 4. PITFALL: integers from floats via rounding are NOT uniform
# ---------------------------------------------------------------------------
# A common mistake: np.round(rng.uniform(1, 6)) to get die values.
# The edge values 1 and 6 only get half an interval each -> half the probability.
print()
print("=" * 70)
print("4) PITFALL: rounding uniform floats to get integers")
print("=" * 70)
rounded = np.round(rng.uniform(1, 6, size=N)).astype(int)
values, counts = np.unique(rounded, return_counts=True)
for v, c in zip(values, counts):
    print(f"    P({v}) = {c / N:.4f}")
print("    -> 1 and 6 are only half as frequent as 2..5. Use rng.integers instead.")


# ---------------------------------------------------------------------------
# 5. REPRODUCIBILITY
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("5) REPRODUCIBILITY with seeds")
print("=" * 70)
rng_a = np.random.default_rng(123)
rng_b = np.random.default_rng(123)
print(f"  same seed equal?      : {np.array_equal(rng_a.normal(size=5), rng_b.normal(size=5))}")
rng_c = np.random.default_rng(124)
print(f"  different seed equal? : {np.array_equal(rng_a.normal(size=5), rng_c.normal(size=5))}")


# ---------------------------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 6a: normal distribution with theoretical density
ax = axes[0, 0]
ax.hist(normal_samples, bins=100, density=True, color="#4C72B0", alpha=0.7)
x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 400)
pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
ax.plot(x, pdf, color="black", lw=1.5, label="theoretical density")
for k in (1, 2, 3):
    ax.axvspan(mu - k * sigma, mu + k * sigma, color="#4C72B0", alpha=0.06)
ax.set_title(f"normal(loc={mu}, scale={sigma})")
ax.legend()

# 6b: effect of the scale parameter
ax = axes[0, 1]
for s in (0.5, 1.0, 2.0):
    ax.hist(rng.normal(0, s, N), bins=150, density=True, histtype="step", lw=1.5,
            label=f"scale={s}")
ax.set_xlim(-7, 7)
ax.set_title("normal: effect of scale (loc=0)")
ax.legend()

# 6c: continuous uniform
ax = axes[1, 0]
ax.hist(uniform_samples, bins=60, density=True, color="#55A868", alpha=0.7)
ax.axhline(1 / (high - low), color="black", lw=1.5, label="theoretical density 1/(high-low)")
ax.set_xlim(low - 1, high + 1)
ax.set_title(f"uniform(low={low}, high={high})")
ax.legend(loc="lower center")

# 6d: discrete uniform vs. rounded floats
ax = axes[1, 1]
width = 0.38
vals = np.arange(1, 7)
die_share = [np.mean(die == v) for v in vals]
round_share = [np.mean(rounded == v) for v in vals]
ax.bar(vals - width / 2, die_share, width, color="#C44E52", label="integers(1, 6, endpoint=True)")
ax.bar(vals + width / 2, round_share, width, color="#8C8C8C", label="round(uniform(1, 6))")
ax.axhline(1 / 6, color="black", lw=1, ls="--", label="1/6")
ax.set_xticks(vals)
ax.set_title("discrete uniform vs. rounded floats")
ax.set_ylim(0, 0.28)  # leave room for the legend above the bars
ax.legend(fontsize=8, loc="upper center", ncol=2)

fig.tight_layout()
fig.savefig("random_demo.png", dpi=120)
print("\nFigure saved as random_demo.png")