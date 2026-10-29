import numpy as np
from numpy.ma.core import argmax

rng = np.random.default_rng()  # random generator (add a seed for reproducible results)

# N = 1000  # number of values per method
#
# normal_values = rng.normal(loc=0.0, scale=1.0, size=N)     # Gaussian, mean 0, std 1
# uniform_values = rng.uniform(low=0.0, high=1.0, size=N)    # floats in [0, 1)
# integer_values = rng.integers(low=1, high=7, size=N)       # integers 1..6 (high excluded)
#
# print(f"normal   mean: {normal_values.mean():.4f}   (expected 0.0)")
# print(f"uniform  mean: {uniform_values.mean():.4f}   (expected 0.5)")
# print(f"integers mean: {integer_values.mean():.4f}   (expected 3.5)")




N = 23 # number of values per method

µ = -0.4
sigma = 0.5



# print(res)
res = [-0.37915695, 0.55927098, -0.34890002, -0.69698177, -0.79115813, -0.29669629, -0.12108567, -0.56936486, -0.07730853, -0.4915665, -0.80586033, -1.27869053, 0.22021786, -0.77748074, -0.3728305, -0.28867085, 0.0701784, -0.37297967, 0.16054218, -0.28494065, -0.48076307, 0.46328484, 0.03096098]
slot_averages = []

# for mu in res:
#     tmp = rng.normal(mu, sigma, 1000)
#     avg = np.mean(tmp)
#     slot_averages.append(avg)
#
#     print(f"µ={mu:12.8f}, avg={avg:12.8f}")



epsilon = 0.1
k = 23
steps = 1000

RealReward, IdealReward = 0, 0

Qa = [0 for _ in range(k)]
Na = [0 for _ in range(k)]

best_arm = np.argmax(res)    # arm with the highest true mean
optimal_picks = 0

def action_selection():
    x = rng.normal(0, 1)
    if x < epsilon:
        return rng.integers(k)
    else:
        return np.argmax(Qa)

for i in range(steps):
    rewards = []
    for mu in res:
        rewards.append(rng.normal(mu, sigma))
    A = action_selection()
    R = rewards[A]
    Na[A] +=1
    Qa[A] = Qa[A] + 1/Na[A] * (R - Qa[A])
    RealReward += R
    IdealReward += max(rewards)
    optimal_picks += (A == best_arm)

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
print(f"real percentage  : {(RealReward/IdealReward)*100:9.2f}")
print(f"ideal reward     : {IdealReward:9.2f}  (always best arm)")
print(f"regret           : {regret:9.2f}")
print(f"optimal arm used : {optimal_picks / steps * 100:8.1f} % of steps")


