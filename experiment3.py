import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, required=True, help='Number of arms')
parser.add_argument('--var', type=str, choices=['low', 'medium', 'high'], required=True, help='Variance level')
args = parser.parse_args()

# Constants
T = 1000000
N_EXPERIMENTS = 50
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter settings
K = args.K
variance_levels = {
    "low": (1, 5),
    "medium": (5, 20),
    "high": (20, 50)
}
vmin, vmax = variance_levels[args.var]

# UCB Algorithms
def known_variance_ucb(K, means, variances, best_arm):
    counts = np.zeros(K)
    emp_means = np.zeros(K)
    regret = np.zeros(T + 1)
    for t in range(1, T + 1):
        ucb_values = [np.inf if counts[a] == 0 else emp_means[a] + np.sqrt((4 * variances[a] * np.log(T)) / counts[a]) for a in range(K)]
        arm = np.argmax(ucb_values)
        reward = np.random.normal(means[arm], np.sqrt(variances[arm]))
        counts[arm] += 1
        emp_means[arm] += (reward - emp_means[arm]) / counts[arm]
        regret[t] = regret[t - 1] + (means[best_arm] - means[arm])
    return regret

def unknown_variance_ucb(K, means, variances, best_arm):
    counts = np.zeros(K)
    emp_means = np.zeros(K)
    emp_vars = np.ones(K)
    regret = np.zeros(T + 1)
    M2 = np.zeros(K)
    for t in range(1, T + 1):
        ucb_values = []
        for a in range(K):
            if counts[a] == 0:
                ucb_values.append(np.inf)
            else:
                n = counts[a]
                correction = max(1 - 2 * np.sqrt((2 * np.log(T)) / n), 1e-6)
                ucb = emp_means[a] + np.sqrt(emp_vars[a]) * np.sqrt((4 * np.log(T)) / (n * correction))
                ucb_values.append(ucb)
        arm = np.argmax(ucb_values)
        reward = np.random.normal(means[arm], np.sqrt(variances[arm]))
        counts[arm] += 1
        n = counts[arm]
        delta = reward - emp_means[arm]
        emp_means[arm] += delta / n
        M2[arm] += delta * (reward - emp_means[arm])
        if n > 1:
            emp_vars[arm] = M2[arm] / (n - 1)
        regret[t] = regret[t - 1] + (means[best_arm] - means[arm])
    return regret

def standard_ucb(K, means, variances, best_arm):
    counts = np.zeros(K)
    emp_means = np.zeros(K)
    regret = np.zeros(T + 1)
    for t in range(1, T + 1):
        ucb_values = [np.inf if counts[a] == 0 else emp_means[a] + np.sqrt((4 * np.log(T)) / counts[a]) for a in range(K)]
        arm = np.argmax(ucb_values)
        reward = np.random.normal(means[arm], np.sqrt(variances[arm]))
        counts[arm] += 1
        emp_means[arm] += (reward - emp_means[arm]) / counts[arm]
        regret[t] = regret[t - 1] + (means[best_arm] - means[arm])
    return regret


# Generate problem instance
print(f"\n--- Generating instance: K={K}, Variance={args.var} ({vmin}-{vmax}) ---")

means = [np.random.uniform(0, 1) for _ in range(K)]
variances = [np.random.uniform(vmin, vmax) for _ in range(K)]
best_arm = np.argmax(means)

print(f"Generated means and variances (Best arm: {best_arm}, mean: {means[best_arm]:.4f})")

# Save table
df = pd.DataFrame({
    "Arm Index": list(range(K)),
    "Mean": means,
    "Variance": variances
})
table_path = f"{OUTPUT_DIR}/table_K{K}_{args.var}.csv"
df.to_csv(table_path, index=False)
print(f"Saved arm table to: {table_path}")

# Run experiments
print("Running experiments...")
known_regrets = np.empty((N_EXPERIMENTS, T + 1))
unknown_regrets = np.empty((N_EXPERIMENTS, T + 1))
standard_regrets = np.empty((N_EXPERIMENTS, T + 1))
for i in range(N_EXPERIMENTS):
    known_regrets[i] = known_variance_ucb(K, means, variances, best_arm)
    unknown_regrets[i] = unknown_variance_ucb(K, means, variances, best_arm)
    standard_regrets[i] = standard_ucb(K, means, variances, best_arm)
print("Finished experiments")

# Averages and std dev
known_avg = np.mean(known_regrets, axis=0)
known_std = np.std(known_regrets, axis=0)
unknown_avg = np.mean(unknown_regrets, axis=0)
unknown_std = np.std(unknown_regrets, axis=0)
standard_avg = np.mean(standard_regrets, axis=0)
standard_std = np.std(standard_regrets, axis=0)

# Plot
plt.figure(figsize=(10, 5))
x = np.arange(T + 1)

# CUD colorblind-friendly palette
# Blue: #0072B2, Vermillion: #D55E00, Green: #2ca02c

plt.plot(x, known_avg, label="VarUCB-Known", color="#0072B2", linestyle=':', linewidth=2)
plt.plot(x, unknown_avg, label="VarUCB-Unknown", color="#2ca02c", linestyle='--', linewidth=2)
plt.plot(x, standard_avg, label="UCB", color="#D55E00", linestyle='-', linewidth=2)

plt.fill_between(x, known_avg - known_std, known_avg + known_std, color="#0072B2", alpha=0.2)
plt.fill_between(x, unknown_avg - unknown_std, unknown_avg + unknown_std, color="#2ca02c", alpha=0.2)
# No fill for standard UCB (too wide for clarity)

plt.xlabel("Rounds", fontsize=12)
plt.ylabel("Cumulative Regret", fontsize=12)
plt.title(f"Cumulative Regret Comparison (UCB, VarUCB-Known, VarUCB-Unknown), K = {K}, {args.var.capitalize()} Variance", fontsize=14)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend()
plt.tight_layout()

plot_path = f"{OUTPUT_DIR}/plot_K{K}_{args.var}.png"
plt.savefig(plot_path)
plt.close()
print(f"Saved plot to: {plot_path}")
