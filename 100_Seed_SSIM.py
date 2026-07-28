import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

num_seeds = 100

print(f"Starting Baseline Simulation ({num_seeds} seeds)...")

all_op_A, all_op_B = [], []
all_id_A, all_id_B = [], []
all_se_A, all_se_B = [], []

for i in range(num_seeds):
    if (i + 1) % 10 == 0:
        print(f"Running seed {i + 1}/{num_seeds}...")
        
    random_seed = np.random.randint(0, 1000000)
    run_data = run_simulation(seed=random_seed) 

    all_op_A.append(run_data['op_A'])
    all_op_B.append(run_data['op_B'])
    all_id_A.append(run_data['id_A'])
    all_id_B.append(run_data['id_B'])
    all_se_A.append(run_data['se_A'])
    all_se_B.append(run_data['se_B'])

print("Simulations complete! Calculating macro-level statistics...")

mat_op_A, mat_op_B = np.array(all_op_A), np.array(all_op_B)
mat_id_A, mat_id_B = np.array(all_id_A), np.array(all_id_B)
mat_se_A, mat_se_B = np.array(all_se_A), np.array(all_se_B)

num_recorded_points = mat_op_A.shape[1]
time_steps = np.linspace(0, num_steps, num_recorded_points)

df_summary = pd.DataFrame({
    'time_step': time_steps,
    'mean_op_A': np.mean(mat_op_A, axis=0), 'sd_op_A': np.std(mat_op_A, axis=0),
    'mean_op_B': np.mean(mat_op_B, axis=0), 'sd_op_B': np.std(mat_op_B, axis=0),
    'mean_id_A': np.mean(mat_id_A, axis=0), 'sd_id_A': np.std(mat_id_A, axis=0),
    'mean_id_B': np.mean(mat_id_B, axis=0), 'sd_id_B': np.std(mat_id_B, axis=0),
    'mean_se_A': np.mean(mat_se_A, axis=0), 'sd_se_A': np.std(mat_se_A, axis=0),
    'mean_se_B': np.mean(mat_se_B, axis=0), 'sd_se_B': np.std(mat_se_B, axis=0)
})

script_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(script_dir, "exp_baseline_100_summary.csv")
df_summary.to_csv(summary_path, index=False, sep=";", decimal=",")
print(f"Data saved to {summary_path}")

plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

x_vals = df_summary['time_step']

ax1.plot(x_vals, df_summary['mean_op_A'], color='navy', linewidth=3, label='Group 1 (Activists)')
ax1.fill_between(x_vals, df_summary['mean_op_A'] - df_summary['sd_op_A'], df_summary['mean_op_A'] + df_summary['sd_op_A'], color='navy', alpha=0.15)
ax1.plot(x_vals, df_summary['mean_op_B'], color='darkred', linewidth=3, label='Group 2 (Mainstream)')
ax1.fill_between(x_vals, df_summary['mean_op_B'] - df_summary['sd_op_B'], df_summary['mean_op_B'] + df_summary['sd_op_B'], color='darkred', alpha=0.15)
ax1.set_title("Mean Opinion over Time")
ax1.set_ylabel("Variable Value (0 to 1)")

ax2.plot(x_vals, df_summary['mean_id_A'], color='navy', linewidth=3)
ax2.fill_between(x_vals, df_summary['mean_id_A'] - df_summary['sd_id_A'], df_summary['mean_id_A'] + df_summary['sd_id_A'], color='navy', alpha=0.15)
ax2.plot(x_vals, df_summary['mean_id_B'], color='darkred', linewidth=3)
ax2.fill_between(x_vals, df_summary['mean_id_B'] - df_summary['sd_id_B'], df_summary['mean_id_B'] + df_summary['sd_id_B'], color='darkred', alpha=0.15)
ax2.set_title("Mean Identity Standard over Time")

ax3.plot(x_vals, df_summary['mean_se_A'], color='navy', linewidth=3)
ax3.fill_between(x_vals, df_summary['mean_se_A'] - df_summary['sd_se_A'], df_summary['mean_se_A'] + df_summary['sd_se_A'], color='navy', alpha=0.15)
ax3.plot(x_vals, df_summary['mean_se_B'], color='darkred', linewidth=3)
ax3.fill_between(x_vals, df_summary['mean_se_B'] - df_summary['sd_se_B'], df_summary['mean_se_B'] + df_summary['sd_se_B'], color='darkred', alpha=0.15)
ax3.set_title("Mean Self-Esteem over Time")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time Steps")
    # Optional: adjust x-axis formatting if numbers get too large
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.grid(True, linestyle='--', alpha=0.5)

ax1.legend()
fig.suptitle("Base Simulation Dynamics (100 Seeds Averaged)", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
