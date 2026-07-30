import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

num_seeds = 100

drain_rates = [0.10, 0.15, 0.20]
fixed_charge_rate = 0.05 
summary_stats = []      

print(f"Starting Negativity Bias Sweep (100 seeds)...")

for drain_val in drain_rates:
    print(f"Running Condition: delta_drain = {drain_val:.2f}")
    
    fin_op_A, fin_op_B = [], []
    fin_id_A, fin_id_B = [], []
    fin_se_A, fin_se_B = [], []
    
    for i in range(num_seeds):
        random_seed = np.random.randint(0, 1000000)
        run_data = run_simulation(seed=random_seed, delta_charge=fixed_charge_rate, delta_drain=drain_val) 
        
        fin_op_A.append(run_data['op_A'][-1])
        fin_op_B.append(run_data['op_B'][-1])
        fin_id_A.append(run_data['id_A'][-1])
        fin_id_B.append(run_data['id_B'][-1])
        fin_se_A.append(run_data['se_A'][-1])
        fin_se_B.append(run_data['se_B'][-1])

    summary_stats.append({
        'delta_drain': drain_val,
        'mean_op_A': np.mean(fin_op_A), 'sd_op_A': np.std(fin_op_A),
        'mean_op_B': np.mean(fin_op_B), 'sd_op_B': np.std(fin_op_B),
        'mean_id_A': np.mean(fin_id_A), 'sd_id_A': np.std(fin_id_A),
        'mean_id_B': np.mean(fin_id_B), 'sd_id_B': np.std(fin_id_B),
        'mean_se_A': np.mean(fin_se_A), 'sd_se_A': np.std(fin_se_A),
        'mean_se_B': np.mean(fin_se_B), 'sd_se_B': np.std(fin_se_B)
    })

print("Simulations complete!")

script_dir = os.path.dirname(os.path.abspath(__file__))
df_summary = pd.DataFrame(summary_stats)
summary_path = os.path.join(script_dir, "exp_negativity_bias_100_summary.csv")
df_summary.to_csv(summary_path, index=False, sep=";", decimal=",")

plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

x_vals = df_summary['delta_drain']

ax1.plot(x_vals, df_summary['mean_op_A'], marker='s', color='navy', linewidth=3, markersize=8, label='Group A')
ax1.fill_between(x_vals, df_summary['mean_op_A'] - df_summary['sd_op_A'], df_summary['mean_op_A'] + df_summary['sd_op_A'], color='navy', alpha=0.15)
ax1.plot(x_vals, df_summary['mean_op_B'], marker='s', color='darkred', linewidth=3, markersize=8, label='Group B')
ax1.fill_between(x_vals, df_summary['mean_op_B'] - df_summary['sd_op_B'], df_summary['mean_op_B'] + df_summary['sd_op_B'], color='darkred', alpha=0.15)
ax1.set_title("Final Mean Opinion")
ax1.set_ylabel("Metric Value")

ax2.plot(x_vals, df_summary['mean_id_A'], marker='s', color='navy', linewidth=3, markersize=8)
ax2.fill_between(x_vals, df_summary['mean_id_A'] - df_summary['sd_id_A'], df_summary['mean_id_A'] + df_summary['sd_id_A'], color='navy', alpha=0.15)
ax2.plot(x_vals, df_summary['mean_id_B'], marker='s', color='darkred', linewidth=3, markersize=8)
ax2.fill_between(x_vals, df_summary['mean_id_B'] - df_summary['sd_id_B'], df_summary['mean_id_B'] + df_summary['sd_id_B'], color='darkred', alpha=0.15)
ax2.set_title("Final Mean Identity")

ax3.plot(x_vals, df_summary['mean_se_A'], marker='s', color='navy', linewidth=3, markersize=8)
ax3.fill_between(x_vals, df_summary['mean_se_A'] - df_summary['sd_se_A'], df_summary['mean_se_A'] + df_summary['sd_se_A'], color='navy', alpha=0.15)
ax3.plot(x_vals, df_summary['mean_se_B'], marker='s', color='darkred', linewidth=3, markersize=8)
ax3.fill_between(x_vals, df_summary['mean_se_B'] - df_summary['sd_se_B'], df_summary['mean_se_B'] + df_summary['sd_se_B'], color='darkred', alpha=0.15)
ax3.set_title("Final Mean Self-Esteem")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Drain Rate ($\delta^-$)")
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle='--', alpha=0.5)

ax1.legend()
fig.suptitle(r"Impact of Negativity Bias ($\delta^+$ = 0.05) on System Consensus (100 Seeds)", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
