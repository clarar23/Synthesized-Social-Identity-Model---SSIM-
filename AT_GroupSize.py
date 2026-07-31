import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

target_seed = 975569
proportions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
summary_stats = []      

print(f"Starting Group Size Sweep for Seed: {target_seed}...")

for prop in proportions:
    print(f"  Running Condition: prop_A = {prop:.2f}")
    run_data = run_simulation(seed=target_seed, prop_A=prop) 
    
    summary_stats.append({
        'prop_A': prop,
        'fin_op_A': run_data['op_A'][-1], 'fin_op_B': run_data['op_B'][-1],
        'fin_id_A': run_data['id_A'][-1], 'fin_id_B': run_data['id_B'][-1],
        'fin_se_A': run_data['se_A'][-1], 'fin_se_B': run_data['se_B'][-1]
    })

print("Sweep complete!")

script_dir = os.path.dirname(os.path.abspath(__file__))
df_summary = pd.DataFrame(summary_stats)
summary_path = os.path.join(script_dir, f"exp_groupsize_micro_seed_{target_seed}.csv")
df_summary.to_csv(summary_path, index=False, sep=";", decimal=",")

plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

x_vals = df_summary['prop_A'] * 100 

ax1.plot(x_vals, df_summary['fin_op_A'], marker='o', color='navy', linewidth=3, label='Group A')
ax1.plot(x_vals, df_summary['fin_op_B'], marker='o', color='darkred', linewidth=3, label='Group B')
ax1.set_title("Final Opinion")
ax1.set_ylabel("Parameter Value")

ax2.plot(x_vals, df_summary['fin_id_A'], marker='o', color='navy', linewidth=3)
ax2.plot(x_vals, df_summary['fin_id_B'], marker='o', color='darkred', linewidth=3)
ax2.set_title("Final Identity")

ax3.plot(x_vals, df_summary['fin_se_A'], marker='o', color='navy', linewidth=3)
ax3.plot(x_vals, df_summary['fin_se_B'], marker='o', color='darkred', linewidth=3)
ax3.set_title("Final Self-Esteem")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim(0, 1)
    ax.set_xlabel("Proportion of Activists (%)")
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle='--', alpha=0.5)

ax1.legend()
fig.suptitle(f"Impact of Minority Size on System Consensus (Seed: {target_seed})", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
