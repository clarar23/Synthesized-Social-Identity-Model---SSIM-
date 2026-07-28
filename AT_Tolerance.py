import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

# =====================================================================
# --- EXECUTION BLOCK (SINGLE SEED SWEEP) ---
# =====================================================================
target_seed = 975569
tolerances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
summary_stats = []      

print(f"Starting Tolerance Sweep for Seed: {target_seed}...")

for tol in tolerances:
    print(f"  Running Condition: threshold_base = {tol:.1f}")
    run_data = run_simulation(seed=target_seed, threshold_base=tol) 
    
    # Store the final values for all 3 metrics
    summary_stats.append({
        'threshold_base': tol,
        'fin_op_A': run_data['op_A'][-1], 'fin_op_B': run_data['op_B'][-1],
        'fin_id_A': run_data['id_A'][-1], 'fin_id_B': run_data['id_B'][-1],
        'fin_se_A': run_data['se_A'][-1], 'fin_se_B': run_data['se_B'][-1]
    })

print("Sweep complete!")

# =====================================================================
# --- DATA EXPORT ---
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
df_summary = pd.DataFrame(summary_stats)
summary_path = os.path.join(script_dir, f"exp_tolerance_micro_seed_{target_seed}.csv")
df_summary.to_csv(summary_path, index=False, sep=";", decimal=",")

# =====================================================================
# --- PLOTTING: 1x3 BIFURCATION GRID ---
# =====================================================================
plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

x_vals = df_summary['threshold_base']

# --- PLOT 1: OPINIONS ---
ax1.plot(x_vals, df_summary['fin_op_A'], marker='o', color='navy', linewidth=3, label='Group A')
ax1.plot(x_vals, df_summary['fin_op_B'], marker='o', color='darkred', linewidth=3, label='Group B')
ax1.set_title("Final Mean Opinion")
ax1.set_ylabel("Metric Value")

# --- PLOT 2: IDENTITIES ---
ax2.plot(x_vals, df_summary['fin_id_A'], marker='o', color='navy', linewidth=3)
ax2.plot(x_vals, df_summary['fin_id_B'], marker='o', color='darkred', linewidth=3)
ax2.set_title("Final Identity Standard")

# --- PLOT 3: SELF-ESTEEM ---
ax3.plot(x_vals, df_summary['fin_se_A'], marker='o', color='navy', linewidth=3)
ax3.plot(x_vals, df_summary['fin_se_B'], marker='o', color='darkred', linewidth=3)
ax3.set_title("Final Self-Esteem")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim(0, 1)
    ax.set_xlabel("Baseline Tolerance")
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle='--', alpha=0.5)

ax1.legend()
fig.suptitle(f"Impact of Tolerance on System Consensus (Seed: {target_seed})", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()