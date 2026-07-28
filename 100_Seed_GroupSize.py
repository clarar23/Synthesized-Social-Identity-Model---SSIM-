import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

num_seeds = 100

# =====================================================================
# --- EXECUTION BLOCK: GROUP SIZE SWEEP (100 SEEDS) ---
# =====================================================================
proportions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
summary_stats = []      

print(f"Starting Group Size Sweep (100 seeds)...")

for prop in proportions:
    print(f"Running Condition: prop_A = {prop:.2f}")
    
    # Track all 3 metrics
    fin_op_A, fin_op_B = [], []
    fin_id_A, fin_id_B = [], []
    fin_se_A, fin_se_B = [], []
    
    for i in range(num_seeds):
        random_seed = np.random.randint(0, 1000000)
        run_data = run_simulation(seed=random_seed, prop_A=prop) 
        
        # Extract the final values at step 200,000 for all metrics
        fin_op_A.append(run_data['op_A'][-1])
        fin_op_B.append(run_data['op_B'][-1])
        fin_id_A.append(run_data['id_A'][-1])
        fin_id_B.append(run_data['id_B'][-1])
        fin_se_A.append(run_data['se_A'][-1])
        fin_se_B.append(run_data['se_B'][-1])

    # Calculate Means and SDs for all metrics
    summary_stats.append({
        'prop_A': prop,
        'mean_op_A': np.mean(fin_op_A), 'sd_op_A': np.std(fin_op_A),
        'mean_op_B': np.mean(fin_op_B), 'sd_op_B': np.std(fin_op_B),
        'mean_id_A': np.mean(fin_id_A), 'sd_id_A': np.std(fin_id_A),
        'mean_id_B': np.mean(fin_id_B), 'sd_id_B': np.std(fin_id_B),
        'mean_se_A': np.mean(fin_se_A), 'sd_se_A': np.std(fin_se_A),
        'mean_se_B': np.mean(fin_se_B), 'sd_se_B': np.std(fin_se_B)
    })

print("Simulations complete!")

# =====================================================================
# --- DATA EXPORT ---
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
df_summary = pd.DataFrame(summary_stats)
summary_path = os.path.join(script_dir, "exp_groupsize_100_summary.csv")
df_summary.to_csv(summary_path, index=False, sep=";", decimal=",")

# =====================================================================
# --- PLOTTING: 1x3 BIFURCATION GRID (100 SEEDS) ---
# =====================================================================
plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

x_vals = df_summary['prop_A'] * 100 

# --- PLOT 1: OPINIONS ---
ax1.plot(x_vals, df_summary['mean_op_A'], marker='o', color='navy', linewidth=3, label='Group A')
ax1.fill_between(x_vals, df_summary['mean_op_A'] - df_summary['sd_op_A'], df_summary['mean_op_A'] + df_summary['sd_op_A'], color='navy', alpha=0.15)
ax1.plot(x_vals, df_summary['mean_op_B'], marker='o', color='darkred', linewidth=3, label='Group B')
ax1.fill_between(x_vals, df_summary['mean_op_B'] - df_summary['sd_op_B'], df_summary['mean_op_B'] + df_summary['sd_op_B'], color='darkred', alpha=0.15)
ax1.set_title("Final Mean Opinion")
ax1.set_ylabel("Parameter Value")

# --- PLOT 2: IDENTITIES ---
ax2.plot(x_vals, df_summary['mean_id_A'], marker='o', color='navy', linewidth=3)
ax2.fill_between(x_vals, df_summary['mean_id_A'] - df_summary['sd_id_A'], df_summary['mean_id_A'] + df_summary['sd_id_A'], color='navy', alpha=0.15)
ax2.plot(x_vals, df_summary['mean_id_B'], marker='o', color='darkred', linewidth=3)
ax2.fill_between(x_vals, df_summary['mean_id_B'] - df_summary['sd_id_B'], df_summary['mean_id_B'] + df_summary['sd_id_B'], color='darkred', alpha=0.15)
ax2.set_title("Final Mean Identity")

# --- PLOT 3: SELF-ESTEEM ---
ax3.plot(x_vals, df_summary['mean_se_A'], marker='o', color='navy', linewidth=3)
ax3.fill_between(x_vals, df_summary['mean_se_A'] - df_summary['sd_se_A'], df_summary['mean_se_A'] + df_summary['sd_se_A'], color='navy', alpha=0.15)
ax3.plot(x_vals, df_summary['mean_se_B'], marker='o', color='darkred', linewidth=3)
ax3.fill_between(x_vals, df_summary['mean_se_B'] - df_summary['sd_se_B'], df_summary['mean_se_B'] + df_summary['sd_se_B'], color='darkred', alpha=0.15)
ax3.set_title("Final Mean Self-Esteem")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim(0, 1)
    ax.set_xlabel("Proportion of Activists (%)")
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle='--', alpha=0.5)

ax1.legend()
fig.suptitle("Impact of Minority Size on System Consensus (100 Seeds)", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()