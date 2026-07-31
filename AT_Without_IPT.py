import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os

from SSIM import run_simulation, n, num_steps, record_every

target_seed = 975569
print(f"Running Micro-Simulation Ablation (Seed: {target_seed})...")
print("Condition: Identity Process Theory DISABLED (gamma=0)")

run_data = run_simulation(seed=target_seed, gamma=0)

print("Simulation complete.")

print("\nFormatting micro-data for CSV export (this may take a few seconds)...")
micro_csv_rows = []
script_dir = os.path.dirname(os.path.abspath(__file__))

for step_idx in range(len(run_data['op_A'])):
    step_num = step_idx * record_every
    for agent_idx in range(n):
        micro_csv_rows.append({
            'seed': target_seed,
            'condition': 'no_SIA',
            'interaction_step': step_num,
            'agent_id': agent_idx,
            'group': run_data['agent_groups'][agent_idx],
            'opinion': run_data['ind_opinions'][step_idx][agent_idx],
            'identity': run_data['ind_identities'][step_idx][agent_idx],
            'self_esteem': run_data['ind_selfesteem'][step_idx][agent_idx]
        })

df_micro = pd.DataFrame(micro_csv_rows)
micro_save_path = os.path.join(script_dir, f"ablation_no_IPT_micro_seed_{target_seed}.csv")
df_micro.to_csv(micro_save_path, index=False, sep=";", decimal=",")

print(f"--> Micro-data successfully stored at:\n{micro_save_path}")

print("Drawing micro-level graphs...")
plt.rcParams.update({'font.size': 14}) 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6)) 

x_axis = np.linspace(0, num_steps / n, len(run_data['op_A']))
window_size = int(5000 / record_every)

opinions_matrix = np.array(run_data['ind_opinions'])
identities_matrix = np.array(run_data['ind_identities'])
selfesteem_matrix = np.array(run_data['ind_selfesteem'])

for i, group in enumerate(run_data['agent_groups']):
    color = 'steelblue' if group == "A" else 'indianred'
    ax1.plot(x_axis, opinions_matrix[:, i], color=color, alpha=0.15, linewidth=1)
    ax2.plot(x_axis, identities_matrix[:, i], color=color, alpha=0.15, linewidth=1)
    ax3.plot(x_axis, selfesteem_matrix[:, i], color=color, alpha=0.15, linewidth=1)

ax1.plot(x_axis, run_data['op_A'], color='navy', linewidth=3, label="Group A Mean")
ax1.plot(x_axis, run_data['op_B'], color='darkred', linewidth=3, label="Group B Mean")
ax2.plot(x_axis, run_data['id_A'], color='navy', linewidth=3)
ax2.plot(x_axis, run_data['id_B'], color='darkred', linewidth=3)

mean_se_A_roll = pd.Series(run_data['se_A']).rolling(window=window_size, min_periods=1).mean()
mean_se_B_roll = pd.Series(run_data['se_B']).rolling(window=window_size, min_periods=1).mean()
ax3.plot(x_axis, mean_se_A_roll, color='navy', linewidth=3)
ax3.plot(x_axis, mean_se_B_roll, color='darkred', linewidth=3)

fig.suptitle(f"Ablation: Identity Process Theory DISABLED | Micro Trajectories (Seed: {target_seed})", fontsize=18, fontweight='bold')

ax1.set_title("Fast Loop: Individual Opinions", pad=15)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel("Interactions per Agent")

ax2.set_title("Slow Loop: Individual Identities (No Shared Anchor)", pad=15)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel("Interactions per Agent")

ax3.set_title("Raw Individual Self-Esteem", pad=15)
ax3.set_ylim(-0.05, 1.05)
ax3.set_xlabel("Interactions per Agent")

ax1.legend(loc="upper right")
plt.tight_layout()
plt.show()
