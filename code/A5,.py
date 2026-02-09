import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

# ==========================================
# 1. 模拟数据加载 (请替换为读取你的 excel)
# ==========================================
# 模拟 1000 条真实数据，包含一些极端的长尾数据
np.random.seed(2026)
data_size = 1000
mock_data = {
    'Q_full_Ah': np.concatenate([np.random.normal(2.7, 0.05, 800), np.random.normal(2.2, 0.1, 200)]),  # 新旧混合
    'P_total_uW': np.concatenate([np.random.normal(5.5e6, 0.5e6, 900), np.random.normal(8.0e6, 1.0e6, 100)]),
    # 大部分5.5W，偶尔飙升到8W
    'SOH': np.concatenate([np.ones(800), np.random.uniform(0.8, 0.95, 200)])
}
df = pd.DataFrame(mock_data)


# 简化的物理参数
def get_ocv_simple(soc):
    return 3.4 + 0.5 * soc - 0.5 * np.exp(-20 * soc)


def run_simulation(power_w, q_ah, r_int):
    # 简单的欧拉法加速演示 (实际用 solve_ivp)
    dt = 60  # 1分钟一步
    t = 0
    soc = 1.0
    time_log = [0]
    soc_log = [1.0]

    while soc > 0 and t < 10 * 3600:
        v_ocv = get_ocv_simple(soc)
        delta = v_ocv ** 2 - 4 * r_int * power_w
        if delta < 0: break  # 电压崩溃
        current = (v_ocv - np.sqrt(delta)) / (2 * r_int)

        d_soc = -current / (3600 * q_ah) * dt
        soc += d_soc
        t += dt

        # 记录
        if t % 300 == 0:  # 每5分钟记录一次以绘图
            time_log.append(t / 3600)
            soc_log.append(max(0, soc))

    return t / 3600, time_log, soc_log


# ==========================================
# 2. 运行数据驱动模拟
# ==========================================
N_SIM = 300  # 模拟次数
results = []
trajectories = []

for _ in range(N_SIM):
    # Bootstrap 采样
    sample = df.sample(1).iloc[0]
    p_w = sample['P_total_uW'] / 1e6
    q = sample['Q_full_Ah']
    r = 0.12 * (1 + (1 - sample['SOH']))  # SOH修正内阻

    tte, t_arr, s_arr = run_simulation(p_w, q, r)
    results.append(tte)
    trajectories.append((t_arr, s_arr))

# ==========================================
# 3. 绘制最终组合图
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
fig = plt.figure(figsize=(14, 6))

# --- 子图 A: 轨迹扇形图 (保留 Image 1 的美感) ---
ax1 = fig.add_subplot(121)
# 绘制所有轨迹
for t, s in trajectories:
    ax1.plot(t, np.array(s) * 100, color='gray', alpha=0.08, linewidth=1)

# 绘制平均轨迹 (Nominal)
mean_p = df['P_total_uW'].mean() / 1e6
mean_q = df['Q_full_Ah'].mean()
mean_tte, t_nom, s_nom = run_simulation(mean_p, mean_q, 0.12)
ax1.plot(t_nom, np.array(s_nom) * 100, color='#2c3e50', linewidth=2.5, label='Mean Data Profile')

ax1.set_title('(a) Stochastic SOC Trajectories', fontweight='bold')
ax1.set_xlabel('Time (Hours)')
ax1.set_ylabel('State of Charge (%)')
ax1.set_ylim(0, 100)
ax1.set_xlim(0, max(results) * 1.1)
ax1.legend()
ax1.text(0.02, 0.03, "Source: Bootstrap Sampling\nfrom Master Table", transform=ax1.transAxes,
         fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

# --- 子图 B: 数据驱动分布图 (保留 Image 2 的真实性) ---
ax2 = fig.add_subplot(122)
sns.histplot(results, kde=True, color='teal', alpha=0.6, ax=ax2, stat='density')

# 标注统计量
mean_val = np.mean(results)
ci_low, ci_high = np.percentile(results, [2.5, 97.5])

ax2.axvline(mean_val, color='k', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}h')
ax2.axvline(ci_low, color='#e74c3c', linestyle=':', linewidth=1.5, label='95% CI')
ax2.axvline(ci_high, color='#e74c3c', linestyle=':', linewidth=1.5)

ax2.set_title('(b) Data-Driven TTE Distribution', fontweight='bold')
ax2.set_xlabel('Time-to-Empty (Hours)')
ax2.set_ylabel('Probability Density')
ax2.legend()

plt.tight_layout()
plt.show()
