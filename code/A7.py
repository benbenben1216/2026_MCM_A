import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'


def generate_dynamic_profile(duration_hours=24, dt_min=1):
    n_steps = int(duration_hours * 60 / dt_min)

    # 状态: 0:Standby(0.5W), 1:Social(2W), 2:Gaming(6W)
    # 转移矩阵: P[i][j] 从状态i跳到状态j的概率
    P = np.array([
        [0.92, 0.07, 0.01],  # Standby 很难直接跳到 Gaming
        [0.15, 0.80, 0.05],  # Social 比较稳定
        [0.05, 0.20, 0.75]  # Gaming 容易退回 Social
    ])

    powers = [0.5, 2.0, 6.0]

    current_state = 0
    p_profile = []
    t_axis = []

    # 为了让图好看，我们增加一些高斯噪声
    np.random.seed(42)

    for t in range(n_steps):
        # 采样当前功率
        base = powers[current_state]
        noise = np.random.normal(0, base * 0.15)  # 15% 波动
        val = max(0.1, base + noise)

        # 偶尔加一个瞬时脉冲 (系统唤醒)
        if np.random.rand() < 0.02:
            val += 3.0

        p_profile.append(val)
        t_axis.append(t * dt_min / 60)

        # 状态转移
        current_state = np.random.choice([0, 1, 2], p=P[current_state])

    return np.array(t_axis), np.array(p_profile)


t, p = generate_dynamic_profile(12)  # 生成12小时

plt.figure(figsize=(8, 5))
plt.plot(t, p, color='#2c3e50', linewidth=1, label='Real-time Power Load')
plt.fill_between(t, p, color='#2c3e50', alpha=0.1)

plt.title('Assumption Check: Stochastic Usage Pattern (Markov Chain)', fontsize=14, fontweight='bold')
plt.ylabel('Power Consumption (Watts)', fontsize=12)
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylim(0, 10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
