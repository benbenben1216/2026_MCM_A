import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 设置绘图风格 (与论文风格一致)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'


def run_peukert_sensitivity():
    # === A. 复现随机负载 (与左图逻辑一致) ===
    duration_hours = 9.5
    dt_min = 1
    n_steps = int(duration_hours * 60 / dt_min)
    time_axis = np.linspace(0, duration_hours, n_steps)

    # 简单的随机负载生成 (模拟: 待机为主, 间歇性高脉冲)
    np.random.seed(42)  # 固定种子
    power_profile = []
    current_state = 0  # 0:Low, 1:High

    for _ in range(n_steps):
        if current_state == 0:
            p = 0.3 + np.random.normal(0, 0.05)  # 待机 0.3W
            if np.random.rand() < 0.1: current_state = 1  # 10%概率醒来
        else:
            p = 2.5 + np.random.normal(0, 0.5)  # 工作 2.5W
            # 偶尔极高脉冲
            if np.random.rand() < 0.1: p = 7.0
            if np.random.rand() < 0.3: current_state = 0  # 30%概率睡去
        power_profile.append(max(0.1, p))

    power_profile = np.array(power_profile)

    # === B. 核心：Peukert 灵敏度计算 ===
    # 公式: dQ = I^k * dt
    # 现象解释: 当 I < 1A 时, I^k < I (k越大越小)。
    # 由于大部分时间处于待机(0.3W/3.8V ≈ 0.08A)，k值越大，计算出的有效损耗反而越小。

    k_values = [1.00, 1.05, 1.10, 1.15, 1.20]
    results = {}

    # 电池设定
    capacity_ah = 3.2
    voltage_avg = 3.8

    for k in k_values:
        q_total_as = capacity_ah * 3600
        q_remain = q_total_as
        soc_history = []
        tte_point = None

        for idx, p in enumerate(power_profile):
            # 计算电流
            current = p / voltage_avg

            # Peukert 修正
            eff_current = current ** k

            # 积分
            q_remain -= eff_current * (dt_min * 60)
            soc = q_remain / q_total_as * 100
            soc_history.append(soc)

            # 记录耗尽时间
            if soc <= 0 and tte_point is None:
                tte_point = time_axis[idx]

        # 如果没耗尽，用最终时间
        if tte_point is None:
            tte_point = duration_hours

        results[k] = {'soc': soc_history, 'tte': tte_point}

    # === C. 绘图 ===
    plt.figure(figsize=(9, 6))

    # 使用渐变色板 (Flare)
    colors = sns.color_palette("flare", n_colors=len(k_values))

    for i, k in enumerate(k_values):
        data = results[k]
        soc = np.array(data['soc'])

        # 仅绘制 SOC > 0 的部分
        mask = soc > 0
        plt.plot(time_axis[mask], soc[mask],
                 color=colors[i],
                 linewidth=2.5,
                 label=f'k={k:.2f} (TTE={data["tte"]:.2f}h)')

    plt.title('Sensitivity to Discharge Efficiency (Peukert k)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('State of Charge (%)', fontsize=12)

    # 装饰
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.xlim(0, duration_hours)
    plt.ylim(-5, 105)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_peukert_sensitivity()
