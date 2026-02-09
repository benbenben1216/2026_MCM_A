import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==========================================
# 1. 马尔可夫链负载生成器 (Markov Chain Generator)
# ==========================================
def generate_dynamic_profile(duration_hours, transition_matrix, power_states, dt_min=1):
    """
    生成一个随时间跳变的用户负载曲线
    :param duration_hours: 模拟时长
    :param transition_matrix: 3x3 转移概率矩阵
    :param power_states: [Standby_W, Video_W, Gaming_W]
    :param dt_min: 时间步长(分钟)
    """
    n_steps = int(duration_hours * 60 / dt_min)

    # 状态映射: 0:Standby, 1:Video, 2:Gaming
    current_state = 0  # 初始默认为待机
    power_profile = []
    time_points = []

    # 累积概率用于采样
    cum_probs = np.cumsum(transition_matrix, axis=1)

    for t in range(n_steps):
        # 记录当前功率
        # 添加一点高斯噪声，模拟同一状态下的微小波动
        base_p = power_states[current_state]
        noise = np.random.normal(0, base_p * 0.1)  # 10% 波动
        actual_p = max(0.05, base_p + noise)

        power_profile.append(actual_p)
        time_points.append(t * dt_min / 60)  # 小时

        # 决定下一时刻状态
        r = np.random.rand()
        # 根据当前状态的转移概率选择下一个状态
        if r < cum_probs[current_state][0]:
            next_state = 0
        elif r < cum_probs[current_state][1]:
            next_state = 1
        else:
            next_state = 2
        current_state = next_state

    return np.array(time_points), np.array(power_profile)


# ==========================================
# 2. 电池动态模型 (含参数扰动接口)
# ==========================================
def run_battery_simulation(time_hours, power_profile, q_total_ah, peukert_k=1.05, temp_c=25):
    """
    在动态负载下运行电池耗尽模拟
    """
    dt = (time_hours[1] - time_hours[0]) * 3600  # 秒

    # 初始化
    current_soc = 1.0
    q_remain_as = q_total_ah * 3600  # Amp-seconds

    # 简化的内阻模型 (Ohm)
    r_int = 0.15 * (1 + 0.01 * (25 - temp_c))  # 低温内阻增加

    soc_history = [1.0]
    voltage_history = []

    survival_time = 0

    for i, p_load in enumerate(power_profile):
        if current_soc <= 0:
            break

        # 1. 计算 OCV (开路电压) - 简化的 SOC-OCV 曲线
        # V = 3.0 + 0.7*SOC + 0.5*SOC^3
        v_ocv = 3.0 + 0.7 * current_soc + 0.5 * (current_soc ** 3)

        # 2. 计算电流 I (P = VI -> P = (V_ocv - IR)I)
        # I^2 R - V_ocv I + P = 0
        delta = v_ocv ** 2 - 4 * r_int * p_load

        if delta < 0:  # 电压崩溃
            voltage_history.append(3.0)  # 记录截止电压
            break

        current = (v_ocv - np.sqrt(delta)) / (2 * r_int)

        # 检查截止电压 (3.0V)
        v_term = v_ocv - current * r_int
        voltage_history.append(v_term)

        if v_term < 3.0:
            break

        # 3. 库仑积分 (引入 Peukert 效应扰动)
        # 有效电流 I_eff = I^k
        i_eff = current ** peukert_k

        # 更新容量
        # dQ = I_eff * dt
        dq = i_eff * dt
        q_remain_as -= dq
        current_soc = q_remain_as / (q_total_ah * 3600)

        soc_history.append(max(0, current_soc))
        survival_time = time_hours[i]

    return survival_time, soc_history, voltage_history


# ==========================================
# 3. 执行灵敏度分析
# ==========================================
# 参数设置
TRANSITION_MATRIX = np.array([
    [0.95, 0.04, 0.01],  # Standby -> ...
    [0.10, 0.85, 0.05],  # Video -> ...
    [0.05, 0.25, 0.70]  # Gaming -> ... (游戏容易退回视频，较难直接待机)
])
POWER_STATES = [0.3, 2.5, 6.0]  # W

# A. 生成一次典型的动态负载
t_axis, p_profile = generate_dynamic_profile(24, TRANSITION_MATRIX, POWER_STATES)

# B. 扰动测试：Peukert 系数 (k) 的影响
k_values = [1.0, 1.05, 1.1, 1.15, 1.2]  # 1.0=理想线性, 1.2=严重非线性
results = []

print("Running Sensitivity Analysis on Peukert Coefficient...")

plt.figure(figsize=(12, 5))

# 子图 1: 动态负载展示
plt.subplot(1, 2, 1)
plt.plot(t_axis, p_profile, color='#34495e', linewidth=0.5, alpha=0.8)
plt.fill_between(t_axis, p_profile, color='#34495e', alpha=0.1)
plt.title('Stochastic Usage Pattern (Markov Chain)', fontsize=12)
plt.xlabel('Time (Hours)')
plt.ylabel('Power Load (Watts)')
plt.grid(True, alpha=0.3)

# 子图 2: 不同假设下的 SOC 轨迹
plt.subplot(1, 2, 2)

colors = sns.color_palette("flare", len(k_values))

for idx, k in enumerate(k_values):
    tte, soc_hist, _ = run_battery_simulation(t_axis, p_profile, 4.0, peukert_k=k)

    # 绘图
    # 注意 soc_hist 长度可能比 t_axis 短（如果提前耗尽）
    sim_len = len(soc_hist)
    plt.plot(t_axis[:sim_len], np.array(soc_hist) * 100, color=colors[idx], label=f'k={k:.2f} (TTE={tte:.2f}h)')

    # 计算灵敏度
    # 假设 k=1.05 是基准
    if k == 1.05:
        base_tte = tte
    else:
        pass  # 后续处理

plt.axhline(0, color='black', linestyle='--')
plt.title('Sensitivity to Discharge Efficiency (Peukert k)', fontsize=12)
plt.xlabel('Time (Hours)')
plt.ylabel('State of Charge (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 计算相对灵敏度
print(f"Baseline TTE (k=1.05): {base_tte:.2f} hours")
