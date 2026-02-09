import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

# ==========================================
# 1. 数据加载与预处理 (模拟读取你的Excel)
# ==========================================
# 在实际运行时，请使用: df = pd.read_excel('master_modeling_table.xlsx')
# 这里我们手动构造一些符合你数据结构的示例数据以确保代码可运行
data = {
    'battery_state_id': ['Cell01_new'] * 50 + ['Cell01_aged'] * 50,
    'Q_full_Ah': [2.77] * 50 + [2.20] * 50,  # 新电池2.77Ah, 老化电池2.2Ah
    'P_total_uW': np.concatenate([np.random.normal(5500000, 500000, 50), np.random.normal(5500000, 1000000, 50)]),
    # 约5.5W
    # 假设 OCV 系数 (简化)
    'ocv_c0': [3.34] * 100, 'ocv_c1': [2.44] * 100, 'ocv_c2': [-9.55] * 100,
    'ocv_c3': [20.92] * 100, 'ocv_c4': [-20.32] * 100, 'ocv_c5': [7.38] * 100,
    'SOH': [1.0] * 50 + [0.8] * 50,
    'temp_c': [33.3] * 100
}
df = pd.DataFrame(data)


# ==========================================
# 2. 定义基于多项式的物理模型
# ==========================================
def get_poly_ocv(soc, coeffs):
    """ 使用数据中的 c0-c5 系数计算 OCV """
    # coeffs = [c0, c1, c2, c3, c4, c5]
    soc = np.clip(soc, 0, 1)
    return sum(c * (soc ** i) for i, c in enumerate(coeffs))


def battery_dynamics(t, y, power_w, r_int, q_ah, ocv_coeffs):
    soc = y[0]
    if soc <= 0: return [-1e-7]

    v_ocv = get_poly_ocv(soc, ocv_coeffs)

    # P = VI -> P = (OCV - IR)I -> I^2 R - OCV I + P = 0
    delta = v_ocv ** 2 - 4 * r_int * power_w

    if delta < 0: return [-0.5]  # 电压崩溃

    current = (v_ocv - np.sqrt(delta)) / (2 * r_int)

    # 库仑计数
    d_soc_dt = -current / (3600 * q_ah)
    return [d_soc_dt]


# ==========================================
# 3. 数据驱动的蒙特卡洛模拟
# ==========================================
N_SIMULATIONS = 500
results_tte = []

print(f"Running {N_SIMULATIONS} Data-Driven Simulations...")

for i in range(N_SIMULATIONS):
    # --- 关键修改：从真实数据行中采样 ---

    # A. 随机抽取一个电池状态 (模拟电池制造和老化的不确定性)
    # 随机选一行数据
    battery_row = df.sample(1).iloc[0]

    # 提取该电池的具体参数
    q_ah = battery_row['Q_full_Ah']
    ocv_coeffs = [battery_row[f'ocv_c{k}'] for k in range(6)]

    # 内阻模型 (简单温度修正)
    temp_c = battery_row['temp_c']  # 这里的温度也是数据里真实的
    # R = R_ref * exp(...) 简单起见我们取个基准值修正
    r_int = 0.1 * (1 + 0.5 * (1 - battery_row['SOH']))  # SOH越低内阻越大

    # B. 随机抽取一个负载功率 (模拟用户行为波动)
    # 从 P_total_uW 列采样并转换为瓦特
    # 这里可以添加逻辑：比如只从 'Gaming' 相关的行里抽
    power_sample_row = df.sample(1).iloc[0]
    power_w = power_sample_row['P_total_uW'] / 1e6


    # -----------------------------------

    # 运行 ODE
    def cutoff_event(t, y):
        soc = y[0]
        if soc <= 0.001: return 0
        v_ocv = get_poly_ocv(soc, ocv_coeffs)
        delta = v_ocv ** 2 - 4 * r_int * power_w
        if delta < 0: return 0
        current = (v_ocv - np.sqrt(delta)) / (2 * r_int)
        return (v_ocv - current * r_int) - 3.0


    cutoff_event.terminal = True
    cutoff_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, y: battery_dynamics(t, y, power_w, r_int, q_ah, ocv_coeffs),
        t_span=[0, 24 * 3600],
        y0=[1.0],
        events=cutoff_event,
        method='RK45'
    )

    tte = sol.t[-1] / 3600 if sol.t_events[0].size == 0 else sol.t_events[0][0] / 3600
    results_tte.append(tte)

# ==========================================
# 4. 绘图
# ==========================================
plt.figure(figsize=(8, 5))
sns.histplot(results_tte, kde=True, color='teal', alpha=0.6)
plt.axvline(np.mean(results_tte), color='k', linestyle='--', label='Mean TTE')
plt.title('Data-Driven Uncertainty Quantification\n(Sampling directly from AndroWatts & Mendeley Data)')
plt.xlabel('Time-to-Empty (Hours)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mean TTE: {np.mean(results_tte):.2f} hours")
print(
    f"Uncertainty Range (95% CI): {np.percentile(results_tte, 2.5):.2f} - {np.percentile(results_tte, 97.5):.2f} hours")
