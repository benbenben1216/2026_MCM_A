import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp


# --- 1. 电池物理模型 (含温度敏感性) ---
class BatteryPhysics:
    def __init__(self):
        # 拟合系数 (OCV vs SOC)
        self.ocv_coeffs = [3.3487, 2.4407, -9.555, 20.922, -20.325, 7.381]
        self.Q_nom_Ah = 2.78  # 标称容量
        self.R0_ref = 0.08  # 25度时的参考内阻
        self.R1 = 0.02
        self.C1 = 2000
        self.V_cutoff = 3.0  # 关机电压

    def get_ocv(self, soc):
        soc = np.clip(soc, 0, 1)
        c = self.ocv_coeffs
        return c[0] + c[1] * soc + c[2] * soc ** 2 + c[3] * soc ** 3 + c[4] * soc ** 4 + c[5] * soc ** 5

    def get_params_at_temp(self, temp_c):
        """
        温度修正模型：
        1. Arrhenius方程修正内阻 (低温内阻呈指数级升高)
        2. 线性/经验公式修正容量 (低温活性降低)
        """
        temp_k = temp_c + 273.15
        ref_k = 298.15  # 25°C

        # 内阻修正 (Beta ~ 2500-3000)
        r_mult = np.exp(3000 * (1 / temp_k - 1 / ref_k))
        r0 = self.R0_ref * r_mult

        # 容量修正 (假设 20度以下每降低1度，容量衰减 1.2%)
        if temp_c < 20:
            q_factor = 1.0 - 0.012 * (20 - temp_c)
        else:
            q_factor = 1.0  # 高温通常不增加容量，维持标称

        q_eff_ah = self.Q_nom_Ah * np.clip(q_factor, 0.4, 1.05)
        return r0, q_eff_ah * 3600  # 返回 欧姆 和 库仑


# --- 2. 仿真求解器 ---
def run_simulation(temp_c, power_watts, battery):
    R0, Q_total_C = battery.get_params_at_temp(temp_c)

    # 定义导数和事件
    def derivatives(t, y):
        soc, u1 = y
        if soc <= 0: return [0, 0]
        v_ocv = battery.get_ocv(soc)
        v_int = v_ocv - u1
        # 二次方程解电压
        delta = v_int ** 2 - 4 * power_watts * R0
        if delta < 0: return [-1e-3, 0]  # 电压崩溃
        v_term = (v_int + np.sqrt(delta)) / 2

        i = power_watts / v_term
        return [-i / Q_total_C, i / battery.C1 - u1 / (battery.R1 * battery.C1)]

    # 终止条件：电压 < 3.0V
    def event_voltage_cutoff(t, y):
        v_ocv = battery.get_ocv(y[0])
        delta = (v_ocv - y[1]) ** 2 - 4 * power_watts * R0
        if delta < 0: return -1.0
        return ((v_ocv - y[1] + np.sqrt(delta)) / 2) - battery.V_cutoff

    event_voltage_cutoff.terminal = True

    # 终止条件：SOC < 0
    def event_soc_empty(t, y):
        return y[0] - 0.001

    event_soc_empty.terminal = True

    sol = solve_ivp(derivatives, [0, 48 * 3600], [1.0, 0.0],
                    events=[event_voltage_cutoff, event_soc_empty],
                    max_step=60, method='RK45')

    return {
        'temp': temp_c,
        'time': sol.t / 3600,
        'soc': sol.y[0],
        'duration': sol.t[-1] / 3600,
        'final_soc': sol.y[0][-1]
    }


# --- 3. 设置变量与运行 ---
battery = BatteryPhysics()
# 固定场景：重度游戏 (5.5W)，观察不同温度的表现
fixed_load_watts = 5.5
temperatures = [-10, 0, 10, 25, 40]  # 摄氏度

results = []
for t in temperatures:
    res = run_simulation(t, fixed_load_watts, battery)
    results.append(res)

# --- 4. 可视化 ---
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

# 图1: SOC 曲线
ax1 = fig.add_subplot(gs[0])
# 使用 Cool-Warm 颜色映射
colormap = cm.get_cmap('coolwarm', len(temperatures))

for i, res in enumerate(results):
    color = colormap(i)
    label = f"{res['temp']}°C (Dur: {res['duration']:.2f}h)"
    ax1.plot(res['time'], res['soc'] * 100, color=color, linewidth=2.5, label=label)

    # 在曲线末端标注终止原因
    if res['final_soc'] > 0.01:
        ax1.text(res['time'][-1], res['soc'][-1] * 100 + 1, "Low Voltage!",
                 color=color, fontsize=9, fontweight='bold')

ax1.set_title(f"Impact of Temperature on SOC (Load: {fixed_load_watts}W)", fontsize=14)
ax1.set_xlabel("Time (Hours)", fontsize=12)
ax1.set_ylabel("SOC (%)", fontsize=12)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)
ax1.legend(title="Ambient Temp")

# 图2: 续航时间条形图
ax2 = fig.add_subplot(gs[1])
temps = [str(r['temp']) + "°C" for r in results]
durs = [r['duration'] for r in results]
colors = [colormap(i) for i in range(len(temperatures))]

bars = ax2.bar(temps, durs, color=colors, alpha=0.8)
ax2.set_title("Runtime vs Temperature", fontsize=14)
ax2.set_xlabel("Temperature", fontsize=12)
ax2.set_ylabel("Time-to-Empty (Hours)", fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# 在柱子上标数值
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}h', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
