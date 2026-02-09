import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# --- 1. 定义使用场景 ---
class UsageScenario:
    def __init__(self, name, screen, cpu, gpu, network, gps, temp_c):
        self.name = name
        self.components = {
            'Screen': screen,
            'CPU': cpu,
            'GPU': gpu,
            'Network': network,
            'GPS': gps,
            'Static': 0.1  # 基础电路待机功耗
        }
        self.total_power_w = sum(self.components.values())
        self.temp_c = temp_c


# 定义三种典型模式
scenarios = [
    UsageScenario("Standby Mode", screen=0.0, cpu=0.1, gpu=0.0, network=0.1, gps=0.0, temp_c=25),
    UsageScenario("Video Streaming", screen=1.2, cpu=0.5, gpu=0.2, network=0.6, gps=0.0, temp_c=30),
    UsageScenario("Heavy Gaming", screen=1.5, cpu=2.0, gpu=2.5, network=0.5, gps=0.0, temp_c=40)
]


# --- 2. 电池物理模型 (保留核心物理逻辑) ---
class BatteryPhysics:
    def __init__(self):
        # 拟合系数
        self.ocv_coeffs = [3.3487, 2.4407, -9.555, 20.922, -20.325, 7.381]
        self.Q_nom_Ah = 2.78
        self.R0_ref = 0.08
        self.R1 = 0.02
        self.C1 = 2000
        self.V_cutoff = 3.0

    def get_ocv(self, soc):
        soc = np.clip(soc, 0, 1)
        c = self.ocv_coeffs
        return c[0] + c[1] * soc + c[2] * soc ** 2 + c[3] * soc ** 3 + c[4] * soc ** 4 + c[5] * soc ** 5

    def get_params_at_temp(self, temp_c):
        temp_k = temp_c + 273.15
        ref_k = 298.15
        r_mult = np.exp(2500 * (1 / temp_k - 1 / ref_k))
        r0 = self.R0_ref * r_mult
        q_eff = self.Q_nom_Ah * 3600  # 库仑
        return r0, q_eff


# --- 3. 仿真求解器 ---
def run_simulation(scenario, battery):
    R0, Q_total_C = battery.get_params_at_temp(scenario.temp_c)
    P_load = scenario.total_power_w
    y0 = [1.0, 0.0]  # SOC=100%, U1=0V

    def derivatives(t, y):
        soc, u1 = y
        if soc <= 0: return [0, 0]

        v_ocv = battery.get_ocv(soc)
        v_internal = v_ocv - u1

        # 二次方程解端电压 (用于计算电流)
        delta = v_internal ** 2 - 4 * P_load * R0
        if delta < 0: return [-1e-3, 0]  # 电压崩溃
        v_term = (v_internal + np.sqrt(delta)) / 2

        i_current = P_load / v_term
        d_soc = -i_current / Q_total_C
        d_u1 = (i_current / battery.C1) - (u1 / (battery.R1 * battery.C1))
        return [d_soc, d_u1]

    # 事件检测
    def event_voltage_cutoff(t, y):
        v_ocv = battery.get_ocv(y[0])
        v_int = v_ocv - y[1]
        delta = v_int ** 2 - 4 * P_load * R0
        if delta < 0: return -1.0
        return ((v_int + np.sqrt(delta)) / 2) - battery.V_cutoff

    event_voltage_cutoff.terminal = True
    event_voltage_cutoff.direction = -1

    def event_soc_empty(t, y):
        return y[0] - 0.001

    event_soc_empty.terminal = True
    event_soc_empty.direction = -1

    sol = solve_ivp(derivatives, [0, 100 * 3600], y0,
                    events=[event_voltage_cutoff, event_soc_empty],
                    max_step=60, method='RK45')

    return {'sc': scenario, 'time': sol.t / 3600, 'soc': sol.y[0], 'dur': sol.t[-1] / 3600}


# --- 4. 运行仿真 ---
battery = BatteryPhysics()
results = [run_simulation(sc, battery) for sc in scenarios]

# --- 5. 可视化 (无电压曲线版) ---
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3)  # 2行3列

# 上半部分：SOC 曲线 (占据整行)
ax_soc = fig.add_subplot(gs[0, :])

# 绘制 SOC 曲线
colors = ['#2ca02c', '#1f77b4', '#d62728']  # 绿、蓝、红
for i, res in enumerate(results):
    c = colors[i]
    ax_soc.plot(res['time'], res['soc'] * 100, color=c, linewidth=2.5, label=f"{res['sc'].name}")

# 设置 SOC 图样式
ax_soc.set_title("Smartphone Battery Drain Prediction (SOC vs Time)", fontsize=16, pad=15)
ax_soc.set_ylabel("State of Charge (%)", fontsize=13)
ax_soc.set_xlabel("Time (Hours)", fontsize=13)
ax_soc.set_ylim(0, 105)  # 稍微留一点空间
ax_soc.set_xlim(left=0)
ax_soc.grid(True, alpha=0.3, linestyle='--')
ax_soc.legend(fontsize=12, loc='upper right')

# 添加说明文本
info_text = "Model logic: Discharge stops when SOC reaches 0% OR Voltage drops below 3.0V"
ax_soc.text(0.01, 0.02, info_text, transform=ax_soc.transAxes, color='gray', fontsize=10, style='italic')

# 下半部分：3个饼图 (功耗占比)
pie_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
pie_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']

for i, res in enumerate(results):
    ax = pie_axes[i]
    sc = res['sc']

    # 提取非零数据
    labels = []
    values = []
    for k, v in sc.components.items():
        if v > 0.05:  # 过滤极小值
            labels.append(k)
            values.append(v)

    # 绘制环形图 (Donut Chart)
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                      startangle=140, colors=pie_colors, pctdistance=0.85,
                                      textprops={'fontsize': 10})

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    # 中间写 耗尽时间
    ax.text(0, 0, f"{sc.total_power_w:.1f}W\n\n{res['dur']:.1f}h",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

    ax.set_title(f"{sc.name}", fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
plt.show()
