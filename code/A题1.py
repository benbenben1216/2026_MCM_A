import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class BasicBatteryModel:
    def __init__(self, capacity_mah, r0, r1, c1):
        """
        初始化电池参数 (参考用户上传论文中的典型参数)
        :param capacity_mah: 电池容量 (mAh)
        :param r0: 欧姆内阻 (Ohm)
        :param r1: 极化电阻 (Ohm)
        :param c1: 极化电容 (F)
        """
        self.Q_total = capacity_mah / 1000.0  # 转换为 Ah
        self.R0 = r0
        self.R1 = r1
        self.C1 = c1
        self.V_cutoff = 3.0  # 截止电压通常为 3.0V - 3.2V

    def get_ocv(self, soc):
        """
        OCV-SOC 曲线函数 (拟合锂离子电池典型曲线)
        SOC 0.0 -> ~3.0V
        SOC 1.0 -> ~4.2V
        使用一个典型的非线性多项式拟合
        """
        # 这是一个经验公式，模拟锂电池平坦区和两端的非线性
        # Uoc = -0.94*s^5 + 2.8*s^4 - 2.8*s^3 + 0.96*s^2 + 0.8*s + 3.0 (简化示例)
        # 为了更简单且物理合理，这里使用插值或简单的对数模型组合
        # 这里使用一个简化的 Nernst 方程变体供演示：
        return 3.0 + 1.2 * soc  # 线性近似 (简单模型)
        # *进阶：若要更精确，可以使用多项式拟合真实数据:
        # return 3.14 + 0.84*soc + (1-soc)*0.1 # 仅作示例

    def state_derivatives(self, t, state, current_amps):
        """
        定义微分方程组
        State = [SOC, U1]
        """
        soc, u1 = state

        # 1. d(SOC)/dt = -I / (3600 * Q)
        d_soc_dt = -current_amps / (3600.0 * self.Q_total)

        # 2. d(U1)/dt = I/C1 - U1/(R1*C1)
        d_u1_dt = (current_amps / self.C1) - (u1 / (self.R1 * self.C1))

        return [d_soc_dt, d_u1_dt]

    def solve_discharge(self, start_soc, load_current, duration_hours):
        """
        模拟放电过程
        :param start_soc: 初始 SOC (0-1)
        :param load_current: 恒定负载电流 (A)
        :param duration_hours: 模拟时长 (小时)
        """
        t_span = (0, duration_hours * 3600)  # 秒
        y0 = [start_soc, 0.0]  # 初始状态: SOC=start, U1=0 (假设静置开始)

        # 定义求解器所需的函数包装，固定 Current 参数
        fun = lambda t, y: self.state_derivatives(t, y, load_current)

        # 事件函数：当电压降到截止电压时停止模拟
        def cutoff_event(t, y):
            soc, u1 = y
            v_term = self.get_ocv(soc) - u1 - load_current * self.R0
            return v_term - self.V_cutoff

        cutoff_event.terminal = True  # 终止积分
        cutoff_event.direction = -1  # 仅检测下降穿过

        # 使用 Runge-Kutta 方法求解 ODE
        sol = solve_ivp(fun, t_span, y0, method='RK45', events=cutoff_event, dense_output=True)

        return sol


# --- 设置与运行 ---

# 1. 设置参数 (参考论文中的数量级，R0=0.07, R1=0.01, C1=2400)
# 假设电池容量 4000mAh
battery = BasicBatteryModel(capacity_mah=4000, r0=0.07, r1=0.01, c1=2400)

# 2. 模拟场景：恒定 0.5A 电流放电 (约等于玩轻量级游戏或看视频)
load_current = 0.5  # Amps
start_soc = 1.0  # 100%

print(f"开始模拟：负载电流 {load_current}A, 初始电量 {start_soc * 100}%")
solution = battery.solve_discharge(start_soc, load_current, duration_hours=10)

# 3. 提取结果
time_hours = solution.t / 3600
soc_values = solution.y[0]
u1_values = solution.y[1]

# 计算对应的端电压 V_term
v_ocv = np.array([battery.get_ocv(s) for s in soc_values])
v_term = v_ocv - u1_values - load_current * battery.R0

# --- 绘图 ---
fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()

# 绘制 SOC 曲线
color = 'tab:blue'
ax1.set_xlabel('Time (Hours)')
ax1.set_ylabel('SOC (0-1)', color=color)
ax1.plot(time_hours, soc_values, color=color, linewidth=2, label='SOC')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

# 绘制电压曲线
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Terminal Voltage (V)', color=color)
ax2.plot(time_hours, v_term, color=color, linewidth=2, linestyle='--', label='Voltage')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=3.0, color='black', linestyle=':', label='Cutoff (3.0V)')

# 标注耗尽时间
time_to_empty = time_hours[-1]
plt.title(f'Battery Drain Model (Constant Load {load_current}A)\nTime-to-Empty: {time_to_empty:.2f} Hours')
fig.tight_layout()
plt.show()

print(f"预测耗尽时间 (Time-to-Empty): {time_to_empty:.2f} 小时")
