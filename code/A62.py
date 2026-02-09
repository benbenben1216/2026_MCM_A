import numpy as np
import matplotlib.pyplot as plt

# 1. 定义模型参数
# 权重系数 (基于前面的灵敏度分析结果设定)
w_screen = 2.0  # 屏幕功耗权重
w_perf = 3.5  # 性能功耗权重 (游戏时很高)
w_net = 0.8  # 网络功耗权重
p_base = 0.5  # 基础待机功耗

# 用户偏好权重 (Alpha, Beta, Gamma)
# 假设用户最看重屏幕可见度，其次是流畅度
u_screen_w = 0.5
u_perf_w = 0.3
u_net_w = 0.2

# 2. 生成决策空间 (网格搜索)
x_s = np.linspace(0.1, 1.0, 20)  # 亮度不能为0
x_p = np.linspace(0.1, 1.0, 20)
x_n = np.linspace(0.1, 1.0, 5)  # 网络只有几个档位

# 结果容器
points_power = []
points_qoe = []
configs = []

for s in x_s:
    for p in x_p:
        for n in x_n:
            # 目标函数 1: 功耗 (越低越好)
            # 假设物理规律: 亮度是指数功耗, 性能是平方功耗
            power = p_base + w_screen * (s ** 1.4) + w_perf * (p ** 2.0) + w_net * n

            # 目标函数 2: QoE (越高越好)
            # 使用对数函数模拟边际效用递减: 亮度从10%提道20%的感知提升 > 90%提道100%
            qoe = u_screen_w * np.log(1 + 5 * s) + \
                  u_perf_w * np.log(1 + 3 * p) + \
                  u_net_w * n

            points_power.append(power)
            points_qoe.append(qoe)
            configs.append((s, p, n))

# 3. 寻找 Pareto 前沿
# 简单的筛选算法：对于每个点，如果没有其他点“既比它功耗低又比它体验好”，它就是 Pareto 最优
pareto_power = []
pareto_qoe = []
pareto_configs = []

data = sorted(zip(points_power, points_qoe, configs))  # 按功耗排序
current_max_qoe = -1

for p, q, c in data:
    if q > current_max_qoe:
        pareto_power.append(p)
        pareto_qoe.append(q)
        pareto_configs.append(c)
        current_max_qoe = q

# 4. 绘图
plt.figure(figsize=(10, 6))

# 所有方案点 (灰色)
plt.scatter(points_power, points_qoe, c='lightgray', s=5, label='Feasible Solutions')

# Pareto 前沿 (红色)
plt.plot(pareto_power, pareto_qoe, c='#e74c3c', linewidth=2, label='Pareto Frontier (Optimal Trade-offs)')
plt.scatter(pareto_power, pareto_qoe, c='#c0392b', s=30)

# 标注推荐点 (Knee Point)
# 寻找斜率突变点，或者距离理想点最近的点
knee_idx = len(pareto_power) // 2
knee_p, knee_q = pareto_power[knee_idx], pareto_qoe[knee_idx]
knee_conf = pareto_configs[knee_idx]

plt.annotate(f'Recommended Mode\n(Bright: {knee_conf[0]:.1f}, Perf: {knee_conf[1]:.1f})',
             xy=(knee_p, knee_q), xytext=(knee_p + 1, knee_q - 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Multi-Objective Optimization: Battery Life vs. User Experience', fontsize=14)
plt.xlabel('Total Power Consumption (Watts) [Lower is Better for Battery]', fontsize=12)
plt.ylabel('User Quality of Experience (QoE) [Higher is Better]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

print(f"推荐平衡点配置: 屏幕亮度={knee_conf[0]:.2f}, 性能释放={knee_conf[1]:.2f}, 网络={knee_conf[2]:.2f}")
