import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'


# ==========================================
# 1. 核心计算模型 (保持物理逻辑)
# ==========================================
def calculate_tte(capacity_ah, soh, temp_c, power_w):
    # 物理模型: Q_eff * V_eff / P_eff
    q_eff = capacity_ah * soh
    if temp_c < 20: q_eff *= (1 - 0.012 * (20 - temp_c))  # 低温容量衰减

    # 低温电压效率惩罚 (模拟内阻增大导致截止电压提前)
    v_eff = 1.0
    if temp_c < 10: v_eff = 0.85

    # Peukert效应 (大电流容量打折)
    p_eff = power_w * (1 + 0.05 * (power_w / 5.0))

    # 防止除零
    if p_eff <= 0: return 0

    energy_wh = q_eff * 3.7 * v_eff
    return energy_wh / p_eff


# ==========================================
# 2. 定义三种场景
# ==========================================
scenarios = [
    {
        "name": "Standby Mode",
        "base_params": {'Screen': 0, 'CPU': 0.1, 'Net': 0.1, 'Bg': 0.1, 'Temp': 25, 'SOH': 0.95},
        "color": "#27ae60"  # Green
    },
    {
        "name": "Video Streaming",
        "base_params": {'Screen': 1.0, 'CPU': 0.4, 'Net': 0.3, 'Bg': 0.1, 'Temp': 30, 'SOH': 0.95},
        "color": "#2980b9"  # Blue
    },
    {
        "name": "Heavy Gaming",
        "base_params": {'Screen': 1.5, 'CPU': 3.5, 'Net': 0.5, 'Bg': 0.1, 'Temp': 40, 'SOH': 0.95},
        "color": "#c0392b"  # Red
    }
]

# 定义通用敏感性因子变化
variations = [
    ("Temperature", 'Temp', 0, 45, "Ambient Temp"),
    ("Screen Brightness", 'Screen', 0.5, 2.0, "Screen Brightness"),
    ("Processor Load", 'CPU', 0.1, 5.0, "CPU/GPU Load"),
    ("Network Signal", 'Net', 0.05, 1.5, "Signal Strength"),
    ("Battery Health", 'SOH', 0.8, 1.0, "Battery Health"),
]

# ==========================================
# 3. 联合绘图
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # sharey 让Y轴标签只显示一次


def get_p(p_dict):
    return p_dict['Screen'] + p_dict['CPU'] + p_dict['Net'] + p_dict['Bg']


for idx, sc in enumerate(scenarios):
    ax = axes[idx]
    base_p = sc['base_params']

    # 计算基准 TTE
    actual_base_power = get_p(base_p)
    base_tte = calculate_tte(4.0, base_p['SOH'], base_p['Temp'], actual_base_power)

    results = []

    for name, key, low_val, high_val, label in variations:
        # 待机模式跳过屏幕
        if sc['name'] == "Standby Mode" and key == 'Screen':
            continue

        # Low Case (Optimization)
        p_low = base_p.copy()
        p_low[key] = low_val
        if key in ['Screen', 'CPU', 'Net', 'Bg']:
            p_now = get_p(p_low)
            tte_low = calculate_tte(4.0, p_low['SOH'], p_low['Temp'], p_now)
        else:
            tte_low = calculate_tte(4.0, p_low['SOH'], p_low['Temp'], actual_base_power)

        # High Case (Stress)
        p_high = base_p.copy()
        p_high[key] = high_val
        if key in ['Screen', 'CPU', 'Net', 'Bg']:
            p_now = get_p(p_high)
            tte_high = calculate_tte(4.0, p_high['SOH'], p_high['Temp'], p_now)
        else:
            tte_high = calculate_tte(4.0, p_high['SOH'], p_high['Temp'], actual_base_power)

        results.append({
            'Factor': label,
            'Low_Impact': tte_low - base_tte,
            'High_Impact': tte_high - base_tte,
            'Range': abs(tte_high - tte_low)
        })

    # 排序
    df_res = pd.DataFrame(results).sort_values('Range', ascending=True)

    # 绘图逻辑
    # 负向条 (Stress)
    ax.barh(df_res['Factor'], df_res['High_Impact'], color='#95a5a6', alpha=0.6, label='Stress Impact')
    # 正向条 (Benefit)
    bars = ax.barh(df_res['Factor'], df_res['Low_Impact'], color=sc['color'], alpha=0.9, label='Optimization Benefit')

    # 基准线
    ax.axvline(0, color='black', linewidth=1, linestyle='-')

    # 标题
    ax.set_title(f"{sc['name']}\n(Base TTE: {base_tte:.1f}h)", fontsize=14, fontweight='bold', color=sc['color'])
    ax.set_xlabel('Change in TTE (Hours)')

    # 在 Standby 模式（时间特别长）做一些刻度限制，防止把其他图挤没了
    # 或者让 X 轴不共享，独立缩放（默认就是独立的）

    # 添加数值标签 (仅在 Optimization 侧，避免拥挤)
    for i, v in enumerate(df_res['Low_Impact']):
        ax.text(v, i, f" +{v:.1f}h", va='center', fontsize=9, color=sc['color'], fontweight='bold')

    ax.grid(axis='x', linestyle='--', alpha=0.5)

# 统一调整
plt.suptitle("Scenario-Based Sensitivity Analysis: What Drives Battery Drain?", fontsize=18, y=1.05)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # 增加子图间距

# 只在第一个图显示图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

plt.show()
