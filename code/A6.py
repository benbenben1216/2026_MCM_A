import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'


# ==========================================
# 1. 核心计算模型 (保持物理逻辑)
# ==========================================
def calculate_tte(capacity_ah, soh, temp_c, power_w):
    # 简单的物理模型
    q_eff = capacity_ah * soh
    if temp_c < 20: q_eff *= (1 - 0.012 * (20 - temp_c))  # 低温容量衰减

    # 低温电压效率惩罚 (模拟内阻增大)
    v_eff = 1.0
    if temp_c < 10: v_eff = 0.85

    # Peukert效应 (大电流容量打折)
    p_eff = power_w * (1 + 0.05 * (power_w / 5.0))

    energy_wh = q_eff * 3.7 * v_eff
    return energy_wh / p_eff if p_eff > 0 else 0


# ==========================================
# 2. 定义三种场景的基准参数
# ==========================================
scenarios = {
    "Standby Mode": {
        "base_params": {'Screen': 0, 'CPU': 0.1, 'Net': 0.1, 'Bg': 0.1, 'Temp': 25, 'SOH': 0.95},
        "base_power": 0.3,  # W
        "color": "#2ecc71"  # Green
    },
    "Video Streaming": {
        "base_params": {'Screen': 1.0, 'CPU': 0.4, 'Net': 0.3, 'Bg': 0.1, 'Temp': 30, 'SOH': 0.95},
        "base_power": 1.8,  # W
        "color": "#3498db"  # Blue
    },
    "Heavy Gaming": {
        "base_params": {'Screen': 1.5, 'CPU': 3.5, 'Net': 0.5, 'Bg': 0.1, 'Temp': 40, 'SOH': 0.95},
        "base_power": 5.6,  # W
        "color": "#e74c3c"  # Red
    }
}

# ==========================================
# 3. 定义敏感性因子 (通用)
# ==========================================
# 定义变化的幅度 (Delta)
variations = [
    # (Factor Name, Param Key, Low Value, High Value, Display Label)
    ("Temperature", 'Temp', 0, 45, "Ambient Temp\n(0°C vs 45°C)"),
    ("Screen Brightness", 'Screen', 0.5, 2.0, "Screen Brightness\n(Dim vs Max)"),
    ("Processor Load", 'CPU', 0.1, 5.0, "CPU/GPU Load\n(Idle vs Peak)"),
    ("Network Signal", 'Net', 0.05, 1.5, "Signal Strength\n(Good vs Weak)"),
    ("Battery Health", 'SOH', 0.8, 1.0, "Battery Health\n(Old vs New)"),
]

# ==========================================
# 4. 循环生成三张图
# ==========================================

for sc_name, sc_data in scenarios.items():
    base_p = sc_data['base_params']


    # 重新计算该场景的基准 TTE (确保逻辑一致)
    # 注意：为了简化，我们直接用 base_power 计算 TTE，
    # 但敏感性分析时会修改 base_power 的组成部分

    # 辅助函数：根据参数计算总功率
    def get_p(p_dict):
        return p_dict['Screen'] + p_dict['CPU'] + p_dict['Net'] + p_dict['Bg']


    # 修正基准功率
    actual_base_power = get_p(base_p)
    base_tte = calculate_tte(4.0, base_p['SOH'], base_p['Temp'], actual_base_power)

    results = []

    for name, key, low_val, high_val, label in variations:
        # 特殊处理：如果是待机模式，屏幕亮度变化没有意义 (因为屏幕是关的)
        if sc_name == "Standby Mode" and key == 'Screen':
            continue

        # Low Case
        p_low = base_p.copy()
        p_low[key] = low_val
        if key in ['Screen', 'CPU', 'Net']:  # 这些影响功率
            power_low = get_p(p_low)
            tte_low = calculate_tte(4.0, p_low['SOH'], p_low['Temp'], power_low)
        else:  # 这些影响电池容量/内阻 (SOH, Temp)
            tte_low = calculate_tte(4.0, p_low['SOH'], p_low['Temp'], actual_base_power)

        # High Case
        p_high = base_p.copy()
        p_high[key] = high_val
        if key in ['Screen', 'CPU', 'Net']:
            power_high = get_p(p_high)
            tte_high = calculate_tte(4.0, p_high['SOH'], p_high['Temp'], power_high)
        else:
            tte_high = calculate_tte(4.0, p_high['SOH'], p_high['Temp'], actual_base_power)

        # 记录差值
        results.append({
            'Factor': label,
            'Low_Impact': tte_low - base_tte,  # 正值代表延寿
            'High_Impact': tte_high - base_tte,  # 负值代表减寿
            'Range': abs(tte_high - tte_low)
        })

    # 排序
    df_res = pd.DataFrame(results).sort_values('Range', ascending=True)

    # --- 绘图 (美化版) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制条形 (使用场景主题色)
    # 延寿的部分 (Low Setting) -> 绿色/正向
    # 减寿的部分 (High Setting) -> 灰色/负向 (突出对比)

    # 负向条 (Impact of Stress)
    ax.barh(df_res['Factor'], df_res['High_Impact'], color='#95a5a6', alpha=0.8, height=0.6, label='High Stress Impact')
    # 正向条 (Benefit of Optimization)
    ax.barh(df_res['Factor'], df_res['Low_Impact'], color=sc_data['color'], alpha=0.9, height=0.6,
            label='Optimization Benefit')

    # 添加中间基准线
    ax.axvline(0, color='black', linewidth=1.5, linestyle='-')

    # 添加数值标签
    for i, v in enumerate(df_res['Low_Impact']):
        ax.text(v + 0.1, i, f"+{v:.1f}h", va='center', fontsize=10, color=sc_data['color'], fontweight='bold')
    for i, v in enumerate(df_res['High_Impact']):
        ax.text(v - 0.2, i, f"{v:.1f}h", va='center', ha='right', fontsize=10, color='gray')

    # 标题和修饰
    ax.set_title(f'Sensitivity Analysis: {sc_name}\n(Baseline TTE: {base_tte:.1f} Hours)', fontsize=16,
                 fontweight='bold', pad=20)
    ax.set_xlabel('Change in Battery Life (Hours)', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # 调整坐标轴范围让图好看
    max_val = max(df_res['Low_Impact'].max(), abs(df_res['High_Impact'].min())) * 1.2
    ax.set_xlim(-max_val, max_val)

    # 图例
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.show()
