import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'


def plot_mixed_vs_constant_load():
    # 时间轴
    t = np.linspace(0, 10, 1000)

    # === 1. 持续高负载 (Constant Gaming) ===
    # 模拟一直玩游戏，平均电流大
    # 假设 5小时就没电了
    discharge_time_gaming = 5.0
    soc_gaming = 100 - (100 / discharge_time_gaming) * t
    soc_gaming = np.clip(soc_gaming, 0, 100)  # 不小于0

    # === 2. 慢速切换/混合模式 (Mixed Use) ===
    # 模拟: 1h 游戏 (高耗电) + 1h 视频/待机 (低耗电)
    # 这种模式下，平均电流比纯游戏小，所以能活得更久
    soc_mixed = [100.0]
    current_soc = 100.0
    dt = t[1] - t[0]
    period = 2.0  # 2小时一个周期

    for time_point in t[1:]:
        cycle_pos = time_point % period

        # 前1小时: 游戏 (掉电快, 斜率大)
        if cycle_pos < 1.0:
            rate = 20.0  # 每小时掉20% (和纯游戏类似)
        # 后1小时: 视频/待机 (掉电慢, 斜率小)
        else:
            rate = 3.0  # 每小时掉3%

        current_soc -= rate * dt
        soc_mixed.append(current_soc)

    soc_mixed = np.array(soc_mixed)
    soc_mixed = np.clip(soc_mixed, 0, 100)

    # === 绘图 ===
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    # 混合模式 (阶梯状)
    plt.plot(t, soc_mixed, color='#2980b9', linewidth=2.5,
             label='Mixed Pattern (Gaming + Standby)')

    # 持续高负载 (直线陡峭)
    # 只画到电量耗尽
    mask_gaming = soc_gaming > 0
    plt.plot(t[mask_gaming], soc_gaming[mask_gaming], color='#c0392b', linewidth=2.5, linestyle='--',
             label='Constant Gaming (Heavy Load)')
    # 补齐到0的垂直线(视觉上)或停在0

    # 辅助线
    plt.axhline(0, color='black', linewidth=1.2)

    # === 标注特征 ===

    # 1. 阶梯特征标注
    # 找一个平缓段
    mid_flat = 3.5  # 第3.5小时是平缓段
    idx_flat = np.argmin(np.abs(t - mid_flat))
    plt.text(mid_flat, soc_mixed[idx_flat] + 5, "Rest Phase\n(Low Discharge)",
             color='#2980b9', fontsize=11, ha='center', fontweight='bold')

    # 找一个陡峭段
    mid_steep = 2.5  # 第2.5小时是陡峭段
    idx_steep = np.argmin(np.abs(t - mid_steep))
    plt.text(mid_steep - 0.2, soc_mixed[idx_steep] - 10, "Active Phase\n(High Discharge)",
             color='#2980b9', fontsize=11, ha='right')

    # 2. 寿命对比 (TTE Gap)
    # 纯游戏 TTE = 5h
    # 混合模式 TTE = ? (计算一下)
    tte_mixed = t[np.argmin(np.abs(soc_mixed - 0))]
    # 标注差距
    plt.annotate('', xy=(tte_mixed, 5), xytext=(5.0, 5),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    plt.text((tte_mixed + 5.0) / 2, 10, f'Extended TTE\n+{tte_mixed - 5.0:.1f} Hours',
             color='purple', ha='center', fontweight='bold')

    plt.title('Battery Discharge Trajectory: Constant vs. Mixed Load', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Hours)', fontsize=14)
    plt.ylabel('State of Charge (%)', fontsize=14)
    plt.xlim(0, 9.5)
    plt.ylim(0, 105)
    plt.legend(loc='upper right', frameon=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mixed_vs_constant_load()
