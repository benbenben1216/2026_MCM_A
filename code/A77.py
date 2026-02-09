import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'


def plot_monte_carlo_error():
    # 参数设置
    n_samples = 5000
    soc_levels = [0.2, 0.5, 0.8]  # 低、中、高三种初始电量
    sigma_soc = 0.02  # 初始误差 2%

    # 模拟函数: 非线性 TTE 响应
    # 假设 TTE = f(SOC) = k * SOC / (1 + a/SOC)
    # (物理意义: 低SOC时电压下降快，可用容量非线性缩减)
    def tte_model(soc):
        base_time = soc * 5.0  # 线性基准
        nonlinear_penalty = 1 / (1 + 0.1 / soc)  # 低电量惩罚因子
        return base_time * nonlinear_penalty

    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#f1c40f', '#2ecc71']
    labels = ['Low SOC (20%)', 'Mid SOC (50%)', 'High SOC (80%)']

    for i, mu_soc in enumerate(soc_levels):
        # 1. 生成带有高斯噪声的初始 SOC
        soc_samples = np.random.normal(mu_soc, sigma_soc, n_samples)
        soc_samples = np.clip(soc_samples, 0.05, 1.0)  # 物理限制

        # 2. 计算对应的 TTE
        tte_samples = tte_model(soc_samples)

        # 3. 归一化处理 (转化为百分比偏差)
        tte_ref = tte_model(mu_soc)
        tte_pct_change = (tte_samples - tte_ref) / tte_ref * 100

        # 4. 绘制核密度估计 (KDE)
        sns.kdeplot(tte_pct_change, fill=True, color=colors[i], alpha=0.3,
                    label=labels[i], linewidth=2)

        # 计算误差放大因子 (标准差之比)
        std_input = sigma_soc / mu_soc
        std_output = np.std(tte_samples) / tte_ref
        amplification = std_output / std_input

        # 标注
        peak_x = 0
        peak_y = plt.gca().get_ylim()[1]
        plt.text(5, 0.1 + i * 0.05, f"{labels[i]}: Error Amp = {amplification:.2f}x",
                 color=colors[i], fontweight='bold')

    plt.title('Monte Carlo Analysis: Error Propagation from Initial State', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage Deviation in Predicted TTE (%)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend(loc='upper left')
    plt.xlim(-15, 15)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_monte_carlo_error()
