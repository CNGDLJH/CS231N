import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 设置参数 --------------------------
alpha0 = 0.1       # 初始学习率
T = 100            # 总训练轮次
gamma = 0.5        # Step调度的衰减系数（γ）
s = 20             # Step调度的衰减步长（s）
epsilon = 1e-8     # Inverse sqrt调度的防除零小常数

# 生成训练轮次数组（t从0到T-1）
t = np.arange(0, T)

# -------------------------- 2. 计算各调度的学习率 --------------------------
# Step调度：αt = α0 × γ^(t/s)
alpha_step = alpha0 * (gamma ** (t / s))

# Cosine调度：αt = 0.5×α0×(1 + cos(tπ/T))
alpha_cosine = 0.5 * alpha0 * (1 + np.cos(t * np.pi / T))

# Linear调度：αt = α0×(1 - t/T)
alpha_linear = alpha0 * (1 - t / T)

# Inverse sqrt调度（修正版，避免分母为0）：αt = α0 / sqrt(t + ε)
alpha_inverse_sqrt = alpha0 / (np.sqrt(t + epsilon))

# -------------------------- 3. 绘制图像 --------------------------
plt.figure(figsize=(12, 7))

# 绘制各调度曲线
plt.plot(t, alpha_step, label='Step Scheduler', marker='.', linestyle='-', linewidth=2)
plt.plot(t, alpha_cosine, label='Cosine Scheduler', marker='.', linestyle='-', linewidth=2)
plt.plot(t, alpha_linear, label='Linear Scheduler', marker='.', linestyle='-', linewidth=2)
plt.plot(t, alpha_inverse_sqrt, label='Inverse sqrt Scheduler', marker='.', linestyle='-', linewidth=2)

# 图像美化
plt.title('Learning Rate Schedulers Comparison', fontsize=14)
plt.xlabel('Training Epoch (t)', fontsize=12)
plt.ylabel('Learning Rate (αt)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, T-1)
plt.ylim(0, alpha0 + 0.01)  # y轴范围适配初始学习率

# 显示图像
plt.show()