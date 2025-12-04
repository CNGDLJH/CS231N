import numpy as np
import matplotlib.pyplot as plt

# 设置全局样式（适配论文/报告）
plt.rcParams['font.sans-serif'] = ['Arial']  # 避免中文乱码
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线条宽度
plt.rcParams['xtick.direction'] = 'in'  # x轴刻度向内
plt.rcParams['ytick.direction'] = 'in'  # y轴刻度向内

# 超参数设置（实战常用值）
initial_lr = 0.1  # 初始学习率
total_epochs = 100  # 总训练轮次
epochs = np.arange(1, total_epochs + 1)  #  epoch序列（1~100）

# 1. Step调度（ResNet风格：30、60、90 epoch时×0.1）
def step_lr(epoch, initial_lr, decay_epochs=[30, 60, 90], decay_factor=0.1):
    lr = initial_lr
    for decay_epoch in decay_epochs:
        if epoch >= decay_epoch:
            lr *= decay_factor
        else:
            break
    return lr
step_lrs = [step_lr(epoch, initial_lr) for epoch in epochs]

# 2. Cosine调度（后期缓慢下降）
cosine_lrs = 0.5 * initial_lr * (1 + np.cos(np.pi * epochs / total_epochs))

# 3. Linear调度（线性衰减至0）
linear_lrs = initial_lr * (1 - epochs / total_epochs)

# 4. Inverse sqrt调度（前期下降快，后期稳定）
inverse_sqrt_lrs = initial_lr / np.sqrt(epochs)  # 避免t=0，用epoch（1~100）

# 绘制图像
plt.figure(figsize=(10, 6))  # 图大小（适配Word/论文）

# 绘制四条调度曲线（不同颜色+线型区分）
plt.plot(epochs, step_lrs, label='Step Schedule', color='#1f77b4', linewidth=2.5, linestyle='-', marker='o', markersize=3, markevery=5)
plt.plot(epochs, cosine_lrs, label='Cosine Schedule', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s', markersize=3, markevery=5)
plt.plot(epochs, linear_lrs, label='Linear Schedule', color='#2ca02c', linewidth=2.5, linestyle='-.', marker='^', markersize=3, markevery=5)
plt.plot(epochs, inverse_sqrt_lrs, label='Inverse Sqrt Schedule', color='#d62728', linewidth=2.5, linestyle=':', marker='d', markersize=3, markevery=5)

# 设置坐标轴和标签
plt.xlabel('Training Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Learning Rate', fontsize=14, fontweight='bold')
plt.title('Comparison of Learning Rate Scheduling Strategies', fontsize=16, fontweight='bold', pad=20)

# 调整刻度
plt.xticks(np.arange(0, total_epochs + 1, 10), fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(bottom=0)  # y轴从0开始，避免截断

# 添加图例（位置在右上角，不遮挡曲线）
plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, framealpha=0.9)

# 添加网格（辅助阅读）
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 紧凑布局（避免标签被截断）
plt.tight_layout()

# 保存图像（高分辨率，支持PNG/SVG）
plt.savefig('learning_rate_schedules.png', dpi=300, bbox_inches='tight')
plt.savefig('learning_rate_schedules.svg', format='svg', bbox_inches='tight')

# 显示图像
plt.show()