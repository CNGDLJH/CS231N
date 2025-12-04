import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示（避免标题/标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 1. 生成输入数据（覆盖激活函数关键区间：[-10, 10]，足够显示饱和特性）
x = np.linspace(-10, 10, 1000)  # 生成1000个均匀分布的点

# 2. 定义4个激活函数
def relu(x):
    """ReLU函数：max(0, x)"""
    return np.maximum(0, x)

def leaky_relu(x, negative_slope=0.01):
    """Leaky ReLU函数：max(αx, x)，α=0.01（常用值）"""
    return np.where(x >= 0, x, negative_slope * x)

def sigmoid(x):
    """Sigmoid函数：1/(1+e^(-x))"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh函数：(e^x - e^(-x))/(e^x + e^(-x))"""
    return np.tanh(x)  # 直接用numpy优化实现，比手动计算更高效

# 3. 计算输出值
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

# 4. 创建2x2子图布局（4个函数分屏显示，方便对比）
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('深度学习常用激活函数图像', fontsize=16, fontweight='bold')

# 绘制ReLU
axes[0, 0].plot(x, y_relu, color='#1f77b4', linewidth=2.5)
axes[0, 0].set_title('ReLU函数', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('σ(x) = max(0, x)', fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)  # 水平线y=0
axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)  # 垂直线x=0

# 绘制Leaky ReLU
axes[0, 1].plot(x, y_leaky_relu, color='#ff7f0e', linewidth=2.5)
axes[0, 1].set_title('Leaky ReLU函数', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('σ(x) = max(0.01x, x)', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)

# 绘制Sigmoid
axes[1, 0].plot(x, y_sigmoid, color='#2ca02c', linewidth=2.5)
axes[1, 0].set_title('Sigmoid函数', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('σ(x) = 1/(1+e^(-x))', fontsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
axes[1, 0].axhline(y=0.5, color='red', linestyle=':', alpha=0.8, label='σ(0)=0.5')  # 标注中点
axes[1, 0].legend()

# 绘制Tanh
axes[1, 1].plot(x, y_tanh, color='#d62728', linewidth=2.5)
axes[1, 1].set_title('Tanh函数', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('σ(x) = tanh(x)', fontsize=12)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
axes[1, 1].axhline(y=1, color='red', linestyle=':', alpha=0.8, label='σ(∞)=1')
axes[1, 1].axhline(y=-1, color='red', linestyle=':', alpha=0.8, label='σ(-∞)=-1')
axes[1, 1].legend()

# 调整子图间距，避免标签重叠
plt.tight_layout()

# 保存图像（可选，保存为高清PNG，可插入Word文档）
plt.savefig('激活函数图像.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()