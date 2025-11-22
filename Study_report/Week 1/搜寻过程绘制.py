import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Define the Loss Function and Gradient (Convex Function)
# ------------------------------------------------------
# Loss function: L(w1, w2) = 0.1*w1² + w2² (elliptical contours, simulating "valley" structure)
def loss_function(w1, w2):
    return 0.1 * w1**2 + w2**2

# Compute gradient (analytical solution)
def compute_gradient(w1, w2):
    grad_w1 = 0.2 * w1  # dL/dw1
    grad_w2 = 2 * w2    # dL/dw2
    return np.array([grad_w1, grad_w2])

# ------------------------------------------------------
# 2. Define Optimization Algorithms
# ------------------------------------------------------
def sgd_optimize(initial_w, lr, num_iter, batch_size):
    w = initial_w.copy()
    w_history = [w.copy()]
    np.random.seed(42)  # Fixed random seed for reproducibility
    
    for _ in range(num_iter):
        # Simulate mini-batch gradient noise (core of SGD: noisy gradient estimate)
        noise = np.random.normal(0, 0.1, size=2)
        grad = compute_gradient(w[0], w[1]) + noise
        
        # SGD update rule
        w = w - lr * grad
        w_history.append(w.copy())
    
    return np.array(w_history)

def sgd_momentum_optimize(initial_w, lr, num_iter, batch_size, rho=0.9):
    w = initial_w.copy()
    w_history = [w.copy()]
    vx = np.zeros_like(w)  # Momentum velocity
    np.random.seed(42)
    
    for _ in range(num_iter):
        noise = np.random.normal(0, 0.1, size=2)
        grad = compute_gradient(w[0], w[1]) + noise
        
        # Momentum update rule
        vx = rho * vx + ( 1 - rho ) * grad
        w = w - lr * vx
        w_history.append(w.copy())
    
    return np.array(w_history)

def rmsprop_optimize(initial_w, lr, num_iter, batch_size, decay_rate=0.99, eps=1e-7):
    w = initial_w.copy()
    w_history = [w.copy()]
    grad_squared = np.zeros_like(w)  # Accumulated gradient squared
    np.random.seed(42)
    
    for _ in range(num_iter):
        noise = np.random.normal(0, 0.1, size=2)
        grad = compute_gradient(w[0], w[1]) + noise
        
        # RMSProp update rule
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * grad**2
        w = w - lr * grad / (np.sqrt(grad_squared) + eps)
        w_history.append(w.copy())
    
    return np.array(w_history)

def adam_vanilla_optimize(initial_w, lr, num_iter, batch_size, beta1=0.9, beta2=0.999, eps=1e-7):
    w = initial_w.copy()
    w_history = [w.copy()]
    m = np.zeros_like(w)  # First moment (momentum)
    v = np.zeros_like(w)  # Second moment (RMSProp)
    np.random.seed(42)
    
    for _ in range(num_iter):
        noise = np.random.normal(0, 0.1, size=2)
        grad = compute_gradient(w[0], w[1]) + noise
        
        # Vanilla Adam update rule (NO bias correction)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        w = w - lr * m / (np.sqrt(v) + eps)
        w_history.append(w.copy())
    
    return np.array(w_history)

def adam_bias_correction_optimize(initial_w, lr, num_iter, batch_size, beta1=0.9, beta2=0.999, eps=1e-7):
    w = initial_w.copy()
    w_history = [w.copy()]
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    np.random.seed(42)
    
    for t in range(1, num_iter + 1):  # t starts from 1 for bias correction
        noise = np.random.normal(0, 0.1, size=2)
        grad = compute_gradient(w[0], w[1]) + noise
        
        # Adam with bias correction
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction terms
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        w_history.append(w.copy())
    
    return np.array(w_history)

# ------------------------------------------------------
# 3. Initialize Hyperparameters
# ------------------------------------------------------
initial_w = np.array([-5.0, 3.0])  # Common starting point for all algorithms
learning_rate = 0.1
num_iterations = 100
batch_size = 8

# Run optimizations
sgd_history = sgd_optimize(initial_w, learning_rate, num_iterations, batch_size)
sgd_momentum_history = sgd_momentum_optimize(initial_w, learning_rate, num_iterations, batch_size)
rmsprop_history = rmsprop_optimize(initial_w, learning_rate, num_iterations, batch_size)
adam_vanilla_history = adam_vanilla_optimize(initial_w, learning_rate, num_iterations, batch_size)
adam_bias_history = adam_bias_correction_optimize(initial_w, learning_rate, num_iterations, batch_size)

# ------------------------------------------------------
# 4. Plot Optimization Paths
# ------------------------------------------------------
# Create meshgrid for loss function contours
w1_range = np.linspace(-6, 6, 100)
w2_range = np.linspace(-4, 4, 100)
w1_grid, w2_grid = np.meshgrid(w1_range, w2_range)
loss_grid = loss_function(w1_grid, w2_grid)

# Plot
plt.figure(figsize=(12, 7))

# Draw loss function contours
contour = plt.contour(w1_grid, w2_grid, loss_grid, levels=20, cmap='viridis', alpha=0.7)
plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')  # Label contour values

# Plot paths of different algorithms
plt.plot(sgd_history[:, 0], sgd_history[:, 1], 'r-o', linewidth=1.5, markersize=3, label='SGD')
plt.plot(sgd_momentum_history[:, 0], sgd_momentum_history[:, 1], 'g-s', linewidth=1.5, markersize=3, label='SGD + Momentum')
plt.plot(rmsprop_history[:, 0], rmsprop_history[:, 1], 'b-^', linewidth=1.5, markersize=3, label='RMSProp')
plt.plot(adam_vanilla_history[:, 0], adam_vanilla_history[:, 1], 'm-*', linewidth=1.5, markersize=3, label='Adam (No Bias Correction)')
plt.plot(adam_bias_history[:, 0], adam_bias_history[:, 1], 'c-D', linewidth=1.5, markersize=3, label='Adam (With Bias Correction)')

# Mark key points
plt.scatter(initial_w[0], initial_w[1], color='black', s=100, label='Start Point', zorder=5)
plt.scatter(0, 0, color='red', s=150, marker='*', label='Global Minimum', zorder=10)

# Set labels and title (English only)
plt.xlabel('Parameter w1', fontsize=12)
plt.ylabel('Parameter w2', fontsize=12)
plt.title('Optimization Paths of Different Algorithms', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

# Show plot
plt.show()