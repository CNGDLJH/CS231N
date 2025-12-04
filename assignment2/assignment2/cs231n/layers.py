from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer."""
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    out = x_flat.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer."""
    x, w, b = cache
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    
    db = np.sum(dout, axis=0)
    dw = x_flat.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs)."""
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs)."""
    x = cache
    dx = dout * (x > 0)
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification."""
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization."""
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    
    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    
    out, cache = None, None
    
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        x_centered = x - sample_mean
        std = np.sqrt(sample_var + eps)
        x_norm = x_centered / std
        out = gamma * x_norm + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = (x, gamma, beta, x_centered, std, sample_var)
        
    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    
    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization."""
    x, gamma, beta, x_centered, std, sample_var = cache
    N = x.shape[0]
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * (x_centered / std), axis=0)
    
    dx_norm = dout * gamma
    dx_centered = dx_norm / std
    dvar = np.sum(dx_norm * x_centered * (-0.5) * (sample_var + 1e-5)**(-1.5), axis=0)
    dmean = np.sum(dx_centered * (-1), axis=0) + dvar * np.mean(-2 * x_centered, axis=0)
    
    dx = dx_centered + (dvar * 2 * x_centered) / N + dmean / N
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization."""
    x, gamma, beta, x_centered, std, sample_var = cache
    N = x.shape[0]
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * (x_centered / std), axis=0)
    
    dx_norm = dout * gamma
    dx = (1.0 / N) * (1.0 / std) * (N * dx_norm - np.sum(dx_norm, axis=0) - (x_centered / sample_var) * np.sum(dx_norm * x_centered, axis=0))
    
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization."""
    eps = ln_param.get("eps", 1e-5)
    N, D = x.shape
    
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    x_centered = x - mean
    x_norm = x_centered / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    
    cache = (x, gamma, beta, x_centered, x_norm, var, eps)
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization."""
    x, gamma, beta, x_centered, x_norm, var, eps = cache
    N, D = x.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    dx_norm = dout * gamma
    dx_centered = dx_norm / np.sqrt(var + eps)
    dvar = np.sum(dx_norm * x_centered * (-0.5) * (var + eps)**(-1.5), axis=1, keepdims=True)
    dmean = np.sum(dx_centered * (-1), axis=1, keepdims=True) + dvar * np.mean(-2 * x_centered, axis=1, keepdims=True)
    
    dx = dx_centered + (dvar * 2 * x_centered) / D + dmean / D
    
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout."""
    p, mode = dropout_param["p"], dropout_param["mode"]
    
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])
    
    mask = None
    out = None
    
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    
    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout."""
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    
    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer."""
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    out = np.zeros((N, F, H_out, W_out))
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
    
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer."""
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_out = dout.shape[2]
    W_out = dout.shape[3]
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    db = np.sum(dout, axis=(0, 2, 3))
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]
                    dw[f] += x_slice * dout[n, f, i, j]
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]
    
    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer."""
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1
    
    out = np.zeros((N, C, H_out, W_out))
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    x_slice = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(x_slice)
    
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer."""
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_out = dout.shape[2]
    W_out = dout.shape[3]
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    x_slice = x[n, c, h_start:h_end, w_start:w_end]
                    mask = (x_slice == np.max(x_slice))
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]
    
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization."""
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization."""
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization."""
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape
    
    assert C % G == 0, "C must be divisible by G"
    group_size = C // G
    
    x_reshaped = x.reshape(N, G, group_size, H, W)
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    
    x_centered = x_reshaped - mean
    x_norm = x_centered / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    
    out = gamma * x_norm + beta
    cache = (x, G, gamma, beta, x_reshaped, x_centered, var, eps)
    
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization."""
    x, G, gamma, beta, x_reshaped, x_centered, var, eps = cache
    N, C, H, W = x.shape
    group_size = C // G
    
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * (x_centered.reshape(N, C, H, W)), axis=(0, 2, 3), keepdims=True)
    
    dx_norm = dout * gamma
    dx_norm_reshaped = dx_norm.reshape(N, G, group_size, H, W)
    
    dx_centered = dx_norm_reshaped / np.sqrt(var + eps)
    dvar = np.sum(dx_norm_reshaped * x_centered * (-0.5) * (var + eps)**(-1.5), axis=(2, 3, 4), keepdims=True)
    dmean = np.sum(dx_centered * (-1), axis=(2, 3, 4), keepdims=True) + dvar * np.mean(-2 * x_centered, axis=(2, 3, 4), keepdims=True)
    
    dx_reshaped = dx_centered + (dvar * 2 * x_centered) / (group_size * H * W) + dmean / (group_size * H * W)
    dx = dx_reshaped.reshape(N, C, H, W)
    
    return dx, dgamma, dbeta