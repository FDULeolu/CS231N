from .layers import *
from .fast_layers import *
import numpy as np


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_params):
    """Convenience layer that performs an affine transform with a batch norm followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Weight for the batch norm layer
    - bn_params: Parameters used in batch normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cahce: Object to give to the backward pass, (fc_cache, bn_cache, relu_cache)
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_a, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
    out, relu_cache = relu_forward(bn_a)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_batchnorm_relu_backward(dout, cache):
    """Backward pass for the affine-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx_bn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dx_bn, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_layernorm_relu_forward(x, w, b, gamma, beta, ln_params):
    """Convenience layer that performs an affine transform with a layer norm followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Weight for the batch norm layer
    - ln_params: Parameters used in layer normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cahce: Object to give to the backward pass, (fc_cache, ln_cache, relu_cache)
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_a, ln_cache = layernorm_forward(a, gamma, beta, ln_params)
    out, relu_cache = relu_forward(bn_a)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache

def affine_layernorm_relu_backward(dout, cache):
    """Backward pass for the affine-layernorm-relu convenience layer.
    """
    fc_cache, ln_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx_bn, dgamma, dbeta = layernorm_backward(da, ln_cache)
    dx, dw, db = affine_backward(dx_bn, fc_cache)
    return dx, dw, db, dgamma, dbeta

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****Tools for conv layer*****

def img2col(x, HH, WW, stride, pad):
    """Turn the input image x into a vector for conv layer
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride

    Input:
      - x: The shape of x is (N, C, H, W)
      - HH: Height of conv kernel
      - WW: Width of conv kernel
      - stride: Stride of conv layer
      - pad: Pad of conv layer
    
    Return:
      - col: The shape of col is (N * H' * W', C * HH * WW)
    """

    N, C, H, W = x.shape

    # Padding
    pad_width = [(0, 0), (0, 0), (pad, pad), (pad, pad)]
    x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

    H1 = (H + 2 * pad - HH) // stride + 1
    W1 = (W + 2 * pad - WW) // stride + 1

    col = np.zeros((N, C, H1, W1, HH, WW))
    for i in range(H1):
        for j in range(W1):
            col[:, :, i, j, :, :] = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
    col = col.transpose(0, 2, 3, 1, 4, 5).reshape(N * H1 * W1, -1)

    return col

def col2img(X, HH, WW, H1, W1, stride, pad):
    N = X.shape[0] // (H1 * W1)
    C = X.shape[1] // (HH * WW)
    H = (H1 - 1) * stride + HH - 2 * pad
    W = (W1 - 1) * stride + WW - 2 * pad

    X = X.reshape(N, H1, W1, C, HH, WW).transpose(0, 3, 1, 2, 4, 5)
    x = np.zeros((N, C, H+2*pad, W+2*pad))
    for i in range(H1):
        for j in range(W1):
            x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += X[:, :, i, j, :, :]
    x = x[:, :, pad:-pad, pad:-pad]
    return x


# *****Tools for conv layer*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
