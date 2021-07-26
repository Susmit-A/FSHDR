import numpy as np
import tensorflow as tf

def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(bytes("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n", 'UTF-8'))
        f.write(bytes("-Y %d +X %d\n" % (image.shape[0], image.shape[1]), 'UTF-8'))
        brightest = np.max(image, axis=2)

        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)

        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)

def tonemap(im):
    im = tf.clip_by_value(im, 1.0e-10, 1.0)
    return tf.math.log(1.0 + 5000.0 * im) / tf.math.log(1.0 + 5000.0)

def ldr_to_hdr(im, exp_bias, gamma=2.2):
    im = np.clip(im, 0.0, 1.0)
    t = 2 ** exp_bias
    im_out = im ** gamma
    im_out = im_out / t
    im_out = np.clip(im_out, 0.0, 1.0)
    return im_out
    
def hdr_to_ldr(im, exp_bias, gamma=2.2):
    im = tf.clip_by_value(im, 0.0, 1.0)
    t = 2 ** exp_bias
    im_out = im * t
    im_out = im_out ** (1.0 / gamma)
    im_out = tf.clip_by_value(im_out, 0.0, 1.0)
    return im_out
    
'''
from math import ceil, log, pow
from scipy.interpolate import interp1d
from scipy.io import loadmat
crf = loadmat('matlab_liu_code/BaslerCRF.mat')['BaslerCRF']

def ldr_to_hdr2(img, exp):
    etime = pow(2, exp)
    inp = (img * (pow(2,16) - 1)).astype(np.uint16)
    (h, w, c) = inp.shape
    inp = np.reshape(inp, (h*w, c))
    
    out = np.zeros((h*w, c), np.float32)
    for i in range(3):
        out[:,i] = np.take(crf[:, i], inp[:, i])
    return np.clip((np.reshape(out, (h, w, c)) / etime), 0.0, 1.0).astype(np.float32)

def hdr_to_ldr2(img, exp):
    etime = pow(2, exp)
    inp = img * etime
    out = inp
    
    for i in range(3):
        func = interp1d(x=crf[:, i], y=range(65536), kind='nearest', fill_value='extrapolate')
        out[:, :, i] = func(inp[:, :, i])
    return np.clip(out / (pow(2, 16) - 1), 0.0, 1.0).astype(np.float32)
'''