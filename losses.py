import tensorflow as tf
from utils import tonemap, hdr_to_ldr

def MSE(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)
    
def MSE_TM(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return MSE(tonemap(y_true), tonemap(y_pred))
    
def MAE(y_true, y_pred):
    return tf.keras.losses.MAE(y_true, y_pred)

def MAE_TM(y_true, y_pred):
    return tf.keras.losses.MAE(tonemap(y_true), tonemap(y_pred))
 
def ME_SCL(me, y_pred, exp_me):
    me_pred = hdr_to_ldr(y_pred, exp_me)
    return MSE(me, me_pred)

def PSNR_L(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def PSNR_T(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.psnr(tonemap(y_true), tonemap(y_pred), max_val=1.0)