import argparse, os, glob, cv2, sys
import numpy as np
print(" ".join(sys.argv))

parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str, default='BridgeNet')
parser.add_argument('--weights_loc',    type=str, default=None)
parser.add_argument('--model_name',     type=str, default=None)

parser.add_argument('--dataset',        type=str, default='SIG17')           #['SIG17' or 'ICCP19']
parser.add_argument('--image_type',     type=str, default='flow_corrected')  #['normal' or 'flow_corrected']
parser.add_argument('--gpu_num',        type=str, default='0')
parser.add_argument('--val_downsample', type=int, default=1)
parser.add_argument('--rtx_mixed_precision', action='store_true')

args = parser.parse_args()

if args.model not in ['BridgeNet', 'AHDR', 'WE', 'Resnet']:
    print("Unknown Model. Exiting.")
    exit()
else:
    print("Using {} model".format(args.model))
if args.dataset not in ['SIG17', 'ICCP19']:
    print("Unknown Dataset. Exiting.")
    exit()
else:
    print("Using {} dataset".format(args.dataset))
if args.image_type not in ['normal', 'flow_corrected']:
    print("Unknown Image Type. Exiting.")
    exit()
else:
    print("Using {} images".format(args.image_type))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
if args.rtx_mixed_precision:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

from dataloader import *
from models import *
from losses import *
from utils import *


model_name = args.model if args.model_name is None else args.model_name
model = models[args.model](name=model_name)
model.load_weights(args.weights_loc)
print("Loading model weights from ", args.weights_loc)

losses = [MSE_TM]
metrics = [PSNR_L, PSNR_T]

init_sequences_S1(dataset=args.dataset, image_type=args.image_type)
init_validation(dataset=args.dataset, image_type=args.image_type)
val_loader = Validation()

folder = 'val_' + args.dataset
print("\nValidation")
progbar = tf.keras.utils.Progbar(len(val_loader))
step = 1
if not os.path.exists(os.path.join('results', model_name)):
    os.mkdir(os.path.join('results', model_name))
elif os.path.exists(os.path.join('results', model_name, folder)):
    os.rmdir(os.path.join('results', model_name, folder))
os.mkdir(os.path.join('results', model_name, folder))

for i in range(len(val_loader)):
    loss_vals = []
    metric_vals = []
    os.mkdir(os.path.join('results', model_name, folder, str(i)))

    X, Y, exp = val_loader[i]
    if args.val_downsample > 1:
        inp = tf.image.resize(X, (X.shape[1] // args.val_downsample, X.shape[2] // args.val_downsample))
        Y = tf.image.resize(Y, (Y.shape[1] // args.val_downsample, Y.shape[2] // args.val_downsample))
    else:
        inp = X
    pred = model.predict(inp)
    
    radiance_writer(os.path.join('results', model_name, folder, str(i), str(i) + '.hdr'), np.squeeze(pred, axis=0))
    radiance_writer(os.path.join('results', model_name, folder, str(i), str(i) + '_gt.hdr'), np.squeeze(Y, axis=0))
    for l in losses:
        _loss = tf.reduce_mean(l(Y, pred))
        loss_vals.append((l.__name__.lower(), tf.reduce_mean(_loss)))
    for m in metrics:
        _metric = tf.reduce_mean(m(Y, pred))
        metric_vals.append((m.__name__.lower(), tf.reduce_mean(_metric)))
    progbar.update(step, loss_vals + metric_vals)
    step += 1
