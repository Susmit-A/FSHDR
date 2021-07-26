import argparse, sys, os, cv2
import numpy as np
print(" ".join(sys.argv))
parser = argparse.ArgumentParser()

parser.add_argument('--model',                  type=str, default='BridgeNet')       #['BridgeNet' or 'AHDR', 'WE', 'Resnet']
parser.add_argument('--resume_weights_loc',     type=str, default=None)
parser.add_argument('--starting_epoch',         type=int, default=0)

parser.add_argument('--num_static',             type=int, default=5)                 # Use None for using all
parser.add_argument('--num_SCL_dynamic',        type=int, default=64)                # Use None for using all, except the supervised samples
parser.add_argument('--num_supervised_dynamic', type=int, default=5)
parser.add_argument('--dataset',                type=str, default='SIG17')           #['SIG17' or 'ICCP19']
parser.add_argument('--image_type',             type=str, default='flow_corrected')  #['normal' or 'flow_corrected']

parser.add_argument('--gpu_num',                type=str, default='0')
parser.add_argument('--rtx_mixed_precision',    action='store_true')

parser.add_argument('--model_name',             type=str, default=None)
parser.add_argument('--epochs',                 type=int, default=75)
parser.add_argument('--batch_size',             type=int, default=4)
parser.add_argument('--steps_per_batch',        type=int, default=5000)
parser.add_argument('--optimizer',              type=str, default='adam')
parser.add_argument('--momentum',               type=float, default=0.9)
parser.add_argument('--nesterov',               type=bool, default=True)
parser.add_argument('--amsgrad',                type=bool, default=False)
parser.add_argument('--start_lr',               type=float, default=1e-4)
parser.add_argument('--decay_steps',            type=int, default=10)
parser.add_argument('--decay_rate',             type=float, default=0.75)
parser.add_argument('--alpha',                  type=float, default=0.5)
parser.add_argument('--alpha_inc_steps',        type=int, default=10)
parser.add_argument('--alpha_inc_rate',         type=float, default=0.1)
parser.add_argument('--val_downsample',         type=int, default=2)
parser.add_argument('--save_val_results',       action='store_true')
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
if args.num_static < 0 or args.num_SCL_dynamic < 0 or args.num_supervised_dynamic < 0:
    print("Provide positive numbers for each training set. Exiting.")
    exit()

print("Using {} labeled dynamic samples, {} unlabeled dynamic samples, and {} labeled static samples for Stage 1 Training".format(
                args.num_supervised_dynamic, args.num_SCL_dynamic, args.num_static))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
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

if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists(os.path.join('results', model_name)):
    os.mkdir(os.path.join('results', model_name))
if not os.path.exists(os.path.join('results', model_name, 'stage1')):
    os.mkdir(os.path.join('results', model_name, 'stage1'))

if args.resume_weights_loc is not None:
    print("Loading model weights from ", args.resume_weights_loc)
    model.load_weights(args.resume_weights_loc)


epoch = args.starting_epoch
lr = args.start_lr
alpha = args.alpha
steps = args.steps_per_batch
def schd():
    return lr

optimizers = {'adam': tf.keras.optimizers.Adam(schd, amsgrad=args.amsgrad),
              'sgd': tf.keras.optimizers.SGD(schd, momentum=args.momentum, nesterov=args.nesterov),
              'nadam': tf.keras.optimizers.Nadam(schd),
              'adadelta': tf.keras.optimizers.Adadelta(schd),
              'rmsprop': tf.keras.optimizers.RMSprop(schd, momentum=args.momentum)}

opt = optimizers[args.optimizer]
if args.rtx_mixed_precision:
    opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

losses = [MSE_TM]
metrics = [PSNR_L, PSNR_T]


init_sequences_S1(dataset=args.dataset, image_type=args.image_type)
init_validation()

if args.num_supervised_dynamic != 0:
    if not os.path.exists(os.path.join('results', model_name, 'labeled_keys.txt')):
        choice_dict = unique_exposures(dataset=args.dataset)
        keys = []
        for i in range(args.num_supervised_dynamic):
            dict_key = list(choice_dict.keys())[i % len(choice_dict.keys())]
            sampled_key = np.random.choice(choice_dict[dict_key])
            choice_dict[dict_key].remove(sampled_key)
            if len(choice_dict[dict_key]) == 0:
                choice_dict.pop(dict_key)
            keys.append(sampled_key)
        print("Randomly selected labeled dynamic samples: ", keys)
        key_file = open(os.path.join('results', model_name, 'labeled_keys.txt'), 'w')
        for key in keys:
            key_file.write(key)
            key_file.write('\n')
        key_file.close()
    else:
        keys = [line.strip() for line in open(os.path.join('results', model_name, 'labeled_keys.txt'))]
        print("Loaded from file labeled dynamic samples: ", keys)
    L_loader = Training_Loader(batch_size=args.batch_size, keys=keys, patch_size=64, static=False)
    L_seq = tf.keras.utils.OrderedEnqueuer(L_loader, use_multiprocessing=False, shuffle=True)
    L_keys = [x.strip().split('/')[-1] for x in keys]
    _L_keys = keys
else:
    L_keys = []
    _L_keys = []

if args.num_static != 0:
    if os.path.exists(os.path.join('results', model_name, 'static_keys.txt')):
        keys = [line.strip() for line in open(os.path.join('results', model_name, 'static_keys.txt'))]
        print("Loaded from file static samples: ", keys)
    elif args.num_static is None:
        keys = train_static_paths.keys()
        print("Using all static samples.")
    else:
        choice_dict = unique_exposures(dataset=args.dataset, static=True)
        keys = []
        i = 0
        while i < args.num_static:
            dict_key = list(choice_dict.keys())[i % len(choice_dict.keys())]
            sampled_key = np.random.choice(choice_dict[dict_key])
            cur = sampled_key.strip().split('/')[-1]
            if cur in L_keys:
                continue
            choice_dict[dict_key].remove(sampled_key)
            if len(choice_dict[dict_key]) == 0:
                choice_dict.pop(dict_key)
            keys.append(sampled_key)
            i += 1
        key_file = open(os.path.join('results', model_name, 'static_keys.txt'), 'w')
        for key in keys:
            key_file.write(key)
            key_file.write('\n')
        key_file.close()
        print("Randomly selected static samples: ", keys)
    S_loader = Training_Loader(args.batch_size, keys=keys, patch_size=64, static=True)
    S_seq = tf.keras.utils.OrderedEnqueuer(S_loader, use_multiprocessing=False, shuffle=True)
    S_keys = [x.strip().split('/')[-1] for x in keys]
else:
    S_keys = []

if args.num_SCL_dynamic != 0:
    total = args.num_SCL_dynamic + len(S_keys) + len(L_keys)
    if os.path.exists(os.path.join('results', model_name, 'unlabeled_keys.txt')):
        keys = [line.strip() for line in open(os.path.join('results', model_name, 'unlabeled_keys.txt'))]
        print("Loaded from file unlabeled dynamic samples: ", keys)
    elif args.num_SCL_dynamic is None:
        keys = [x for x in train_dynamic_paths.keys() if x not in _L_keys]
        print("Using all samples except supervised, as unlabeled dynamic samples.")
    elif (total==74 and args.dataset=='SIG17') or (total==466 and args.dataset=='ICCP19'):
        if args.dataset == 'SIG17':
            all = ['{:03d}'.format(x) for x in range(1, 75)]
        else:
            all = ['{:03d}'.format(x) for x in range(1, 467)]
        all_but_L = [x for x in all if x not in L_keys]
        keys = [paths[args.dataset] + '/train/'+ x for x in all_but_L if x not in S_keys]
        print("Randomly selected unlabeled dynamic samples: ", keys)
    else:
        choice_dict = unique_exposures(dataset=args.dataset)
        keys = []
        i = 0
        print(choice_dict.keys())
        exit()
        while i < args.num_SCL_dynamic:
            dict_key = list(choice_dict.keys())[i % len(choice_dict.keys())]
            sampled_key = np.random.choice(choice_dict[dict_key])
            cur = sampled_key.strip().split('/')[-1]
            if cur in L_keys or cur in S_keys:
                print("looping")
                continue
            choice_dict[dict_key].remove(sampled_key)
            if len(choice_dict[dict_key]) == 0:
                choice_dict.pop(dict_key)
            keys.append(sampled_key)
            i += 1
        print("Randomly selected unlabeled dynamic samples: ", keys)
    key_file = open(os.path.join('results', model_name, 'unlabeled_keys.txt'), 'w')
    for key in keys:
        key_file.write(key)
        key_file.write('\n')
    key_file.close()
    U_loader = Training_Loader(batch_size=args.batch_size, keys=keys, patch_size=64, static=False)
    U_seq = tf.keras.utils.OrderedEnqueuer(U_loader, use_multiprocessing=False, shuffle=True)
    U_keys = [x.strip().split('/')[-1] for x in keys]
else:
    U_keys = []
    
val_loader = Validation()


def validate(epoch):
    print("\nValidation - Epoch ", epoch)
    progbar = tf.keras.utils.Progbar(len(val_loader))
    step = 1
    if not os.path.exists(os.path.join('results', model_name, 'stage1', str(epoch))):
        os.makedirs(os.path.join('results', model_name, 'stage1', str(epoch)))
    model.save_weights(os.path.join('results', model_name, 'stage1', str(epoch), model_name + '.tf'))

    for i in range(len(val_loader)):
        metric_vals = []
        loss_vals = []
        X, Y, exp = val_loader[i]
        
        if args.model == 'Resnet':
            div = 16
            X = X[:, :X.shape[1] - X.shape[1]%div, :X.shape[2] - X.shape[2]%div, :]
            Y = Y[:, :Y.shape[1] - Y.shape[1]%div, :Y.shape[2] - Y.shape[2]%div, :]
        if args.val_downsample > 1:
            inp = tf.image.resize(X, (X.shape[1] // args.val_downsample, X.shape[2] // args.val_downsample))
            Y = tf.image.resize(Y, (Y.shape[1] // args.val_downsample, Y.shape[2] // args.val_downsample))
        else:
            inp = X
        pred = model.predict(inp)
        
        for l in losses:
            _loss = tf.reduce_mean(l(Y, pred))
            loss_vals.append((l.__name__.lower(), _loss))    
        for m in metrics:
            _metric = tf.reduce_mean(m(Y, pred))
            metric_vals.append((m.__name__.lower(), tf.reduce_mean(_metric)))

        if args.save_val_results and epoch % 10 == 0:
            os.makedirs(os.path.join('results', model_name, 'stage1', str(epoch), str(i)))
            radiance_writer(os.path.join('results', model_name, 'stage1', str(epoch), str(i), str(i) + '.hdr'),
                            np.squeeze(pred, axis=0).astype(np.float32))
            radiance_writer(os.path.join('results', model_name, 'stage1', str(epoch), str(i), str(i) + '_gt.hdr'),
                            np.squeeze(Y, axis=0).astype(np.float32))
        progbar.update(step, loss_vals + metric_vals)
        step += 1


print("Stage 1 Training Begins")
if args.num_static > 0:
    S_seq.start(workers=4, max_queue_size=8)
    S_gen = S_seq.get()
if args.num_supervised_dynamic > 0:
    L_seq.start(workers=4, max_queue_size=8)
    L_gen = L_seq.get()
if args.num_SCL_dynamic > 0:
    U_seq.start(workers=4, max_queue_size=8)
    U_gen = U_seq.get()

while epoch < args.epochs:
    if (epoch + 1) % args.decay_steps == 0:
        lr = lr * args.decay_rate
    if (epoch + 1) % args.alpha_inc_steps == 0:
        alpha = alpha + args.alpha_inc_rate
    print("\nTraining - Epoch ", epoch, " | Learning rate = ", schd())
    step = 1
    progbar = tf.keras.utils.Progbar(steps)
    for i in range(steps):
        loss_vals = []
        metric_vals = []
        if args.num_static + args.num_supervised_dynamic > 0:
            if args.num_static > 0 and args.num_supervised_dynamic > 0:
                gen = np.random.choice([S_gen, L_gen])
            elif args.num_supervised_dynamic > 0:
                gen = L_gen
            else:
                gen = S_gen

            # X_sup, Y_sup, exp_sup = S_loader[0]
            # X_sup, Y_sup, exp_sup = L_gen[0]
            # exit()
            X_sup, Y_sup, exp_sup = next(gen)
            if args.num_SCL_dynamic > 0:
                X_SCL, Y_SCL, exp_SCL = next(U_gen)
                X = tf.concat([X_sup, X_SCL], axis=0)
                Y = tf.concat([Y_sup, Y_SCL], axis=0)
            else:
                X = X_sup
                Y = Y_sup
    
            with tf.GradientTape() as tape:
                loss = 0
                pred = model(X)
                for l in losses:
                    _loss = tf.reduce_mean(l(Y[:args.batch_size], pred[:args.batch_size]))
                    loss_vals.append((l.__name__.lower(), _loss))
                    loss += _loss
                if args.num_SCL_dynamic > 0:
                    SCL_pred = pred[args.batch_size:, :, :, :]
                    _loss = ME_SCL(X_SCL[:, :, :, 6:9], SCL_pred, exp_SCL[1])
                    loss_vals.append(('scl_loss', _loss))
                    loss += alpha * _loss
                
                if args.rtx_mixed_precision:
                    scaled_loss = opt.get_scaled_loss(loss)
                    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
                    grads = opt.get_unscaled_gradients(scaled_gradients)
                else:
                    grads = tape.gradient(loss, model.trainable_variables)

        else:
            X_SCL, Y_SCL, exp_SCL = next(U_gen)
            loss_vals = []
            metric_vals = []
            with tf.GradientTape() as tape:
                loss = 0
                pred = model(X_SCL)
                _loss = ME_SCL(X_SCL[:, :, :, 6:9], pred, exp_SCL[1])
                loss_vals.append(('scl_loss', _loss))
                loss += _loss
                    
                if args.rtx_mixed_precision:
                    scaled_loss = opt.get_scaled_loss(loss)
                    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
                    grads = opt.get_unscaled_gradients(scaled_gradients)
                else:
                    grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        for m in metrics:
            _metric = tf.reduce_mean(m(Y, pred))
            metric_vals.append((m.__name__.lower(), _metric))
        loss_vals = [(str(k), v.numpy()) for k, v in loss_vals]
        metric_vals = [(str(k), v.numpy()) for k, v in metric_vals]
        progbar.update(step, loss_vals + metric_vals)
        step += 1
    validate(epoch)
    epoch += 1
    
L_seq.stop()
U_seq.stop()
S_seq.stop()
