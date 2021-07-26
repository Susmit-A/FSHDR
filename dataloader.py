import glob, cv2, os
import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from utils import *


#Please ensure dataset paths are correct
paths = {'SIG17': 'dataset/SIG17', 'ICCP19' : 'dataset/ICCP19'}

train_dynamic_paths = OrderedDict()
train_dynamic_exposures = OrderedDict()
train_dynamic_hdr_paths = OrderedDict()

train_static_paths = OrderedDict()
train_static_exposures = OrderedDict()
train_static_hdr_paths = OrderedDict()

val_paths = OrderedDict()
val_exposures = OrderedDict()
val_hdr_paths = OrderedDict()


def init_sequences_S2(synthetic_data_loc, dataset='SIG17', image_type='flow_corrected',
                      L_keys=None, P_keys=None):
    if L_keys is not None:
        orig_dst_path = paths[dataset]
        for img_path in sorted(glob.glob(orig_dst_path + '/train/[0-9]*'), key=lambda x: int(x.strip().split('/')[-1])):
            img = img_path.strip().split('/')[-1]
            if img in L_keys:
                if image_type == 'flow_corrected':
                    train_dynamic_paths[img_path] = [img_path + '/liu_flow_corrected/le.tif',
                                                     img_path + '/liu_flow_corrected/me.tif',
                                                     img_path + '/liu_flow_corrected/he.tif']
                else:
                    train_dynamic_paths[img_path] = [img_path + '/dynamic/le.tif',
                                                     img_path + '/dynamic/me.tif',
                                                     img_path + '/dynamic/he.tif']
                train_dynamic_exposures[img_path] = [float(line.strip()) for line in open(img_path + '/input_exp.txt')]
                train_dynamic_hdr_paths[img_path] = img_path + '/hdr_gt.hdr'
    
    if P_keys is not None:
        synth_dst_path = synthetic_data_loc
        for img_path in sorted(glob.glob(synth_dst_path + '/[0-9]*'), key=lambda x: int(x.strip().split('/')[-1])):
            img = img_path.strip().split('/')[-1]
            if img in P_keys:
                train_dynamic_paths[img_path] = [img_path + '/le_synthetic.tif',
                                                 img_path + '/me_synthetic.tif',
                                                 img_path + '/he_synthetic.tif']
                train_dynamic_exposures[img_path] = [float(line.strip()) for line in open(img_path + '/input_exp.txt')]
                train_dynamic_hdr_paths[img_path] = img_path + '/synthetic.hdr'


def init_sequences_S1(dataset='SIG17', image_type='flow_corrected'):
    dst_path = paths[dataset]
    for img_path in sorted(glob.glob(dst_path + '/train/[0-9]*'), key=lambda x: int(x.strip().split('/')[-1])):
        img = img_path.strip().split('/')[-1]
        if image_type == 'flow_corrected':
            train_dynamic_paths[img_path] = [img_path + '/liu_flow_corrected/le.tif',
                                             img_path + '/liu_flow_corrected/me.tif',
                                             img_path + '/liu_flow_corrected/he.tif']
        else:
            train_dynamic_paths[img_path] = [img_path + '/dynamic/le.tif',
                                             img_path + '/dynamic/me.tif',
                                             img_path + '/dynamic/he.tif']
        train_dynamic_exposures[img_path] = [float(line.strip()) for line in open(img_path + '/input_exp.txt')]
        train_dynamic_hdr_paths[img_path] = img_path + '/hdr_gt.hdr'
        
        train_static_paths[img_path] = [img_path + '/static/le.tif',
                                        img_path + '/static/me.tif',
                                        img_path + '/static/he.tif']
        train_static_exposures[img_path] = [float(line.strip()) for line in open(img_path + '/input_exp.txt')]
        train_static_hdr_paths[img_path] = img_path + '/hdr_gt.hdr'
    

def init_validation(dataset='SIG17', image_type='flow_corrected'):
    dst_path = paths[dataset]
    for img_path in sorted(glob.glob(dst_path + '/val/[0-9]*'), key=lambda x: int(x.strip().split('/')[-1])):
        if image_type == 'flow_corrected':
            val_paths[img_path] = [img_path + '/liu_flow_corrected/le.tif',
                                   img_path + '/liu_flow_corrected/me.tif',
                                   img_path + '/liu_flow_corrected/he.tif']
        else:
            val_paths[img_path] = [img_path + '/dynamic/le.tif',
                                   img_path + '/dynamic/me.tif',
                                   img_path + '/dynamic/he.tif']
        val_exposures[img_path] = [float(line.strip()) for line in open(img_path + '/input_exp.txt')]
        val_hdr_paths[img_path] = img_path + '/hdr_gt.hdr'


def unique_exposures(dataset='SIG17', static=False):
    exp_dict = {}
    dst_path = paths[dataset]
    for img_path in glob.glob(dst_path + '/train/[0-9]*'):
        img = img_path.strip().split('/')[-1]
        #if static and dataset!='ICCP19' and img in ['051', '048']:
        #    continue
        exp = tuple([float(line.strip()) for line in open(img_path + '/input_exp.txt')])
        if exp not in exp_dict.keys():
            exp_dict[exp] = [img_path]
        else:
            exp_dict[exp].append(img_path)
    return exp_dict


def random_flips(le, me, he, hdr):
    ch = np.random.choice([0, 1, 2, 3])
    if ch == 0:
        le = tf.image.flip_up_down(le)
        me = tf.image.flip_up_down(me)
        he = tf.image.flip_up_down(he)
        hdr = tf.image.flip_up_down(hdr)
    elif ch == 1:
        le = tf.image.flip_left_right(le)
        me = tf.image.flip_left_right(me)
        he = tf.image.flip_left_right(he)
        hdr = tf.image.flip_left_right(hdr)
    elif ch == 2:
        le = tf.image.flip_left_right(tf.image.flip_up_down(le))
        me = tf.image.flip_left_right(tf.image.flip_up_down(me))
        he = tf.image.flip_left_right(tf.image.flip_up_down(he))
        hdr = tf.image.flip_left_right(tf.image.flip_up_down(hdr))
    return le, me, he, hdr


class Training_Loader(Sequence):
    def __init__(self, batch_size, keys=None, patch_size=64, static=False, random_flip=False):
        self.static = static
        if self.static:
            self.train_keys = list(train_static_paths.keys()) if keys is None else keys
        else:
            self.train_keys = list(train_dynamic_paths.keys()) if keys is None else keys
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.flips = random_flip
        self.shuffle()

    def __len__(self):
        return int(len(self.train_keys))

    def shuffle(self):
        np.random.shuffle(self.train_keys)
        
    def __getitem__(self, index):
        keys = self.train_keys
        if self.static:
            img_paths = train_static_paths
            exposures = train_static_exposures
            hdr_paths = train_static_hdr_paths
        else:
            img_paths = train_dynamic_paths
            exposures = train_dynamic_exposures
            hdr_paths = train_dynamic_hdr_paths
        
        batch_X_img = []
        batch_Y = []
        index = np.random.randint(0, self.__len__())
        img = keys[index]

        le, me, he = img_paths[img]
        le = cv2.imread(le, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        me = cv2.imread(me, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        he = cv2.imread(he, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        hdr = cv2.imread(hdr_paths[img], cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        img_exposures = exposures[img]
        
        patch_size = self.patch_size
        for b in range(self.batch_size):
            coord_x = np.random.randint(0, hdr.shape[0] - patch_size)
            coord_y = np.random.randint(0, hdr.shape[1] - patch_size)

            le_patch = le[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
            me_patch = me[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
            he_patch = he[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
            hdr_patch = hdr[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
            if self.flips:
                le_patch, me_patch, he_patch, hdr_patch = random_flips(le_patch, me_patch, he_patch, hdr_patch)
            
            le_hdr_patch = ldr_to_hdr(le_patch, img_exposures[0])
            me_hdr_patch = ldr_to_hdr(me_patch, img_exposures[1])
            he_hdr_patch = ldr_to_hdr(he_patch, img_exposures[2])
            
            x_imgs = [np.concatenate([le_patch, le_hdr_patch], axis=-1),
                      np.concatenate([me_patch, me_hdr_patch], axis=-1),
                      np.concatenate([he_patch, he_hdr_patch], axis=-1)]
            batch_X_img.append(np.concatenate(x_imgs, axis=-1))
            batch_Y.append(hdr_patch)

        ret = [np.stack(batch_X_img, axis=0).astype(np.float32),
               np.stack(batch_Y, axis=0).astype(np.float32), exposures[img]]
        return ret


class Validation(Sequence):
    def __init__(self):
        self.val_keys = list(val_paths.keys())

    def __len__(self):
        return int(len(self.val_keys))
        
    def __getitem__(self, index):
        keys = self.val_keys
        img_paths = val_paths
        exposures = val_exposures
        hdr_paths = val_hdr_paths
        batch_X_img = []
        batch_Y = []
        
        img = keys[index]
        le, me, he = img_paths[img]
        le = cv2.imread(le, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        me = cv2.imread(me, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        he = cv2.imread(he, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        hdr = cv2.imread(hdr_paths[img], cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        img_exposures = exposures[img]

        le_hdr = ldr_to_hdr(le, img_exposures[0])
        me_hdr = ldr_to_hdr(me, img_exposures[1])
        he_hdr = ldr_to_hdr(he, img_exposures[2])

        batch_X_img.append(np.concatenate([np.concatenate([le, le_hdr], axis=-1),
                                           np.concatenate([me, me_hdr], axis=-1),
                                           np.concatenate([he, he_hdr], axis=-1)], axis=-1))
        batch_Y.append(hdr)
        ret = [np.stack(batch_X_img, axis=0).astype(np.float32),
               np.stack(batch_Y, axis=0).astype(np.float32), exposures[img]]
        return ret


class Sequenced_TrainLoader(Sequence):
    def __init__(self):
        self.train_keys = list(train_dynamic_paths.keys())

    def __len__(self):
        return int(len(self.train_keys))
        
    def __getitem__(self, index):
        keys = self.train_keys
        img_paths = train_dynamic_paths
        exposures = train_dynamic_exposures
        hdr_paths = train_dynamic_hdr_paths
        batch_X_img = []
        batch_Y = []
        
        img = keys[index]
        le, me, he = img_paths[img]
        le = cv2.imread(le, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        me = cv2.imread(me, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        he = cv2.imread(he, cv2.IMREAD_UNCHANGED)[:, :, ::-1] / 65535.0
        hdr = cv2.imread(hdr_paths[img], cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        img_exposures = exposures[img]

        le_hdr = ldr_to_hdr(le, img_exposures[0])
        me_hdr = ldr_to_hdr(me, img_exposures[1])
        he_hdr = ldr_to_hdr(he, img_exposures[2])

        batch_X_img.append(np.concatenate([np.concatenate([le, le_hdr], axis=-1),
                                           np.concatenate([me, me_hdr], axis=-1),
                                           np.concatenate([he, he_hdr], axis=-1)], axis=-1))
        batch_Y.append(hdr)
        ret = [np.stack(batch_X_img, axis=0).astype(np.float32),
               np.stack(batch_Y, axis=0).astype(np.float32), exposures[img]]
        return ret
