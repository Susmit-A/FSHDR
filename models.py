from tensorflow.keras.layers import *
from layers import *

def create_BridgeNet(name=None, *args, **kwargs):
    inp = Input((None, None, 18))
    c = 64
    inp1 = inp[:, :, :, 0:6]
    inp2 = inp[:, :, :, 6:12]
    inp3 = inp[:, :, :, 12:18]

    x1 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(inp1)
    x2 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(inp2)
    x3 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(inp3)

    s1 = Concatenate()([x1, x2])
    s1 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(s1)
    s2 = Concatenate()([x2, x3])
    s2 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(s2)

    x1 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(x1)
    x2 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(x2)
    x3 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(x3)

    s1 = Concatenate()([x1, x2, s1])
    s1 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(s1)
    s2 = Concatenate()([x2, x3, s2])
    s2 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(s2)

    x1 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(x1)
    x2 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(x2)
    x3 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(x3)

    s1 = Concatenate()([x1, x2, s1])
    s1 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(s1)
    s2 = Concatenate()([x2, x3, s2])
    s2 = Conv2D(c, (3, 3), padding='same', activation=LeakyReLU(0.2))(s2)

    x = Concatenate()([x2, s1, s2])
    x1 = SDC(c, (3, 3))(x)
    x2 = Concatenate()([x, x1])
    x2 = SDC(c, (3, 3))(x2)
    x3 = Concatenate()([x, x1, x2])

    x = Conv2D(3, (3, 3), padding='same', activation=None)(x3)
    x = Activation('sigmoid', dtype=tf.float32)(x)
    if name is None:
        model = tf.keras.Model(inputs=[inp], outputs=[x])
    else:
        model = tf.keras.Model(inputs=[inp], outputs=[x], name=name)
    return model

def create_ResnetWu(name=None, *args, **kwargs):
    inp_imgs = Input((None, None, 18))
    c = 64
    le = inp_imgs[:, :, :, 0:6]
    me = inp_imgs[:, :, :, 6:12]
    he = inp_imgs[:, :, :, 12:18]
    
    le_ = LeakyReLU(0.2) (StridedConv(c, 5, 2)(le))
    le = BatchNormalization(momentum=0.9, epsilon=1e-5) (StridedConv(c * 2, 5, 2)(le_))
    me_ = LeakyReLU(0.2) (StridedConv(c, 5, 2)(me))
    me = BatchNormalization(momentum=0.9, epsilon=1e-5) (StridedConv(c * 2, 5, 2)(me_))
    he_ = LeakyReLU(0.2) (StridedConv(c, 5, 2)(he))
    he = BatchNormalization(momentum=0.9, epsilon=1e-5) (StridedConv(c * 2, 5, 2)(he_))
    
    x1 = LeakyReLU(0.2) (Concatenate()([le, me, he]))
    x2 = BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c * 4, (5, 5), padding='same', strides=2)(x1))
    x3 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x2))))
    x3 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x3)), x2])
    x4 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x3))))
    x4 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x4)), x3])
    x5 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x4))))
    x5 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x5)), x4])
    x6 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x5))))
    x6 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x6)), x5])
    x7 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x6))))
    x7 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x7)), x6])
    x8 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x7))))
    x8 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x8)), x7])
    x9 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x8))))
    x9 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x9)), x8])
    x10 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x9))))
    x10 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x10)), x9])
    x11 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (ReLU() (x10))))
    x11 = Add() ([BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2D(c*4, (3,3), padding='same') (x11)), x10])
    
    d1 = ReLU() (Concatenate() ([x11, x2]))
    d2 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2DTranspose(c*2, (5,5), strides=(2,2), padding='same') (d1)))
    d2 = Concatenate() ([d2, le, me, he])
    d3 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2DTranspose(c, (5,5), strides=(2,2), padding='same') (d2)))
    d3 = Concatenate() ([d3, le_, me_, he_])
    d4 = ReLU() (BatchNormalization(momentum=0.9, epsilon=1e-5) (Conv2DTranspose(c, (5,5), strides=(2,2), padding='same') (d3)))
    
    out = Conv2D(3, (1, 1), activation='sigmoid') (d4)
    if name is None:
        model = tf.keras.Model(inputs=[inp_imgs], outputs=[out])
    else:
        model = tf.keras.Model(inputs=[inp_imgs], outputs=[out], name=name)
    return model

def create_WEKalantari(name=None, *args, **kwargs):
    inp_imgs = Input((None, None, 18))
    
    x = Conv2D(100, (7, 7), padding='same', activation='relu')(inp_imgs)
    x = Conv2D(100, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    weight = Conv2D(9, (1, 1), activation='sigmoid')(x)
    
    le_hdr = inp_imgs[:, :, :, 3:6]
    me_hdr = inp_imgs[:, :, :, 9:12]
    he_hdr = inp_imgs[:, :, :, 15:18]
    
    le_w = weight[:, :, :, 0:3]
    me_w = weight[:, :, :, 3:6]
    he_w = weight[:, :, :, 6:9]
    
    le = Multiply()([le_hdr, le_w])
    me = Multiply()([me_hdr, me_w])
    he = Multiply()([he_hdr, he_w])
    
    hdr = Add()([le, me, he])
    denom = Add()([le_w, me_w, he_w])
    out = Divider()([hdr, denom])
    if name is None:
        model = tf.keras.Model(inputs=[inp_imgs], outputs=[out])
    else:
        model = tf.keras.Model(inputs=[inp_imgs], outputs=[out], name=name)
    return model

def create_AHDRYan(name=None, *args, **kwargs):
    inp_imgs = Input((None, None, 18))
    le = inp_imgs[:, :, :, 0:6]
    me = inp_imgs[:, :, :, 6:12]
    he = inp_imgs[:, :, :, 12:18]
    c = 64
    
    enc_inp = Input((None, None, 6))
    enc_x = LeakyReLU(0.01) (Conv2D(c, (3, 3), padding='same')(enc_inp))
    encoder = Model(inputs=[enc_inp], outputs=[enc_x])
    le = encoder(le)
    me = encoder(me)
    he = encoder(he)
    
    a1 = LeakyReLU(0.01) (Conv2D(c * 2, (3, 3), padding='same')(Concatenate()([le, me])))
    a1 = Conv2D(c, (3, 3), padding='same', activation='sigmoid')(a1)
    a2 = LeakyReLU(0.01) (Conv2D(c * 2, (3, 3), padding='same')(Concatenate()([he, me])))
    a2 = Conv2D(c, (3, 3), padding='same', activation='sigmoid')(a2)
    le_at = Multiply()([le, a1])
    he_at = Multiply()([he, a2])
    
    x1 = Conv2D(c, (3, 3), padding='same')(Concatenate()([le_at, me, he_at]))
    x2 = DRDB_Unit(int(c/2), (3, 3))(x1)
    x3 = DRDB_Unit(int(c/2), (3, 3))(x2)
    x4 = DRDB_Unit(int(c/2), (3, 3))(x3)
    x5 = Conv2D(c, (1, 1), padding='valid')(Concatenate()([x2, x3, x4]))
    x6 = Conv2D(c, (3, 3), padding='same')(x5)
    x7 = Conv2D(c, (3, 3), padding='same')(Add()([x6, me]))
    
    out = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x7)
    if name is None:
        model = tf.keras.Model(inputs=[inp_imgs], outputs=[out])
    else:
        model = tf.keras.Model(inputs=[inp_imgs], outputs=[out], name=name)
    return model

models = {
    'BridgeNet': create_BridgeNet,
    'AHDR': create_AHDRYan,
    'WE': create_WEKalantari,
    'Resnet': create_ResnetWu
}