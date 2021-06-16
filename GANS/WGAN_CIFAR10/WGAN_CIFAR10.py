# %%
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import os
import shutil
from tensorflow.python.keras.regularizers import L2
from WGAN_GP import WGAN_GP

# general functions

def load_data(batch_size):
    @ tf.function
    def parse_files(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32) / 127.5 - 1
        return image

    ds= tf.data.Dataset.list_files('C:/Users/alans/tensorflow_datasets/celeb_a/img_align_celeba/*.jpg', shuffle=False)
    ds = ds.shuffle(len(ds)).map(parse_files)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=batch_size)
    
    return ds

batch_size = 80
ds = load_data(batch_size)

# %%
def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.LeakyReLU(0.2)(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=True):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding='same', 
                    use_bias=use_bias, kernel_regularizer=L2(0.01))(x)
    return fx

def residual_block(x, filterNumb, isPooling=False):
    strides = 1
    shortcut = x
    bn_x = bn_relu(x)

    if isPooling:
        strides = (2,2)
        shortcut = conv(bn_x, filterNumb, kernel_size=1, strides=strides)

    fx = conv(bn_x, filterNumb, kernel_size=3, strides=strides)
    fx = bn_relu(fx)
    fx = conv(fx, filterNumb, kernel_size=3)
    out = layers.Add()([shortcut, fx]) # skip
    return out

def conv_transpose(model, kernels, strides, activation=None):
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(kernels, (3, 3), strides=strides, padding='same', use_bias=False))   

def generate_image(gan_model, seed=None, isShow=True, isSaveFile=False):
    grid_size = (2,2)
    w, h = grid_size
    img_count = w * h

    if seed is None:
        seed = tf.random.normal(shape=(4, gan_model.seed_size))

    fake_img_batch = gan_model.generator(test_seed, training=False)

    fig = plt.figure(figsize=grid_size, dpi=70)
    fig.set_figheight(4)
    fig.set_figwidth(4)
    for i in range(img_count):
        plt.subplot(w, h, i+1)
        img = tf.cast(fake_img_batch[i, :, :, :] * 127.5 + 127.5, dtype='uint8')
        plt.imshow(img)
        plt.axis('off')

    if isShow: 
        plt.show()
    if isSaveFile:
        if not os.path.exists('./Results'):
            os.makedirs('./Results')
        plt.savefig('./Results/{:06d}.png'.format(gan_model.dis_opt.iterations.numpy()))
        
# models
def create_generator(seed_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*3*128, use_bias=False, input_shape=(seed_size,)))
    model.add(layers.Reshape((4, 3, 128)))
    assert model.output_shape == (None, 4, 3, 128)

    model.add(layers.UpSampling2D((2,2)))
    assert model.output_shape == (None, 8, 6, 128)

    conv_transpose(model, 64, (2,2))
    assert model.output_shape == (None, 16, 12, 64)
    
    model.add(layers.UpSampling2D((2,2)))
    assert model.output_shape == (None, 32, 24, 64)

    conv_transpose(model, 32, (2,2))
    assert model.output_shape == (None, 64, 48, 32)

    conv_transpose(model, 16, (2,2))
    assert model.output_shape == (None, 128, 96, 16)

    conv_transpose(model, 3, (2,2))
    assert model.output_shape == (None, 256, 192, 3)

    model.add(layers.Activation('tanh'))
    model.add(layers.Cropping2D(((19,19),(7,7))))
    assert model.output_shape == (None, 218, 178, 3)
    model.summary()

    return model

def create_discriminator():
    inputs = layers.Input(shape=(218, 178, 3))

    hx = conv(inputs, 18, kernel_size=7, strides=2)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.3)(hx)

    hx = conv(hx, 32, kernel_size=5, strides=2)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.3)(hx)

    hx = conv(hx, 64, kernel_size=5, strides=2)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.3)(hx)

    # hx = conv(hx, 84, kernel_size=5, strides=2)
    # hx = layers.LeakyReLU(0.2)(hx)
    # hx = layers.Dropout(0.3)(hx)

    hx = layers.GlobalAveragePooling2D()(hx)
    hx = layers.Dense(128)(hx)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.2)(hx)

    outputs = layers.Dense(1)(hx)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model

def create_WGAN_GP(seed_size, gen_lr, dis_lr, dis_extra_steps):
    generator = create_generator(seed_size)
    discriminator = create_discriminator()
    gen_opt = tf.keras.optimizers.RMSprop(learning_rate=gen_lr)
    dis_opt = tf.keras.optimizers.RMSprop(learning_rate=dis_lr)
    gen_loss_fn = lambda fake_pred: -tf.reduce_mean(fake_pred)
    dis_loss_fn = lambda real_pred, fake_pred: tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
    gan_model = WGAN_GP(discriminator, generator, seed_size, dis_extra_steps=dis_extra_steps)
    gan_model.compile(gen_opt, dis_opt, gen_loss_fn, dis_loss_fn)
    return gan_model

epochs = 40
seed_size = 100
gen_lr = 0.0001
dis_lr = 0.0001

test_seed = tf.random.normal(shape=(4, seed_size))
gan_model = create_WGAN_GP(seed_size, gen_lr, dis_lr, dis_extra_steps=2)
    

# %% ======= training =================
shutil.rmtree('./tb_logs', ignore_errors=True)
shutil.rmtree('./Results', ignore_errors=True)
class GANMonitor(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        if batch % 40 == 0:
            generate_image(self.model, seed=test_seed, isShow=False, isSaveFile=True)
        
mon_cb = GANMonitor()
save_cb = tf.keras.callbacks.ModelCheckpoint('./Checkpoints/', verbose=1)
board_cb = tf.keras.callbacks.TensorBoard( log_dir='./tb_logs/', write_images=True, update_freq='batch') 
gan_model.fit(ds, epochs=epochs, shuffle=True, callbacks=[board_cb, mon_cb])


# %% ======= testing =================
import glob
import imageio

def create_gif():
    with imageio.get_writer('./Results/result.gif', mode='I') as writer:
        filenames = glob.glob('./Results/*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
create_gif()

# %%
import tensorflow as tf


# %%
