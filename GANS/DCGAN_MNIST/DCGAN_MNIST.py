# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

from tensorflow.python.keras.engine import training

batch_size = 256

def load_data():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape + (1,)).astype('float32') / 127.5 - 1
    ds = tf.data.Dataset.from_tensor_slices(train_images)
    ds = ds.shuffle(len(ds)).batch(batch_size)
    return ds

ds = load_data()

# %%
# ======= general functions
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def conv_transpose(model, kernels, strides, activation=None):
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(kernels, (5, 5), strides=strides, activation=activation, padding='same', use_bias=False))

def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.LeakyReLU()(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=True):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding='same', use_bias=use_bias)(x)
    return fx

def residual_block(x, filterNumb, isPooling=False):
    strides = 1
    shortcut = x
    bn_x = bn_relu(x)

    if isPooling:
        strides = 2
        shortcut = conv(bn_x, filterNumb, kernel_size=1, strides=strides)

    fx = conv(bn_x, filterNumb, kernel_size=3, strides=strides)
    fx = bn_relu(fx)
    fx = conv(fx, filterNumb, kernel_size=3)
    out = layers.Add()([shortcut, fx]) # skip
    return out

def gen_loss_fn(fake_pred):
    target_to_fake = tf.ones_like(fake_pred)
    none_fake_loss = bce_loss(target_to_fake, fake_pred)
    return none_fake_loss

def dis_loss_fn(real_pred, fake_pred):
    real_ground_truth = tf.ones_like(real_pred)
    fake_ground_truth = tf.zeros_like(fake_pred)
    real_loss = bce_loss(real_ground_truth, real_pred)
    fake_loss = bce_loss(fake_ground_truth, fake_pred)
    return real_loss + fake_loss

def generate_image(generator):
    
    grid_size = (3,3)
    w, h = grid_size
    img_count = w * h

    rand_seed_batch = tf.random.normal(shape=(img_count, 100))
    fake_img_batch = generator(rand_seed_batch, training=False)

    fig = plt.figure(figsize=grid_size, dpi=100)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    for i in range(img_count):
        plt.subplot(w, h, i+1)
        plt.imshow(fake_img_batch[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()


# ======= create model functions
def create_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    conv_transpose(model, 128, 1)
    assert model.output_shape == (None, 7, 7, 128)

    conv_transpose(model, 64, 2)
    assert model.output_shape == (None, 14, 14, 64)

    conv_transpose(model, 1, 2, activation='tanh')
    assert model.output_shape == (None, 28, 28, 1)

    return model

def create_discriminator_resnet():
    inputs = layers.Input(shape=(28,28,1)) # 28*28
    hx = conv(inputs, 16, kernel_size=3)
    hx = residual_block(hx, 16)

    hx = residual_block(hx, 32, isPooling=True) # 14*14
    
    hx = residual_block(hx, 64, isPooling=True) # 7*7

    hx = bn_relu(hx)
    hx = layers.GlobalAveragePooling2D()(hx)
    outputs = layers.Dense(1)(hx)

    model = tf.keras.Model(inputs, outputs)
    return model


# %%
# ======= training =================
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as LRDecay
import shutil

epochs = 60
gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
dis_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
generator = create_generator()
discriminator = create_discriminator_resnet()
checkpoint_path = './Checkpoints_DCGAN_MNIST/'
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                 discriminator_optimizer=dis_opt,
                                 generator=generator,
                                 discriminator=discriminator)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

shutil.rmtree('./tb_logs', ignore_errors=True)
gen_summary_writer = tf.summary.create_file_writer('./tb_logs/generator')
dis_summary_writer = tf.summary.create_file_writer('./tb_logs/discriminator')
gen_mean = tf.keras.metrics.Mean(dtype=tf.float32)
dis_mean = tf.keras.metrics.Mean(dtype=tf.float32)

@ tf.function
def train_batch(real_img_batch, rand_seed_batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_img_batch = generator(rand_seed_batch, training=True)

        fake_pred = discriminator(fake_img_batch, training=True)
        real_pred = discriminator(real_img_batch, training=True)

        gen_loss = gen_loss_fn(fake_pred)
        dis_loss = dis_loss_fn(real_pred, fake_pred)

    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    dis_grad = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    dis_opt.apply_gradients(zip(dis_grad, discriminator.trainable_variables))

    return gen_loss, dis_loss

for epoch in range(epochs):
    step = 0
    for real_img_batch in ds:
        step += 1
        rand_seed_batch = tf.random.normal(shape=(batch_size, 100))
        gen_loss, dis_loss = train_batch(real_img_batch, rand_seed_batch)
        tf.print('epoch {e}, batch {b}%, steps {s}'.format(e=epoch, b=int(step/len(ds)*100), s=gen_opt.iterations.numpy())) #, lr=gen_opt.learning_rate(gen_opt.iterations.numpy())
        if step % 10 == 0:
            generate_image(generator)
            with gen_summary_writer.as_default():
                gen_mean(gen_loss)
                tf.summary.scalar('loss', gen_mean.result(), step=gen_opt.iterations.numpy())
            with dis_summary_writer.as_default():
                dis_mean(dis_loss)
                tf.summary.scalar('loss', dis_mean.result(), step=gen_opt.iterations.numpy())
    checkpoint.save(checkpoint_path)

