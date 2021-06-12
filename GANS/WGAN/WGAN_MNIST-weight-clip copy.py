# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import glob
import imageio
import os

batch_size = 512

def load_data():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape + (1,)).astype('float32') / 127.5 - 1
    ds = tf.data.Dataset.from_tensor_slices(train_images)
    ds = ds.shuffle(len(ds)).batch(batch_size)
    return ds

ds = load_data()

# %%
# ======= general functions
test_seed = tf.random.normal(shape=(9, 200))

def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.LeakyReLU()(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=True):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding='same', 
                    use_bias=use_bias)(x)
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

def gen_accuracy_fn(fake_pred):
    fooled = tf.where(fake_pred > 0, 1.0, 0.0)
    return tf.reduce_mean(fooled)

def gen_loss_fn(fake_pred):
    return -tf.reduce_mean(fake_pred)

def dis_loss_fn(real_pred, fake_pred):
    return tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

def generate_image(generator, isShow=True, isSaveFile=False):
    grid_size = (3,3)
    w, h = grid_size
    img_count = w * h

    fake_img_batch = generator(rand_seed_batch, training=False)

    fig = plt.figure(figsize=grid_size, dpi=100)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    for i in range(img_count):
        plt.subplot(w, h, i+1)
        plt.imshow(fake_img_batch[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if isShow: 
        plt.show()
    if isSaveFile:
        if not os.path.exists('./Results'):
            os.makedirs('./Results')
        plt.savefig('./Results/{:06d}.png'.format(gen_opt.iterations.numpy()))

def create_gif():
    with imageio.get_writer('./Results/result.gif', mode='I') as writer:
        filenames = glob.glob('./Results/*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

def conv_transpose(model, kernels, strides, use_bn=True, activation=None):
    if use_bn:
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(kernels, (3, 3), strides=strides, 
                                activation=activation, padding='same', 
                                use_bias=False))

# ======= create model functions
def create_generator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(200,)))
    model.add(layers.Reshape((2, 2, 50)))

    conv_transpose(model, 64, 2, use_bn=False)
    assert model.output_shape == (None, 4, 4, 64)

    conv_transpose(model, 32, 2)
    assert model.output_shape == (None, 8, 8, 32)

    model.add(layers.Cropping2D(((0,1),(0,1))))
    assert model.output_shape == (None, 7, 7, 32)  

    conv_transpose(model, 16, 2)
    assert model.output_shape == (None, 14, 14, 16)

    conv_transpose(model, 1, 2, activation='tanh')
    assert model.output_shape == (None, 28, 28, 1)

    return model

def create_discriminator_resnet():
    inputs = layers.Input(shape=(28,28,1)) # 28*28
    hx = conv(inputs, 16, kernel_size=3)
    hx = residual_block(hx, 16) # 14*14

    hx = residual_block(hx, 32, isPooling=True) # 14*14
    hx = residual_block(hx, 32) # 14*14
    
    hx = residual_block(hx, 64, isPooling=True) # 7*7
    hx = residual_block(hx, 64) # 7*7

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
gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0009)
dis_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
generator = create_generator()
discriminator = create_discriminator_resnet()
checkpoint_path = './Checkpoints/'
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                 discriminator_optimizer=dis_opt,
                                 generator=generator,
                                 discriminator=discriminator)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

shutil.rmtree('./tb_logs', ignore_errors=True)
shutil.rmtree('./Results', ignore_errors=True)
gen_summary_writer = tf.summary.create_file_writer('./tb_logs/generator')
dis_summary_writer = tf.summary.create_file_writer('./tb_logs/discriminator')
gen_mean = tf.keras.metrics.Mean(dtype=tf.float32)
dis_mean = tf.keras.metrics.Mean(dtype=tf.float32)

@ tf.function
def train_gen(rand_seed_batch):
    with tf.GradientTape() as gen_tape:
        fake_img_batch = generator(rand_seed_batch, training=True)
        fake_pred = discriminator(fake_img_batch, training=True)
        gen_loss = gen_loss_fn(fake_pred)
        gen_accuracy = gen_accuracy_fn(fake_pred)

    tf.print(' - gen_accuracy', gen_accuracy, ' gen_loss', gen_loss)
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))

    return gen_loss, gen_accuracy

@ tf.function
def train_dis(real_img_batch, rand_seed_batch):
    with tf.GradientTape() as dis_tape:
        fake_img_batch = generator(rand_seed_batch, training=True)

        fake_pred = discriminator(fake_img_batch, training=True)
        real_pred = discriminator(real_img_batch, training=True)
        gen_accuracy = gen_accuracy_fn(fake_pred)

        dis_loss = dis_loss_fn(real_pred, fake_pred)

    dis_grad = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
    dis_opt.apply_gradients(zip(dis_grad, discriminator.trainable_variables))

    # weight clipping
    for w in discriminator.trainable_variables:
        w.assign(tf.clip_by_value(w, -0.15, 0.15))
    return dis_loss, gen_accuracy

def output_logs(step, gen_loss, dis_loss):
    iterations = dis_opt.iterations.numpy()
    percentage = int(np.floor(step/len(ds)*100))
    epoch = int(np.floor(iterations/len(ds)))

    tf.print('iterations {i}, batch {p}%, epoch {e}'.format(i=iterations, p=percentage, e=epoch))
    
    if iterations % 30 == 0:
        generate_image(generator, isShow=False, isSaveFile=True)
        with gen_summary_writer.as_default():
            gen_mean(gen_loss)
            tf.summary.scalar('loss', gen_mean.result(), step=iterations)
        with dis_summary_writer.as_default():
            dis_mean(dis_loss)
            tf.summary.scalar('loss', dis_mean.result(), step=iterations)

for epoch in range(epochs):
    step = 0
    gen_loss = 0
    for real_img_batch in ds:
        rand_seed_batch = tf.random.normal(shape=(batch_size, 200))
        
        dis_loss, gen_accuracy = train_dis(real_img_batch, rand_seed_batch)
        while(gen_accuracy <= 0.5):
            gen_loss, gen_accuracy = train_gen(rand_seed_batch)
            
        step += 1
        output_logs(step, gen_loss, dis_loss)

    checkpoint.save(checkpoint_path)


# %%
create_gif()
