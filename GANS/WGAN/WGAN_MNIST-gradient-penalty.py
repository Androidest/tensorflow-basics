# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2
import glob
import imageio
import os

batch_size = 256

def load_data():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape + (1,)).astype('float32') / 127.5 - 1
    ds = tf.data.Dataset.from_tensor_slices(train_images)
    ds = ds.shuffle(len(ds)).batch(batch_size, drop_remainder=True)
    return ds

ds = load_data()

# %%
# ======= general functions
test_seed = tf.random.normal(shape=(9, 100))

def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.LeakyReLU(0.2)(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=True):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding='same', 
                    use_bias=use_bias, kernel_regularizer=L2(0.01))(x)
    return fx

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

    fake_img_batch = generator(test_seed, training=False)

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
        plt.savefig('./Results/{:06d}.png'.format(dis_opt.iterations.numpy()))

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
    model.add(layers.Conv2DTranspose(kernels, (3, 3), strides=strides, 
                                activation=activation, padding='same', 
                                use_bias=False))
    if use_bn:
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

def gradient_penalty(real_img_batch, fake_img_batch):
    alpha = tf.random.normal((real_img_batch.shape[0], 1, 1, 1), 0.0, 1.0)
    interpolated = real_img_batch + alpha * (fake_img_batch - real_img_batch)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        dis_pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(dis_pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    # norm = tf.norm(grads, ord='euclidean', axis=-1)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# ======= create model functions
def create_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)

    conv_transpose(model, 128, 2)
    assert model.output_shape == (None, 8, 8, 128)

    conv_transpose(model, 64, 2)
    assert model.output_shape == (None, 16, 16, 64)

    conv_transpose(model, 1, 2, activation='tanh', use_bn=False)
    assert model.output_shape == (None, 32, 32, 1)
    
    model.add(layers.Cropping2D((2,2)))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def create_discriminator():
    inputs = layers.Input(shape=(28,28,1)) # 28*28
    hx = layers.ZeroPadding2D((2,2))(inputs) # 32*32

    hx = conv(hx, 32, kernel_size=5, strides=2)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.3)(hx)

    hx = conv(hx, 64, kernel_size=5, strides=2)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.3)(hx)

    hx = conv(hx, 128, kernel_size=5, strides=2)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.2)(hx)

    hx = layers.Flatten()(hx)
    hx = layers.Dense(512)(hx)
    hx = layers.LeakyReLU(0.2)(hx)
    hx = layers.Dropout(0.2)(hx)

    outputs = layers.Dense(1)(hx)

    model = tf.keras.Model(inputs, outputs)
    return model


# %%
# ======= training =================
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as LRDecay
import shutil

epochs = 40
gen_opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9)
dis_opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9)
generator = create_generator()
discriminator = create_discriminator()
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
    
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))

    gen_accuracy = gen_accuracy_fn(fake_pred)
    tf.print(' -- GEN: gen_accuracy', gen_accuracy, ' gen_loss', gen_loss)
    return gen_loss, gen_accuracy

@ tf.function
def train_dis(real_img_batch, rand_seed_batch):
    with tf.GradientTape() as dis_tape:
        fake_img_batch = generator(rand_seed_batch, training=True)
        fake_pred = discriminator(fake_img_batch, training=True)
        real_pred = discriminator(real_img_batch, training=True)
        gd = gradient_penalty(real_img_batch, fake_img_batch)
        dis_loss = dis_loss_fn(real_pred, fake_pred) + gd * 10.0
        
    dis_grad = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
    dis_opt.apply_gradients(zip(dis_grad, discriminator.trainable_variables))

    gen_accuracy = gen_accuracy_fn(fake_pred)
    tf.print(' - DIS: gen_accuracy', gen_accuracy, ' dis_loss', dis_loss)
    return dis_loss, gen_accuracy

def output_logs(step, gen_loss, dis_loss):
    iterations = dis_opt.iterations.numpy()
    percentage = int(np.floor(step/len(ds)*100))
    epoch = int(np.floor(iterations/len(ds)))

    tf.print('iterations {i}, batch {p}%, epoch {e}'.format(i=iterations, p=percentage, e=epoch))
    
    if iterations % 40 == 0:
        generate_image(generator, isShow=False, isSaveFile=True)
        with gen_summary_writer.as_default():
            gen_mean(gen_loss)
            tf.summary.scalar('loss', gen_mean.result(), step=iterations)
        with dis_summary_writer.as_default():
            dis_mean(dis_loss)
            tf.summary.scalar('loss', dis_mean.result(), step=iterations)

dis_loss = 0
gen_loss = 0
gen_accuracy = 0
for epoch in range(epochs):
    step = 0
    for real_img_batch in ds:
        for i in range(2):
            rand_seed_batch = tf.random.normal(shape=(batch_size, 100))
            dis_loss, gen_accuracy = train_dis(real_img_batch, rand_seed_batch)
        rand_seed_batch = tf.random.normal(shape=(batch_size, 100))
        gen_loss, gen_accuracy = train_gen(rand_seed_batch)
        
        step += 1
        output_logs(step, gen_loss, dis_loss)
    checkpoint.save(checkpoint_path)


# %%
create_gif()
