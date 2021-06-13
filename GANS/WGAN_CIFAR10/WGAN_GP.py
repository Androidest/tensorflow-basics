# %%
import tensorflow as tf

class WGAN_GP(tf.keras.Model):
    
    def __init__(self, discriminator, generator, seed_size, dis_extra_steps=3, gp_weight=10.0):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.seed_size = seed_size
        self.dis_extra_steps = dis_extra_steps
        self.gp_weight = gp_weight

    def compile(self, gen_opt, dis_opt, gen_loss_fn, dis_loss_fn, **kwargs):
        super().compile(**kwargs, run_eagerly=True)
        self.gen_opt = gen_opt
        self.dis_opt = dis_opt
        self.gen_loss_fn = gen_loss_fn
        self.dis_loss_fn = dis_loss_fn

    @ tf.function
    def gradient_penalty(self, real_img_batch, fake_img_batch, batch_size):
        alpha = tf.random.normal((batch_size, 1, 1, 1), 0.0, 1.0)
        interpolated = real_img_batch + alpha * (fake_img_batch - real_img_batch)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            dis_pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(dis_pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.norm(grads, ord='euclidean', axis=-1)
        # norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @ tf.function
    def train_gen(self, rand_seed_batch):
        with tf.GradientTape() as gen_tape:
            fake_img_batch = self.generator(rand_seed_batch, training=True)
            fake_pred = self.discriminator(fake_img_batch, training=True)
            gen_loss = self.gen_loss_fn(fake_pred)
        
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        return gen_loss

    @ tf.function
    def train_dis(self, real_img_batch, rand_seed_batch, batch_size):
        with tf.GradientTape() as dis_tape:
            fake_img_batch = self.generator(rand_seed_batch, training=True)
            fake_pred = self.discriminator(fake_img_batch, training=True)
            real_pred = self.discriminator(real_img_batch, training=True)
            gd = self.gradient_penalty(real_img_batch, fake_img_batch, batch_size)
            dis_loss = self.dis_loss_fn(real_pred, fake_pred) + gd * self.gp_weight
            
        dis_grad = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(dis_grad, self.discriminator.trainable_variables))

        return dis_loss

    def train_step(self, real_img_batch):
        batch_size = tf.shape(real_img_batch)[0]

        for i in range(self.dis_extra_steps):
            rand_seed_batch = tf.random.normal(shape=(batch_size, self.seed_size))
            dis_loss = self.train_dis(real_img_batch, rand_seed_batch, batch_size)
        rand_seed_batch = tf.random.normal(shape=(batch_size, self.seed_size))
        gen_loss = self.train_gen(rand_seed_batch)
        
        return {"dis_loss": dis_loss, "gen_loss": gen_loss}