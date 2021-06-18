# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from tensorflow.python.keras.backend import dropout

classCount = 10
batchSize = 128
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

# random preview data augmentation effect
def randShowGenImage(imGenerator):
    [x, y] = imGenerator.next()
    for i in range(len(y)):
        img = x[i]
        label = class_names[y[i]]
        plt.figure(i)
        plt.imshow(img)
        plt.xlabel(label)

def loadData(classCount, batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (imgWidth, imgHeight, channels) = x_train[0].shape

    imGenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        horizontal_flip=True,
        # brightness_range=(0.4, 1.4),
        # rescale=1./255,
        # rotation_range=10.0, 
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=7.0,
        # zoom_range=(1,1.1),
        fill_mode='constant',
        cval=0.0, # constant value
        data_format='channels_last', #(samples, height, width, channels)
    )

    # add a channel dimension: (n, imgHeight, imgWidth) -> (n, imgHeight, imgWidth, 1), channel=1 with grey-scale images
    y_train = tf.reshape(y_train, shape=(len(y_train))).numpy()
    y_train = tf.one_hot(y_train, 10)
    train_gen = imGenerator.flow(x_train/127.5-1, y_train, batch_size=batchSize, shuffle=True)

    # x_test = imgNorm(x_test)
    y_test = tf.reshape(y_test, shape=(len(y_test))).numpy()
    y_test = tf.one_hot(y_test, 10)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test/127.5-1, y_test)).batch(512)

    return (train_gen, ds_test)

(train_gen, ds_test) = loadData(classCount=classCount, batchSize=batchSize) 
# randShowGenImage(train_gen) 


# %%
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
# ==================== train & valid data =============================
initial_epoch_batch = 0
weight_decay = 0.01
init_learning_rate = 0.001

def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.LeakyReLU(0.01)(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=True, padding='same'):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding=padding, use_bias=use_bias, kernel_regularizer=l2(weight_decay))(x) #
    return fx

def residual_block(x, filterNumb, isPooling=False):
    strides = 1
    shortcut = x
    bn_x = bn_relu(x)

    if isPooling:
        strides = 2
        shortcut = conv(x, filterNumb, kernel_size=1, strides=strides)
    
    fx = conv(bn_x, filterNumb, kernel_size=3, strides=strides)
    fx = bn_relu(fx)
    fx = conv(fx, filterNumb, kernel_size=3)
    out = layers.Add()([shortcut, fx]) # skip
    return out

def create_resnet():
    inputs = layers.Input(shape=(32,32,3)) # 32*32
    hx = conv(inputs, 16, kernel_size=3)
    hx = residual_block(hx, 16)
    hx = residual_block(hx, 16)

    hx = residual_block(hx, 32, isPooling=True)# 16*16
    hx = residual_block(hx, 32)
    hx = residual_block(hx, 32)
    
    hx = residual_block(hx, 64, isPooling=True) # 8*8
    hx = residual_block(hx, 64)
    hx = residual_block(hx, 64)

    hx = residual_block(hx, 128, isPooling=True) # 4*4
    hx = residual_block(hx, 128)
    
    hx = residual_block(hx, 256, isPooling=True) 
    hx = residual_block(hx, 256)

    hx = bn_relu(hx)
    hx = layers.GlobalAveragePooling2D()(hx)
    
    hx = layers.Dropout(0.2)(hx)
    outputs = layers.Dense(10)(hx)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model

if initial_epoch_batch == 0:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    opt_fn = tf.keras.optimizers.Adam(init_learning_rate)
    model = create_resnet()
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=['accuracy'])
else:
    model = tf.keras.models.load_model('./Models/Checkpoint')
    print(model.evaluate(x=ds_test, verbose=1))

# start training
shutil.rmtree('./tb_logs', ignore_errors=True)
tb_cb = tf.keras.callbacks.TensorBoard('./tb_logs/')

ebatch = 10
for i in range(initial_epoch_batch, 5):
    model.fit(x=train_gen, epochs=(i+1)*ebatch, initial_epoch=i*ebatch, 
            callbacks=[tb_cb], 
            validation_data=ds_test, validation_freq=1, verbose=1)
model.save('/Models/CIFAR10_ResNet-18.h5', overwrite=True, include_optimizer=True)


# %% ===================================
# Load best check point and save model
import tensorflow as tf
filePath = './Models/CIFAR10_ResNet-18_0.9014.h5'
model = tf.keras.models.load_model('./Models/Checkpoint')
model.save(filePath, overwrite=True, include_optimizer=False)
print(model.evaluate(x=ds_test, verbose=0))


# %%
# ==================== test with opencv =============================
import cv2
import numpy as np
import tensorflow as tf
from common import videoCapture, screenCapture

winName = 'CIFAR10 Classifier'
canvasSize = 200
outputSize = 32
model = tf.keras.models.load_model('./Models/CIFAR10_ResNet-18_0.9014.h5', compile=False)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# mean = tf.constant([[[125.3069, 122.95015, 113.866]]])
# std = tf.constant([[[62.993256, 62.08861, 66.705]]])
# imgNorm = lambda x: (x-mean)/std

def predict(frame):
    img = tf.image.resize(frame, size=(outputSize, outputSize))
    # img = imgNorm(img)
    img = tf.reshape(img, shape=(1, outputSize, outputSize, 3))
    predict = tf.argmax(model.predict(img), axis=1).numpy()[0]
    result = np.copy(frame)
    cv2.putText(result, str(class_names[predict]), (20,150), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,255,0), 2)
    cv2.imshow(winName, result)


# videoCapture(winName, canvasSize, canvasSize, predict)
screenCapture(winName, canvasSize, canvasSize, predict)






# %%
