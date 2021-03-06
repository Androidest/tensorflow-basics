# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.keras.backend import dropout

classCount = 10
batchSize = 512
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
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        featurewise_center=True,
        featurewise_std_normalization=True,
        brightness_range=(0.5, 1.3),
        # rescale=1./255,
        rotation_range=12.0,
        # shear_range=7.0,
        zoom_range=(1,1.1),
        fill_mode='constant',
        cval=255.0, # constant value
        data_format='channels_last', #(samples, height, width, channels)
    )
    imGenerator.fit(x_train)
    mean = imGenerator.mean
    std = imGenerator.std + 1e-07
    print('mean: '+str(mean))
    print('std: '+str(std))
    imgNorm = lambda x: (x-mean)/std

    # add a channel dimension: (n, imgHeight, imgWidth) -> (n, imgHeight, imgWidth, 1), channel=1 with grey-scale images
    y_train = tf.reshape(y_train, shape=(len(y_train))).numpy()
    train_gen = imGenerator.flow(x_train, y_train, batch_size=batchSize, shuffle=True)

    x_test = imgNorm(x_test)
    y_test = tf.reshape(y_test, shape=(len(y_test))).numpy()
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)

    return (train_gen, ds_test, imgNorm)

(train_gen, ds_test, imgNorm) = loadData(classCount=classCount, batchSize=batchSize) 
# randShowGenImage(train_gen) 



# %%
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
# tf.keras.applications.ResNet50V2
# ==================== train & valid data =============================
initial_epoch_batch = 0
weight_decay = 0.0001
max_learning_rate = 0.1

def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.ReLU()(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=False):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding='same', use_bias=use_bias, kernel_regularizer=l2(weight_decay))(x)
    return fx

def sconv(x, filterNumb, kernel_size, strides=1, use_bias=True):
    fx = layers.SeparableConv2D(filterNumb, kernel_size, strides, padding='same', use_bias=use_bias, kernel_regularizer=l2(weight_decay))(x)
    return fx

def bottleneck_res_block(x, filterNumb, type='identity', useRelu=True):
    strides = 1
    bn_x = bn_relu(x, useRelu=useRelu)

    # different type of shortcut
    if type is 'projection':
        shortcut = conv(bn_x, 4 * filterNumb, kernel_size=3)
    elif type is 'projection2':
        shortcut = conv(bn_x, 4 * filterNumb, kernel_size=1)
    elif type is 'pooling':
        strides = 2
        shortcut = layers.MaxPool2D(pool_size=(2,2), strides=strides)(x)
    elif type is 'pooling2':
        strides = 2
        shortcut = conv(x, 4 * filterNumb, kernel_size=1, strides=strides)
    else: # identity
        shortcut = x # normal bottleneck
    
    fx = conv(bn_x, filterNumb, kernel_size=1)
    fx = bn_relu(fx)
    fx = conv(fx, filterNumb, kernel_size=3, strides=strides)
    fx = bn_relu(fx)
    fx = conv(fx, 4 * filterNumb, kernel_size=1)

    out = layers.Add()([shortcut, fx]) # skip
    return out

def create_resnet():
    inputs = layers.Input(shape=(32,32,3)) # 32*32
    hx = bottleneck_res_block(inputs, 16, type='projection2', useRelu=False)
    hx = bottleneck_res_block(hx, 16)
    hx = bottleneck_res_block(hx, 16)
    hx = bottleneck_res_block(hx, 16, type='pooling2') # 16*16

    hx = bottleneck_res_block(hx, 32, type='projection2') 
    hx = bottleneck_res_block(hx, 32) 
    hx = bottleneck_res_block(hx, 32, type='pooling2') # 8*8

    hx = bottleneck_res_block(hx, 64, type='projection2')
    hx = bottleneck_res_block(hx, 64)
    hx = bottleneck_res_block(hx, 64, type='pooling2') # 4*4

    hx = bottleneck_res_block(hx, 128, type='projection2')
    hx = bottleneck_res_block(hx, 128)
    hx = bottleneck_res_block(hx, 128, type='pooling2') # 2*2

    hx = bottleneck_res_block(hx, 256, type='projection2')
    hx = bottleneck_res_block(hx, 256)

    hx = bn_relu(hx)
    hx = layers.GlobalAveragePooling2D()(hx)
    outputs = layers.Dense(10)(hx)

    model = tf.keras.Model(inputs, outputs)
    return model

def display_history(history):
    history = history.history
    if len(history) == 0:
        return
    plt.figure(0)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

    plt.figure(1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

if initial_epoch_batch == 0:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt_fn = tf.keras.optimizers.Adam(max_learning_rate)
    model = create_resnet()
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=['accuracy'])
else:
    model = tf.keras.models.load_model('./Models/Checkpoint')
    print(model.evaluate(x=ds_test, verbose=1))

# start training
def lr_scheduler(epoch, lr):
    lr = max_learning_rate
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr
lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
save_cb = tf.keras.callbacks.ModelCheckpoint(
    './Models/Checkpoint', monitor='val_accuracy', verbose=1, save_best_only=False,
    save_weights_only=False, mode='max', save_freq='epoch'
)

ebatch = 20
for i in range(initial_epoch_batch, 10):
    history = model.fit(x=train_gen, epochs=(i+1)*ebatch, initial_epoch=i*ebatch, 
                        callbacks=[lr_cb, save_cb], workers=3,
                        validation_data=ds_test, validation_freq=1, verbose=1)
    display_history(history)



# %% ===================================
# Load best check point and save model
import tensorflow as tf
filePath = './Models/CIFAR10_ResNet-bottleneck-29.h5'
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
model = tf.keras.models.load_model('./Models/CIFAR10_ResNet-bottleneck-29.h5', compile=False)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def predict(frame):
    img = tf.image.resize(frame, size=(outputSize, outputSize))
    img = tf.reshape(img, shape=(1, outputSize, outputSize, 3))
    predict = tf.argmax(model.predict(img), axis=1).numpy()[0]
    result = np.copy(frame)
    cv2.putText(result, str(class_names[predict]), (40,150), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,255,0), 2)
    cv2.imshow(winName, result)


# videoCapture(winName, canvasSize, canvasSize, predict)
# screenCapture(winName, canvasSize, canvasSize, predict)






# %%
