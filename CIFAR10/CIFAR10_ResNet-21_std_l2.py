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
batchSize = 64
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
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        # brightness_range=(0.4, 1.4),
        # rescale=1./255,
        # rotation_range=10.0, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=7.0,
        # zoom_range=(1,1.1),
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
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

    return (train_gen, ds_test, imgNorm)

(train_gen, ds_test, imgNorm) = loadData(classCount=classCount, batchSize=batchSize) 
# randShowGenImage(train_gen) 



# %%
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
# ==================== train & valid data =============================
initial_epoch_batch = 0
weight_decay = 0.0001
max_learning_rate = 0.001

def residual_block(x, filterNumb, kernel_size:int=3):
    fx = layers.BatchNormalization()(x)
    fx = layers.ReLU()(fx)
    fx = layers.Conv2D(filterNumb, kernel_size, padding='same', kernel_regularizer=l2(weight_decay))(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.ReLU()(fx)
    fx = layers.Conv2D(filterNumb, kernel_size, padding='same', kernel_regularizer=l2(weight_decay))(fx)
    out = layers.Add()([x,fx]) # skip
    return out

def pooling_residual_block(x, filterNumb, kernel_size:int=3):
    fx = layers.BatchNormalization()(x)
    fx = layers.ReLU()(fx)
    fx = layers.Conv2D(filterNumb, kernel_size, strides=2, padding='same', kernel_regularizer=l2(weight_decay))(fx) # pooling conv with strides 2
    fx = layers.BatchNormalization()(fx)
    fx = layers.ReLU()(fx)
    fx = layers.Conv2D(filterNumb, kernel_size, padding='same', kernel_regularizer=l2(weight_decay))(fx)

    # pooling with conv 3 + max pool instead of conv 1 stride 2, preserve information after down sampling
    x = layers.Conv2D(filterNumb, kernel_size=1, strides=2, padding='same', kernel_regularizer=l2(weight_decay))(x)

    out = layers.Add()([x,fx]) # skip
    return out

def create_resnet():
    inputs = layers.Input(shape=(32,32,3)) # 32*32
    hx = layers.Conv2D(32, kernel_size=3, padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    hx = residual_block(hx, 32)
    hx = residual_block(hx, 32)

    hx = pooling_residual_block(hx, 64) # 16*16
    hx = residual_block(hx, 64)
    
    hx = pooling_residual_block(hx, 128) # 8*8
    hx = residual_block(hx, 128)
    
    hx = pooling_residual_block(hx, 256) # 4*4
    hx = residual_block(hx, 256)

    hx = layers.GlobalMaxPool2D()(hx)
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
lr_cb = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: max_learning_rate * 0.99 ** epoch)
save_cb = tf.keras.callbacks.ModelCheckpoint(
    './Models/Checkpoint', monitor='val_accuracy', verbose=1, save_best_only=False,
    save_weights_only=False, mode='max', save_freq='epoch'
)

ebatch = 20
for i in range(initial_epoch_batch, 5):
    history = model.fit(x=train_gen, epochs=(i+1)*ebatch, initial_epoch=i*ebatch, 
                        callbacks=[lr_cb, save_cb], 
                        validation_data=ds_test, validation_freq=1, verbose=1)
    display_history(history)



# %% ===================================
# Load best check point and save model
import tensorflow as tf
filePath = './Models/CIFAR10_ResNet-21_std_l2_0.9137.h5'
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
model = tf.keras.models.load_model('./Models/CIFAR10_ResNet-21_std_l2_0.9137.h5', compile=False)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

mean = tf.constant([[[125.3069, 122.95015, 113.866]]])
std = tf.constant([[[62.993256, 62.08861, 66.705]]])
imgNorm = lambda x: (x-mean)/std

def predict(frame):
    img = tf.image.resize(frame, size=(outputSize, outputSize))
    img = imgNorm(img)
    img = tf.reshape(img, shape=(1, outputSize, outputSize, 3))
    predict = tf.argmax(model.predict(img), axis=1).numpy()[0]
    result = np.copy(frame)
    cv2.putText(result, str(class_names[predict]), (40,150), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,255,0), 2)
    cv2.imshow(winName, result)


# videoCapture(winName, canvasSize, canvasSize, predict)
screenCapture(winName, canvasSize, canvasSize, predict)






# %%
