# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        horizontal_flip=True,
        brightness_range=(0.4, 1.3),
        rescale=1./255,
        rotation_range=13.0, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=7.0,
        zoom_range=(1,1.1),
        fill_mode='nearest',
        cval=120.0, # constant value
        data_format='channels_last', #(samples, height, width, channels)
    )
    imGenerator.fit(x_train)

    # add a channel dimension: (n, imgHeight, imgWidth) -> (n, imgHeight, imgWidth, 1), channel=1 with grey-scale images
    y_train = tf.reshape(y_train, shape=(len(y_train))).numpy()
    y_test = tf.reshape(y_test, shape=(len(y_test))).numpy()
    train_gen = imGenerator.flow(x_train, y_train, batch_size=batchSize)
    # train_gen = tf.data.Dataset.from_tensor_slices((x_train/255, y_train)).shuffle(1000).batch(batchSize)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test/255, y_test)).batch(128)

    return (train_gen, ds_test)

(train_gen, ds_test) = loadData(classCount=classCount, batchSize=batchSize) 
# randShowGenImage(train_gen) 



# %%
import tensorflow.keras.layers as layers
# ==================== train & valid data =============================

initial_epoch = 0

if initial_epoch == 0:
    model = tf.keras.models.Sequential([
        # Conv 1
        layers.Conv2D(32, kernel_size = 3, padding='same', activation='relu', input_shape=(32,32,3), data_format='channels_last'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size = 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        # Strided Conv 1 (pooling layer)
        layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Conv 2 (Double k3 conv mimic single k5 conv)
        layers.Conv2D(64, kernel_size = 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size = 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        # Strided Conv 1 (pooling layer)
        layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Conv 2
        layers.Conv2D(84, kernel_size = 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(84, kernel_size = 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(84, kernel_size = 5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),

        # layers.Dense(128, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.5),

        layers.Dense(10)
    ])

    # SparseCategoricalCrossentropy uses class indexes as labels not hot-one form
    # Replace standalone softmax layer with embeded softmax in loss function (using from_logits=True)
    # to prevent 0 inputs from softmax layer (0 input will lead error on cross-entropy)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt_fn = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=['accuracy'])
else:
    model = tf.keras.models.load_model('./Models/Checkpoint')
    print(model.evaluate(x=ds_test, verbose=1))

# start training
lr_cb = tf.keras.callbacks.LearningRateScheduler(lambda epochIndex, lr: 0.001 * 0.95 ** epochIndex)
save_cb = tf.keras.callbacks.ModelCheckpoint(
    './Models/Checkpoint', monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=False, mode='max', save_freq='epoch'
)
history = model.fit(x=train_gen, epochs=600, initial_epoch=initial_epoch, callbacks=[lr_cb, save_cb], 
                validation_data=ds_test, validation_freq=1, verbose=1)



# %% ===================================
# Load best check point and save model
import tensorflow as tf
filePath = './Models/CIFAR10_FCNN_DA.h5'
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
model = tf.keras.models.load_model('./Models/CIFAR10_FCNN_DA.h5', compile=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

def predict(frame):
    img = tf.image.resize(frame, size=(outputSize, outputSize)) / 255
    img = tf.reshape(img, shape=(1, outputSize, outputSize, 3))
    predict = tf.argmax(model.predict(img), axis=1).numpy()[0]
    result = np.copy(frame)
    cv2.putText(result, str(class_names[predict]), (40,150), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,255,0), 2)
    cv2.imshow(winName, result)


# videoCapture(winName, canvasSize, canvasSize, predict)
screenCapture(winName, canvasSize, canvasSize, predict)






# %%
