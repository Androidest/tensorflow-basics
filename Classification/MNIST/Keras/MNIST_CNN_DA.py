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

def showNumbImage(x, y, numb):
    numbArr = np.where(y == numb)[0]
    index = numbArr[np.random.randint(0,len(numbArr))]
    plt.imshow(x[index])

def randShowGenImage(x, imGenerator):
    img = x[np.random.randint(0,len(x))].numpy()
    plt.figure()
    plt.imshow(np.reshape(img, (28, 28)))
    img = imGenerator.random_transform(x=img)
    plt.figure()
    plt.imshow(np.reshape(img, (28, 28)))

def loadData(classCount, batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (imgWidth, imgHeight) = x_train[0].shape

    imGenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
        rotation_range=17.0, 
        width_shift_range=0.14,
        height_shift_range=0.14,
        shear_range=13.0,
        zoom_range=(0.82,1.7),
        fill_mode='constant', 
        cval=0.0, # value for points outside the img boundaries with fill_mode='constant'
        data_format='channels_last' #(samples, height, width, channels)
    )

    # add a channel dimension: (60000, 28, 28) -> (60000, 28, 28, 1), channel=1 with grey-scale images
    x_train = tf.reshape(x_train, shape=(-1, imgHeight, imgWidth, 1)) / 255
    train_gen = imGenerator.flow(x_train, y_train, batch_size=batchSize, shuffle=True)

    x_test = tf.reshape(x_test, shape=(-1, imgHeight, imgWidth, 1)) / 255
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2000)

    # randShowGenImage(x_train, imGenerator) # random preview data augmentation effect

    return (train_gen, ds_test)

(train_gen, ds_test) = loadData(classCount=classCount, batchSize=batchSize) 


# %%
import tensorflow.keras.layers as layers
# ==================== train & valid data =============================
model = tf.keras.models.Sequential([
    # Conv 1
    layers.Conv2D(32, kernel_size = 5, padding='same', activation='relu', input_shape=(28,28,1), data_format='channels_last'),
    layers.BatchNormalization(),
    # Strided Conv 1 (pooling layer)
    layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Conv 2
    layers.Conv2D(64, kernel_size = 5, padding='same', activation='relu'),
    layers.BatchNormalization(),
    # Strided Conv 1 (pooling layer)
    layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Fully connected
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    # output 
    layers.Dense(10)
])

# SparseCategoricalCrossentropy uses class indexes as labels not hot-one form
# Replace standalone softmax layer with embeded softmax in loss function (using from_logits=True)
# to prevent 0 inputs from softmax layer (0 input will lead error on cross-entropy)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt_fn = tf.keras.optimizers.Adam(0.001)
lr_cb = tf.keras.callbacks.LearningRateScheduler(lambda epochIndex, lr: 0.001 * 0.96 ** epochIndex)
save_cb = tf.keras.callbacks.ModelCheckpoint(
    './Models/Checkpoint', monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=False, mode='max', save_freq='epoch'
)

model.compile(optimizer=opt_fn, loss=loss_fn, metrics=['accuracy'])
model.fit(x=train_gen, epochs=500, callbacks=[lr_cb, save_cb], 
            validation_data=ds_test, validation_freq=1, verbose=1)


# %% ===================================
# Load best check point and save model
import tensorflow as tf
filePath = './Models/MNIST_CNN_new.h5'
model = tf.keras.models.load_model('./Models/Checkpoint')
model.save(filePath, overwrite=True, include_optimizer=False)
model.evaluate(x=ds_test)


# %%
# ==================== test with opencv =============================
import cv2
import numpy as np
import tensorflow as tf

canvasSize = 300
outputSize = 28
penSize = 15

isDrawing = False
canvas = np.zeros((canvasSize, canvasSize, 3), np.uint8)
model = tf.keras.models.load_model('./Models/MNIST_CNN_new.h5', compile=False)

def draw(event, x, y, flags, param):
    global canvas, isDrawing
    if event == cv2.EVENT_LBUTTONDOWN:
        isDrawing = True
    elif event == cv2.EVENT_MOUSEMOVE and isDrawing:
        cv2.circle(canvas, (x,y), penSize, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        isDrawing = False
        greyImg = tf.image.rgb_to_grayscale(tf.image.resize(canvas, size=(outputSize, outputSize)))
        visualInput = tf.reshape(greyImg, shape=(1,28,28,1)) / 255
        predict = tf.argmax(model.predict(visualInput), axis=1).numpy()[0]
        result = np.zeros((canvasSize, canvasSize, 3), np.uint8)
        cv2.putText(result, str(predict), (100,200), cv2.FONT_HERSHEY_COMPLEX, 6, (0,255,0), 25)
        cv2.imshow('MNIST Result', result)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        canvas = np.zeros((canvasSize, canvasSize, 3), np.uint8)

cv2.namedWindow('MNIST Classifier')
cv2.setMouseCallback('MNIST Classifier', draw)

while(1):
    cv2.imshow('MNIST Classifier', canvas)
    key = cv2.waitKey(1)
    if key != -1 and key != 255:
        break
cv2.destroyAllWindows()



# %%
