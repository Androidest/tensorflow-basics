# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

classCount = 10
batchSize = 64

def loadData(classCount, batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (imgWidth, imgHeight) = x_train[0].shape
    index = np.where(y_train == 7)[0][55]
    plt.imshow(x_train[index])

    # add a channel dimension: (60000, 28, 28) -> (60000, 28, 28, 1), channel=1 with grey-scale images
    x_train = x_train / 255
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchSize)

    x_test = x_test / 255
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2000)

    return (ds_train, ds_test)

(ds_train, ds_test) = loadData(classCount=classCount, batchSize=batchSize)    


# %%
import tensorflow.keras.layers as layers
# ==================== train & valid data =============================
model = tf.keras.models.Sequential([

    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(10)
])

# SparseCategoricalCrossentropy uses class indexes as labels not hot-one form
# Replace standalone softmax layer with embeded softmax in loss function (using from_logits=True)
# to prevent 0 inputs from softmax layer (0 input will lead error on cross-entropy)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt_fn = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=opt_fn, loss=loss_fn, metrics=['accuracy'])
model.fit(x=ds_train, epochs=100, validation_data=ds_test, validation_freq=1, verbose=2)


# %%
# ==================== test with opencv =============================
import cv2
import numpy as np

canvasSize = 300
outputSize = 28
penSize = 15

isDrawing = False
canvas = np.zeros((canvasSize, canvasSize, 3), np.uint8)

def draw(event, x, y, flags, param):
    global canvas, isDrawing
    if event == cv2.EVENT_LBUTTONDOWN:
        isDrawing = True
    elif event == cv2.EVENT_MOUSEMOVE and isDrawing:
        cv2.circle(canvas, (x,y), penSize, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        isDrawing = False
        greyImg = tf.image.rgb_to_grayscale(tf.image.resize(canvas, size=(outputSize, outputSize)))
        visualInput = tf.reshape(greyImg, shape=(1,28,28)) / 255
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
