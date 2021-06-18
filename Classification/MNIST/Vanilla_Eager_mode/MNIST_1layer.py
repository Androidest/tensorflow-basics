# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

classCount = 10
batchSize = 100

def load_flatten_oneHot_dataset(classCount, batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (imgWidth, imgHeight) = x_train[0].shape
    plt.imshow(x_train[7])

    x_train = tf.reshape(x_train, shape=(len(x_train), imgWidth*imgHeight)) / 255 #flatten from (60000, 28, 28) to (60000, 784)
    y_train = tf.one_hot(y_train, classCount) # to one-hot form
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000, seed=123).batch(batchSize)

    x_test = tf.reshape(x_test, shape=(len(x_test), imgWidth*imgHeight)) / 255    #flatten from (10000, 28, 28) to (10000, 784)

    return (ds_train, x_test, y_test)

(ds_train, x_test, y_test) = load_flatten_oneHot_dataset(classCount=classCount, batchSize=batchSize)    


# %%
(_, pixelCount) = ds_train.element_spec[0].shape   #batch shape: ((None, 784),(None, 10))
W = tf.Variable(tf.zeros(shape=(pixelCount, classCount)), dtype=tf.float32, name="W") # Weights: (784, 10)
b = tf.Variable(tf.zeros(shape=(classCount)), dtype=tf.float32, name="b") # Bias: (10,) a one dimention tensor(vector) [...10] 
gradientDescent = tf.keras.optimizers.SGD(0.001)

isFinished = False
for epoch in range(1000):
    batchIndex = 0
    for batch in ds_train:
        batchIndex += 1

        # Train
        (x, y_real) = batch  # y_real is ground truth in one-hot form
        def loss():
            # softmax turns negative value into value that infinite approach 0 & calculate probability distribution for each value
            # without softmax, output would be very different from expected result, impossible to train
            y_predict = tf.nn.softmax(x @ W + b)
            cross_entropy = tf.math.reduce_sum(-y_real * tf.math.log(y_predict), axis=1) # 
            return cross_entropy 
        # a = W * 1
        gradientDescent.minimize(loss, var_list=[W, b])
        # print(tf.reduce_mean(tf.cast(tf.equal(a, W), tf.float32)).numpy())

        # Evaluate
        if batchIndex % 60 == 0:
            y_predict = tf.argmax(x_test @ W + b, 1)
            equality = tf.equal(y_test, y_predict)
            # calculate mean on equality is accuracy, but first convert bool(true, false) into float(1., 0.)
            accuracy = tf.reduce_mean(tf.cast(equality, tf.float32)).numpy() 
            print('Epoch:', epoch,', Batch:', batchIndex, 'Accuracy:', accuracy, flush=True)
            if accuracy > 0.925:
                isFinished = True
                break
        
    if isFinished:
        break
    


# %%
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
        visualInput = tf.reshape(greyImg, shape=(1, outputSize*outputSize)) / 255
        predict = tf.argmax(visualInput @ W + b, 1).numpy()[0]
        result = np.zeros((canvasSize, canvasSize, 3), np.uint8)
        cv2.putText(result, str(predict), (100,200), cv2.FONT_HERSHEY_COMPLEX, 6, (0,255,0), 25)
        cv2.imshow('MNIST Result', result)
        # print(predict.numpy())
    
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
a = tf.constant([[3, 4],[2,5],[5,7]])
b = tf.constant([[3],[2],[2]])
c = tf.reduce_sum(a, 1, true)
c


