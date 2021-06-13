# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

classCount = 10
batchSize = 100

def load_flatten_oneHot_dataset(classCount, batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (imgWidth, imgHeight) = x_train[0].shape
    index = np.where(y_train == 7)[0][55]
    plt.imshow(x_train[index])

    x_train = tf.reshape(x_train, shape=(len(x_train), imgWidth*imgHeight)) / 255 #flatten from (60000, 28, 28) to (60000, 784)
    y_train = tf.one_hot(y_train, classCount) # to one-hot form
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchSize)

    x_test = tf.reshape(x_test, shape=(len(x_test), imgWidth*imgHeight)) / 255    #flatten from (10000, 28, 28) to (10000, 784)

    return (ds_train, x_test, y_test)

(ds_train, x_test, y_test) = load_flatten_oneHot_dataset(classCount=classCount, batchSize=batchSize)    


# %%
# ==================== train & valid data =============================
(_, pixelCount) = ds_train.element_spec[0].shape   #batch shape: ((None, 784),(None, 10))

# more than one layer should use random or pre-trained weights, otherwise loss would not decrease
# different initial weights (start point) have different local extrema,
# means different accuracy limit & learning speed, 
# need multiple re-train to find better weights
W1 = tf.Variable(tf.random.truncated_normal([pixelCount, 300], stddev=0.1)) 
b1 = tf.Variable(tf.zeros(shape=(300))) 

W2 = tf.Variable(tf.random.truncated_normal([300, 100], stddev=0.1)) 
b2 = tf.Variable(tf.zeros(shape=(100))) 

W3 = tf.Variable(tf.random.truncated_normal([100, classCount], stddev=0.1)) 
b3 = tf.Variable(tf.zeros(shape=(classCount))) 
learningRate = 0.001 # smaller learning rate for stable learning

isFinished = False
for epoch in range(1000):
    batchIndex = 0
    for batch in ds_train:
        batchIndex += 1

        # Train
        (x, y_real) = batch  # y_real is ground truth in one-hot form
        with tf.GradientTape() as tape:
            # use ReLU acti-func (Rectified Linear Unit) for better & faster learning
            # by removing unrelated input from previous layer (Don't care about data with minus value)
            h1 = tf.nn.relu(x @ W1 + b1)
            h2 = tf.nn.relu(h1 @ W2 + b2)
            h3 = tf.nn.relu(h2 @ W3 + b3)
            y_predict = tf.nn.softmax(h3)
            loss = -tf.math.reduce_sum(y_real * tf.math.log(y_predict)) # -Σ(y_real * log(y_predict))

        # compute gradient
        grads = tape.gradient(loss, [ W3, b3, W2, b2, W1, b1 ])
        # update weights: w' = w - lr*grad
        for p, g in zip([ W3, b3, W2, b2, W1, b1 ], grads):
            p.assign_sub(learningRate * g)

        # Evaluate
        if batchIndex % 60 == 0:
            h1 = tf.nn.relu(x_test @ W1 + b1)
            h2 = tf.nn.relu(h1 @ W2 + b2)
            h3 = h2 @ W3 + b3
            y_predict = tf.argmax(h3, 1)
            equality = tf.equal(y_test, y_predict)
            # calculate mean on equality is accuracy, but first convert bool(true, false) into float(1., 0.)
            accuracy = tf.reduce_mean(tf.cast(equality, tf.float32)).numpy() 
            print('Epoch:', epoch,', Batch:', batchIndex, 'Accuracy:', accuracy, flush=True)
            if accuracy > 0.98:
                isFinished = True
                break
        
    if isFinished:
        break
    


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
        visualInput = tf.reshape(greyImg, shape=(1, outputSize*outputSize)) / 255

        h1 = tf.nn.relu(visualInput @ W1 + b1)
        h2 = tf.nn.relu(h1 @ W2 + b2)
        h3 = h2 @ W3 + b3
        predict = tf.argmax(h3, 1).numpy()[0]

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
