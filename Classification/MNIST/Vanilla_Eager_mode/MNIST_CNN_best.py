# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

classCount = 10
batchSize = 64

def load_2d_oneHot_dataset(classCount, batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (imgWidth, imgHeight) = x_train[0].shape
    index = np.where(y_train == 7)[0][55]
    plt.imshow(x_train[index])

    # add a channel dimension: (60000, 28, 28) -> (60000, 28, 28, 1), channel=1 with grey-scale images
    x_train = tf.reshape(x_train, shape=(-1, imgHeight, imgWidth, 1)) / 255 
    y_train = tf.one_hot(y_train, classCount) # to one-hot form
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchSize)

    x_test = tf.reshape(x_test, shape=(-1, imgHeight, imgWidth, 1)) / 255
    y_test = tf.constant(y_test, dtype=tf.int32)  
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(500)

    return (ds_train, ds_test)

(ds_train, ds_test) = load_2d_oneHot_dataset(classCount=classCount, batchSize=batchSize)    


# %%
# ==================== train & valid data =============================
(_, imWidth, imHeight, channels) = ds_train.element_spec[0].shape   #batch shape: ((None, 28,28,1),(None, 10))

# convolutional layers
# 32 kinds of 5*5 kernels will be trained later, resulting 32 filtered images with the same size as source
# 1th convolutional layer
conv1_features = 32
kernel_W1 = tf.Variable(tf.random.truncated_normal([5, 5, channels, conv1_features], stddev=0.1)) 
conv_b1 = tf.Variable(tf.zeros(shape=(conv1_features))) 

# 1th strided convolutional layer
skernel_W1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, conv1_features], stddev=0.1)) 
sconv_b1 = tf.Variable(tf.zeros(shape=(conv1_features))) 

# 2th convolutional layer
conv2_features = 64
kernel_W2 = tf.Variable(tf.random.truncated_normal([5, 5, conv1_features, conv2_features], stddev=0.1)) 
conv_b2 = tf.Variable(tf.zeros(shape=(conv2_features))) 

# 2th strided convolutional layer
skernel_W2 = tf.Variable(tf.random.truncated_normal([5, 5, 1, conv2_features], stddev=0.1)) 
sconv_b2 = tf.Variable(tf.zeros(shape=(conv2_features))) 
conv2_output_size = int( imWidth/4 * imHeight/4 * conv2_features )

# fully connected layers
f1_size = 1024
W1 = tf.Variable(tf.random.truncated_normal([conv2_output_size, f1_size], stddev=0.1)) 
b1 = tf.Variable(tf.zeros(shape=(f1_size)))  

# output layer
W2 = tf.Variable(tf.random.truncated_normal([f1_size, classCount], stddev=0.1)) 
b2 = tf.Variable(tf.zeros(shape=(classCount))) 

gradientDescent = tf.keras.optimizers.Adam(0.001) # smaller learning rate for stable learning
dropoutRate = 0.4
dropoutRescale = 1 - dropoutRate

@ tf.function
def batch_norm(W, mean, variance, axes=[0,1,2], training=False):
    if training:
        batch_mean, batch_variance = tf.nn.moments(W, axes=axes)
        mean.assign(mean*bn_decay + batch_mean*(1 - bn_decay)) 
        variance.assign(variance*bn_decay + batch_variance*(1 - bn_decay))
        return tf.nn.batch_normalization(W, batch_mean, batch_variance, offset=0, scale=1, variance_epsilon=0.001)
    else:
        return tf.nn.batch_normalization(W, mean, variance, offset=0, scale=1, variance_epsilon=0.001)


@ tf.function
def cnn_predict(img, isTraining=False):
    # 1th convolutional layer
    conv1 = tf.nn.conv2d(img, kernel_W1, strides=[1,1,1,1], padding='SAME') + conv_b1
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.conv2d(conv1, skernel_W1, strides=[1,2,2,1], padding='SAME') + sconv_b1
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, rate=dropoutRate) if isTraining else conv1 * dropoutRescale

    # 2th convolutional layer
    conv2 = tf.nn.conv2d(conv1, kernel_W2, strides=[1,1,1,1], padding='SAME') + conv_b2
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.conv2d(conv2, skernel_W2, strides=[1,2,2,1], padding='SAME') + sconv_b2
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, rate=dropoutRate) if isTraining else conv2 * dropoutRescale
    conv2 = tf.reshape(conv2, shape=[-1, conv2_output_size])

    # fully connected layers
    h1 = conv2 @ W1 + b1
    h1 = tf.nn.relu(h1)
    h1 = tf.nn.dropout(h1, rate=dropoutRate) if isTraining else h1 * dropoutRescale

    # output layer
    h2 = h1 @ W2 + b2
    h2 = tf.nn.softmax(h2) if isTraining else tf.argmax(h2, axis=1, output_type=tf.int32)
    return h2

isFinished = False
for epoch in range(1000):
    batchIndex = 0
    for batch in ds_train:
        batchIndex += 1

        # Train
        (x, y_real) = batch  # y_real is ground truth in one-hot form
        def loss():
            y_predict = cnn_predict(x, isTraining=True) 
            # calculate loss using cross entropy
            cross_entropy = -tf.math.reduce_sum(y_real * tf.math.log(y_predict)) # -Σ(y_real * log(y_predict))
            return cross_entropy
        gradientDescent.minimize(loss, var_list=[ kernel_W1, conv_b1, skernel_W1, sconv_b1, 
                                                  kernel_W2, conv_b2, skernel_W2, sconv_b2,
                                                  W1, b1, W2, b2 ])

        # Evaluate
        if batchIndex % 10 == 0:
            accuracy = 0
            for batch in ds_test:
                (x_test, y_test) = batch
                y_predict = cnn_predict(x_test)
                equality = tf.equal(y_test, y_predict)
                accuracy += tf.reduce_mean(tf.cast(equality, tf.float32)).numpy() 
            accuracy /= len(ds_test)

            print('Epoch:', epoch,', Batch:', batchIndex, 'Accuracy:', accuracy, flush=True)
            if accuracy > 0.996:
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
        visualInput = tf.reshape(greyImg, shape=(1, 28, 28, 1)) / 255
        predict = cnn_predict(visualInput).numpy()[0]

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
