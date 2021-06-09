# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

classCount = 10
batchSize = 1000

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
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2000)

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
conv_b1 = tf.Variable(tf.zeros(shape=(conv1_features))) # one bias for each feature, add with broadcasting rule

# 1th strided convolutional layer
skernel_W1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, conv1_features], stddev=0.1)) 
sconv_b1 = tf.Variable(tf.zeros(shape=(conv1_features))) # one bias for each feature, add with broadcasting rule

# 2th convolutional layer
conv2_features = 64
kernel_W2 = tf.Variable(tf.random.truncated_normal([5, 5, conv1_features, conv2_features], stddev=0.1)) 
conv_b2 = tf.Variable(tf.zeros(shape=(conv2_features))) # one bias for each feature, add with broadcasting rule

# 2th strided convolutional layer
skernel_W2 = tf.Variable(tf.random.truncated_normal([5, 5, 1, conv2_features], stddev=0.1)) 
sconv_b2 = tf.Variable(tf.zeros(shape=(conv2_features))) # one bias for each feature, add with broadcasting rule

# 3th convolutional layer
conv3_features = 128
kernel_W3 = tf.Variable(tf.random.truncated_normal([5, 5, conv2_features, conv3_features], stddev=0.1)) 
conv_b3 = tf.Variable(tf.zeros(shape=(conv3_features))) # one bias for each feature, add with broadcasting rule

# 2th strided convolutional layer
skernel_W3 = tf.Variable(tf.random.truncated_normal([5, 5, 1, conv3_features], stddev=0.1)) 
sconv_b3 = tf.Variable(tf.zeros(shape=(conv3_features)))

# 4th convolutional layer
kernel_W4 = tf.Variable(tf.random.truncated_normal([4, 4, conv3_features, classCount], stddev=0.1)) 
conv_b4 = tf.Variable(tf.zeros(shape=(classCount))) # one bias for each feature, add with broadcasting rule

gradientDescent = tf.keras.optimizers.Adam(0.01) # smaller learning rate for stable learning

@ tf.function
def cnn_predict(img, isTraining=False):
    # 1th convolutional layer
    conv1 = tf.nn.conv2d(img, kernel_W1, strides=[1,1,1,1], padding='SAME') + conv_b1
    conv1 = tf.nn.conv2d(conv1, skernel_W1, strides=[1,2,2,1], padding='SAME') + sconv_b1
    conv1 = tf.nn.relu(conv1)

    # 2th convolutional layer
    conv2 = tf.nn.conv2d(conv1, kernel_W2, strides=[1,1,1,1], padding='SAME') + conv_b2
    conv2 = tf.nn.conv2d(conv2, skernel_W2, strides=[1,2,2,1], padding='SAME') + sconv_b2
    conv2 = tf.nn.relu(conv2)

    # 3th convolutional layer
    conv3 = tf.nn.conv2d(conv2, kernel_W3, strides=[1,1,1,1], padding='SAME') + conv_b3
    conv3 = tf.nn.conv2d(conv3, skernel_W3, strides=[1,2,2,1], padding='SAME') + sconv_b3
    conv3 = tf.nn.relu(conv3)

    # 4th convolutional layer
    conv4 = tf.nn.conv2d(conv3, kernel_W4, strides=[1,1,1,1], padding='SAME') + conv_b4
    conv4 = tf.nn.relu(conv4)

    # output layer
    output = tf.nn.avg_pool2d(conv4, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
    output = tf.reshape(output, shape=[-1, 10])

    if isTraining:
        return tf.nn.softmax(output) # use softmax to highlight max value by using math, to simplify loss calculation

    else:
        return tf.argmax(output, axis=1, output_type=tf.int32)

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
                                                  kernel_W3, conv_b3, skernel_W3, sconv_b3,
                                                  kernel_W4, conv_b4 ])

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
            if accuracy > 0.993:
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
import tensorflow as tf
a = tf.constant([[[[1,2,4]]],[[[1,2,4]]],[[[1,2,4]]]])
a = tf.reshape(a, shape=[-1,3])
print(a)
# %%
