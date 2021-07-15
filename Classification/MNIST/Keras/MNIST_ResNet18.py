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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (imgWidth, imgHeight) = x_train[0].shape

    imGenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # horizontal_flip=True,
        # brightness_range=(0.4, 1.4),
        # rescale=1./255,
        rotation_range=8.0, 
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=7.0,
        zoom_range=(0.82,1.65),
        fill_mode='constant',
        cval=0.0, # constant value
        data_format='channels_last', #(samples, height, width, channels)
    )

    # add a channel dimension: (n, imgHeight, imgWidth) -> (n, imgHeight, imgWidth, 1), channel=1 with grey-scale images
    x_train = tf.reshape(x_train, shape=(-1, imgHeight, imgWidth, 1))
    train_gen = imGenerator.flow(x_train, y_train, batch_size=batchSize, shuffle=True)

    # x_test = imgNorm(x_test)
    x_test = tf.reshape(x_test, shape=(-1, imgHeight, imgWidth, 1))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)

    return (train_gen, ds_test)

(train_gen, ds_test) = loadData(classCount=classCount, batchSize=batchSize) 
# randShowGenImage(train_gen) 


# %%
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
# ==================== train & valid data =============================
initial_epoch_batch = 5
weight_decay = 0.0001
init_learning_rate = 0.001

def lr_scheduler(epoch):
    return init_learning_rate * 0.95 ** epoch

def bn_relu(x, useRelu=True):
    fx = layers.BatchNormalization()(x)
    if useRelu:
        fx = layers.ReLU()(fx)
    return fx

def conv(x, filterNumb, kernel_size, strides=1, use_bias=True):
    fx = layers.Conv2D(filterNumb, kernel_size, strides, padding='same', use_bias=use_bias, kernel_regularizer=l2(weight_decay))(x)
    return fx

def residual_block(x, filterNumb, isPooling=False):
    strides = 1
    shortcut = x
    bn_x = bn_relu(x)

    if isPooling:
        strides = 2
        shortcut = conv(bn_x, filterNumb, kernel_size=1, strides=strides)
    
    fx = conv(bn_x, filterNumb, kernel_size=3, strides=strides)
    fx = bn_relu(fx)
    fx = conv(fx, filterNumb, kernel_size=3)
    out = layers.Add()([shortcut, fx]) # skip
    return out

def create_resnet():
    inputs = layers.Input(shape=(28,28,1)) # 28*28
    hx = layers.BatchNormalization()(inputs)
    hx = conv(hx, 16, kernel_size=3)
    hx = residual_block(hx, 16)
    hx = residual_block(hx, 16)
    hx = residual_block(hx, 16)

    hx = residual_block(hx, 32, isPooling=True) # 14*14
    hx = residual_block(hx, 32)
    hx = residual_block(hx, 32)
    
    hx = residual_block(hx, 64, isPooling=True) # 7*7
    hx = residual_block(hx, 64)
    hx = residual_block(hx, 64)

    hx = residual_block(hx, 128, isPooling=True)
    hx = residual_block(hx, 128)
    hx = residual_block(hx, 128)

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
    opt_fn = tf.keras.optimizers.Adam(init_learning_rate)
    model = create_resnet()
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=['accuracy'])
else:
    model = tf.keras.models.load_model('./Models/Checkpoint')
    print(model.evaluate(x=ds_test, verbose=1))

# start training
lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
board_cb = tf.keras.callbacks.TensorBoard(
    log_dir='./tb_logs',
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=1,  # How often to log embedding visualizations
    update_freq="epoch",
) 
save_cb = tf.keras.callbacks.ModelCheckpoint(
    './Models/Checkpoint', monitor='val_accuracy', verbose=1, save_best_only=False,
    save_weights_only=False, mode='max', save_freq='epoch'
)

ebatch = 20
for i in range(initial_epoch_batch, 10):
    history = model.fit(x=train_gen, epochs=(i+1)*ebatch, initial_epoch=i*ebatch, 
                        callbacks=[lr_cb, save_cb, board_cb], 
                        validation_data=ds_test, validation_freq=1, verbose=1)
    display_history(history)



# %% ===================================
# Load best check point and save model
import tensorflow as tf
filePath = './Models/MNIST_ResNet-18_0.9941_best.h5'
model = tf.keras.models.load_model('./Models/Checkpoint')
model.save(filePath, overwrite=True, include_optimizer=False)
print(model.evaluate(x=ds_test, verbose=0))


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
model = tf.keras.models.load_model('./Models/MNIST_ResNet-18_0.9941_best.h5', compile=False)

def draw(event, x, y, flags, param):
    global canvas, isDrawing
    if event == cv2.EVENT_LBUTTONDOWN:
        isDrawing = True
    elif event == cv2.EVENT_MOUSEMOVE and isDrawing:
        cv2.circle(canvas, (x,y), penSize, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        isDrawing = False
        greyImg = tf.image.rgb_to_grayscale(tf.image.resize(canvas, size=(outputSize, outputSize)))
        visualInput = tf.reshape(greyImg, shape=(1,28,28,1))
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
