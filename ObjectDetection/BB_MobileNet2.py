# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
# ==================== load data =============================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import matplotlib.patches as patches
import shutil

batchSize = 32
class_names = ['Background', 'Me', 'BB']
tensorboard_path = "./BB_tensorboard"

def parse_files(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32) / 127.5 - 1
        return image, label

def loadData(batchSize):
    dsDir = './BB_Faces/'

    labels = tf.io.parse_tensor(tf.io.read_file(dsDir+'labels.ds'), tf.float32).numpy()
    class_labels, bbox_labels = np.split(labels, [3], axis=1) #split into 2 labels[(1000,3),(1000,4)]

    ds_files = tf.data.Dataset.list_files(dsDir+'images/*.jpg', shuffle=False)
    ds_files = list(ds_files.as_numpy_iterator())

    ds = tf.data.Dataset.from_tensor_slices((ds_files, (class_labels, bbox_labels)))
    ds = ds.shuffle(buffer_size=len(labels), seed=123).map(parse_files)
    ds = ds.prefetch(buffer_size=64)
    print('data size: '+ str(len(labels)))

    ds_train = ds.take(1800)
    ds_test = ds.skip(1800).batch(batchSize)

    return ds_train, ds_test

def displayData(ds):
    plt.figure(figsize=(10, 10))
    for images, (class_labels, bbox_labels) in ds.take(1):
        for i in range(9):
            plt.axis("off")
            axes = plt.subplot(3, 3, i + 1)

            img = (images[i].numpy()+1)/2
            plt.imshow(img)

            class_label = class_labels[i]
            plt.title(class_names[tf.argmax(class_label)])

            [h, w, _] = img.shape
            [x1, y1, x2, y2] = np.multiply(bbox_labels[i], [w, h, w, h])

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            axes.add_patch(rect)             

def data_shift(image, labels):
    classes, bbox = labels
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    if classes[0] == 0:
        dx = tf.random.uniform((1,), -x1*0.99, (1-x2)*0.99)[0]
        dy = tf.random.uniform((1,), -y1*0.99, (1-y2)*0.99)[0]
        bbox = [x1+dx, y1+dy, x2+dx, y2+dy]
    else:
        range = 0.1
        dx = tf.random.uniform((1,), -range, range)[0]
        dy = tf.random.uniform((1,), -range, range)[0]
        bbox = [x1, y1, x2, y2]
    
    h,w,c = image.shape
    image = tfa.image.translate(image, [dx*w, dy*h], interpolation='nearest', fill_mode='nearest')

    return image, (classes, bbox)

def data_augmentation(image, labels):
    classes, bbox = labels
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    rand = tf.random.uniform((1,), 0, 1)
    if rand[0] > 0.5:
        image = tf.image.flip_left_right(image)
        if classes[0] == 0:
            bbox = [1-x2, y1, 1-x1, y2]
        else:
            bbox = [x1, y1, x2, y2]
    else:
            bbox = [x1, y1, x2, y2]
    
    # brightness
    # image = (image + 1) / 2
    # image = image ** tf.random.uniform((1,), 0.7, 1.1)[0]
    # image = image * 2 - 1

    return image, (classes, bbox)

ds_train, ds_test = loadData(batchSize=batchSize)
# displayData(ds_train.take(9).batch(batchSize)) 

# %%
# ==================== train & valid data =============================
mse = tf.keras.losses.MeanSquaredError()

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0:1], box1[:,1:2], box1[:,2:3], box1[:,3:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0:1], box2[:,1:2], box2[:,2:3], box2[:,3:4]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:,0:1] - box1[:,2:3] / 2, box1[:,0:1] + box1[:,2:3] / 2
        b1_y1, b1_y2 = box1[:,1:2] - box1[:,3:4] / 2, box1[:,1:2] + box1[:,3:4] / 2
        b2_x1, b2_x2 = box2[:,0:1] - box2[:,2:3] / 2, box2[:,0:1] + box2[:,2:3] / 2
        b2_y1, b2_y2 = box2[:,1:2] - box2[:,3:4] / 2, box2[:,1:2] + box2[:,3:4] / 2

    # Intersection area
    inter_w = tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1)
    inter_y = tf.minimum(b1_y2, b2_y2) - tf.maximum(b1_y1, b2_y1)
    inter = tf.maximum(inter_w, 0) * tf.maximum(inter_y, 0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + 1e-16

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
                alpha = tf.stop_gradient(v / (1 - iou + v))
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def bboxRegressionLoss(y_true, y_pred):
    # extract third column: width vector
    mask = y_true[:,2:3] > 0
    mask = tf.reshape(mask, shape=(-1,))
    y_true = tf.boolean_mask(y_true, mask) # remove rows using mask
    y_pred = tf.boolean_mask(y_pred, mask)
   
    loss = mse(y_true, y_pred) #+ tf.reduce_mean(1 - bbox_iou(y_true, y_pred, x1y1x2y2=True, CIoU=True))
     
    return loss

def create_model(input_shape):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False
    base_model.training = False # keep all BatchNorm layers in inference mode, no matter freezing or unfreezing

    # base
    inputs = tf.keras.Input(shape=input_shape)
    base_output = base_model(inputs)

    # classification head
    h1 = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1 = tf.keras.layers.Dense(128, activation="relu")(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    class_head = tf.keras.layers.Dense(3, name='class_out')(h1)

    # localization head, with 3 FC with width decreasing smoothly is the best
    h2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation="relu")(base_output)
    h2 = tf.keras.layers.MaxPooling2D(strides=2)(h2)
    h2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation="relu")(h2)
    h2 = tf.keras.layers.AveragePooling2D(strides=2)(h2)
    h2 = tf.keras.layers.Flatten()(h2)
    bbox_head = tf.keras.layers.Dense(4, name='bbox_out', activation='sigmoid')(h2)

    model = tf.keras.Model(inputs, outputs=[class_head, bbox_head])
    optimizer = tf.keras.optimizers.Adam(init_learning_rate)
    losses = [tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              bboxRegressionLoss]
    
    model.compile(optimizer=optimizer, loss=losses) 
    return model, base_model

def learningRate(epoch):
    return init_learning_rate*0.94**epoch

init_learning_rate = 0.0002  #0.00008 adam-iou:0.0003, RMSprops-iou:0.0001
lr_cb = tf.keras.callbacks.LearningRateScheduler(learningRate)

board_cb = tf.keras.callbacks.TensorBoard(
    log_dir=tensorboard_path,
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",
) 

def train_model(model, ds_train, initial_epoch, epochs):
    final_epoch = initial_epoch+epochs
    for e in range(initial_epoch, final_epoch):
        x = ds_train.shuffle(len(ds_train), seed=e*123).map(data_augmentation).batch(batchSize)
        model.fit(x=x, epochs=e+1, initial_epoch=e, 
                callbacks=[board_cb,lr_cb], # board_cb,
                validation_data=ds_test, validation_freq=1,
                verbose=1)

# =========== start training ================
shutil.rmtree(tensorboard_path, ignore_errors=True)
warm_up_epoch = 20
input_shape = list(ds_train.take(1).as_numpy_iterator())[0][0].shape
model, base_model = create_model(input_shape)
train_model(model, ds_train, 0, warm_up_epoch)

# =========== fine-tuned training ================
epochs = 70
lastUnfreezeCount = 88 # start unfreezing from the last layer
for layer in base_model.layers[:len(base_model.layers)-lastUnfreezeCount]:
    layer.trainable = False

train_model(model, ds_train, warm_up_epoch, epochs)
model.save('./Models/BB_MobileNet2.h5', overwrite=True)



# %%
# ==================== test with opencv =============================
import cv2
import numpy as np
import tensorflow as tf

isCenterForm = False
color = (0,255,0)
scale = 0.25
winName = 'BB Classifier'
model = tf.keras.models.load_model('./Models/BB_MobileNet2.h5', compile=False)
class_names = ['Background', 'Me', 'BB']

def predict(frame):
    (height, width, c) = frame.shape
    h = int(height*scale)
    w = int(width*scale)
    img = tf.image.resize(frame, size=(h, w)).numpy()/127.5 - 1
    img = tf.reshape(img, shape=(1, h, w, 3))
    predict_class, predict_bbox = model.predict(img)

    name = class_names[tf.argmax(predict_class, axis=1).numpy()[0]]
    cv2.putText(frame, name, (40,150), cv2.FONT_HERSHEY_COMPLEX, 1.3, color, 1)
    
    if True:
        if isCenterForm:
            [x,y,hw,hh] =  np.multiply(predict_bbox[0], [width, height, width*0.5, height*0.5])
            top_left = (int(x-hw) , int(y-hh))
            bottom_right = (int(x+hw) , int(y+hh))
            cv2.rectangle(frame, pt1=top_left, pt2=bottom_right, color=color, thickness=1)
        else:
            [x1,y1,x2,y2] =  np.multiply(predict_bbox[0], [width, height, width, height])
            top_left = (int(x1), int(y1))
            bottom_right = (int(x2), int(y2))
            cv2.rectangle(frame, pt1=top_left, pt2=bottom_right, color=color, thickness=1)
    
    cv2.imshow(winName, frame)

def videoCapture(winName, callback):
    videoCap = cv2.VideoCapture(0)
    cv2.namedWindow(winName)
    
    while(True):
        ret, frame = videoCap.read()
        if ret==True:
            frame = cv2.flip(frame, 1)
            callback(frame)

        key = cv2.waitKey(1)
        if key != -1 and key != 255:
            videoCap.release()
            cv2.destroyAllWindows()
            break

videoCapture(winName, predict)






# %%
