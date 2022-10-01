from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Softmax, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling1D, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from glob import glob
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import shuffle
import numpy as np
from tensorflow import one_hot


def read_data(gt):
    fixed = np.empty(len(gt))
    for i, filename in enumerate(sorted(gt.keys())):
        fixed[i] = gt[filename]
    return fixed

basic_shape = (224, 224, 3)

def train_classifier(train_gt, train_img_dir, fast_train=False):

    epochs = 150
    batch_size=32
    steps = 70
    cnt = 20
    if fast_train:
        epochs=1
        batch_size=1
        steps=2
    model = Sequential()
    base_model = Xception(input_tensor=Input(shape=basic_shape), include_top=False)
    base_model.trainable = False
    while cnt > 0:
        base_model.layers[-cnt].trainable = True
        cnt -= 1

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(400))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Dense(50, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


    img_names = sorted(glob(train_img_dir + '/' + '*.jpg'))
    gt_results = read_data(train_gt)
    img_names, gt_results = shuffle(img_names, gt_results)
    images = np.empty((len(img_names),) + basic_shape, dtype='float32')

    for i in range(0, len(img_names)):
        images[i] = resize(imread(img_names[i]), basic_shape)
        if fast_train and i > 100:
            images = images[:i]
            gt_results = gt_results[:i]
            break
    
    gt_results = one_hot(gt_results, 50)

    data_gen = ImageDataGenerator(zoom_range=0.15, 
                                  horizontal_flip=True, rotation_range=15, 
                                  height_shift_range=0.3, width_shift_range=0.2)

    train_gen = data_gen.flow(images, gt_results, batch_size=batch_size, shuffle=True)
    model.fit_generator(train_gen, steps_per_epoch=steps, epochs=epochs, verbose=1)

    #model.save('./gdrive/My Drive/birds_model.hdf5')
    return model

def classify(model, test_img_dir):

    img_names = sorted(glob(test_img_dir + '/' + '*.jpg'))
    test_img_dir += '/'
    images = np.empty((len(img_names),) + basic_shape, dtype='float32')

    for i in range(0, len(img_names)):
        images[i] = resize(imread(img_names[i]), basic_shape)

    prediction = model.predict(images)

    output = {}
    for i in range(len(img_names)):
        output[img_names[i].rsplit(sep='/', maxsplit=1)[-1]] = np.argmax(prediction[i])

    return output

