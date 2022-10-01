from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from glob import glob
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import shuffle
import numpy as np

BATCH_SIZE = 100
BASIC_IMG_SHAPE = (100, 100, 3)

def read_data(dir_name = './new_data/', gt_res_name='new_gt.csv'):
    import csv
    gt_res = []
    with open(dir_name + gt_res_name, 'r') as read:
        reader = csv.reader(read)
        for row in reader:
            gt_res.append(np.array([float(elem) for elem in row]))
    return np.array(gt_res).astype('float32')
def save_data(images, results, dir_name = './prepared_data/'):
    import csv
    from skimage.io import imsave
    with open(dir_name + 'res.csv', 'w') as write:
        writer = csv.writer(write)
        writer.writerows(results)
    for i, img in enumerate(images):
        imsave(dir_name + '0' * (5 - len(str(i))) + str(i) + 'my.jpg', img)
def train_detector(train_gt, train_img_dir, fast_train=False):

    epochs = 150
    model = Sequential([

        Conv2D(15, (5, 5), activation='relu', kernel_regularizer='l1_l2', input_shape=BASIC_IMG_SHAPE),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(25, (3, 3), activation='relu', kernel_regularizer='l1_l2'),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(50, (3, 3), activation='relu', kernel_regularizer='l1_l2'),
        MaxPooling2D(),
        BatchNormalization(),

        Flatten(),
        Dropout(0.2),
        Dense(400, activation='relu', kernel_regularizer='l1_l2'),
        Dropout(0.1),
        Dense(150, activation='relu', kernel_regularizer='l1_l2'),
        Dense(28, activation='relu', kernel_regularizer='l1_l2')
    ])

    model.compile(loss='mse', optimizer=Adam(), metrics=['mse'])

    img_names = sorted(glob(train_img_dir + '/' + '*.jpg'))
    img_data = np.zeros((len(img_names),) + BASIC_IMG_SHAPE)

    gt_results = np.zeros((len(img_names), 28))

    num = 0
    for i in range(len(img_names)):

        img = imread(img_names[i]).astype('float32')

        if len(img.shape) != 3:
            img = np.dstack((img, img, img))

        img_data[i] = resize(img, BASIC_IMG_SHAPE)

        gt_results[i] = train_gt[img_names[i][-9:]]
        gt_results[i, ::2] *= BASIC_IMG_SHAPE[1] / img.shape[1]
        gt_results[i, 1::2] *= BASIC_IMG_SHAPE[0] / img.shape[0]

        num += 1

        if fast_train and num >= 100:
            epochs = 1
            gt_results = gt_results[:100]
            img_data = img_data[:100]
            break

    print('End_of_data_reading')

    img_data = (img_data - np.mean(img_data)) / np.std(img_data)

    model.fit(img_data, gt_results, batch_size=BATCH_SIZE, epochs=epochs, verbose=1, validation_split = 0.1, shuffle = True)
    #model.save('facepoints_model.hdf5')
def detect(model, test_img_dir):

    img_names = sorted(glob(test_img_dir + '/' + '*.jpg'))
    name_offset = -9

    img_data = np.zeros((len(img_names),) + BASIC_IMG_SHAPE)
    coefs = np.zeros((len(img_names), 2))
    test_img_dir += '/'

    for i in range(len(img_names)):

        img = imread(img_names[i]).astype('float32')

        if len(img.shape) != 3:
              img = np.dstack((img, img, img))

        coefs[i][0] = img.shape[0] / BASIC_IMG_SHAPE[0]
        coefs[i][1] = img.shape[1] / BASIC_IMG_SHAPE[1]

        img_data[i] = resize(img, BASIC_IMG_SHAPE)

    img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    prediction = model.predict(img_data)

    output = {}
    
    for i in range(len(img_names)):
        prediction[i, ::2] *= coefs[i][1]
        prediction[i, 1::2] *= coefs[i][0]
        output[img_names[i][name_offset:]] = prediction[i]
    return output 
