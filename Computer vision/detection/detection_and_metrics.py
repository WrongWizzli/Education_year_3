# ============================== 1 Classifier model ============================

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dropout, ReLU, Softmax, Flatten
    model = Sequential([     
   
        Conv2D(50, kernel_size=(3,3), input_shape=input_shape),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),

        Conv2D(100, kernel_size=(3,3)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),

        Conv2D(150, kernel_size=(3,3)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),

        Flatten(),

        Dense(250, activation='relu'),
        BatchNormalization(),

        Dense(100, activation='relu'),
        BatchNormalization(),

        Dense(50, activation='relu'),
        BatchNormalization(),

        Dense(2, activation='softmax')

    ])
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    # your code here \/
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import load_model
    data_gen = ImageDataGenerator(rotation_range=15, zoom_range=0.2, height_shift_range=0.3)
    model = get_cls_model((40, 100, 1))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit_generator(data_gen.flow(X, y, batch_size=32), steps_per_epoch=len(X) // 32, epochs=65, verbose=1)
    model.save('classifier_model.h5')
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    from tensorflow.keras.layers import Conv2D, BatchNormalization
    from tensorflow.keras.models import Sequential
    import numpy as np
    fcn_model = Sequential()
    for i, layer in enumerate(cls_model.layers):
        if 'Dense' in str(layer):
            weights = layer.get_weights()
            out_shape = weights[1].shape[0]
            if i and 'Flatten' in str(cls_model.layers[i - 1]):
                shape0 = cls_model.layers[i - 1].input_shape[1:] + (out_shape,)
                conv_weights = weights[0].reshape(shape0)
                layer = Conv2D(shape0[3], kernel_size=(shape0[0], shape0[1]), activation='relu', weights=[conv_weights, weights[1]])
                fcn_model.add(layer)
            else:
                conv_shape = (1, 1, layer.input_shape[1], out_shape)
                conv_weights = weights[0].reshape(conv_shape)
                conv_layer = Conv2D(out_shape, (1, 1), activation='relu', weights=[conv_weights, weights[1]])
                fcn_model.add(conv_layer)
        elif 'Flatten' in str(layer) or 'Dropout' in str(layer):
            pass
        elif 'Batch' in str(layer):
            fcn_model.add(BatchNormalization(weights=layer.get_weights()))
        else:
            fcn_model.add(layer)
    return fcn_model


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    from skimage.transform import resize
    from skimage.io import imsave
    import matplotlib.pyplot as plt
    import numpy as np
    threshold = 0.2
    max_size = (220, 370, 1)
    input_data = np.zeros((len(dictionary_of_images),) + max_size)
    for i, key in enumerate(dictionary_of_images.keys()):
        shape = dictionary_of_images[key].shape
        input_data[i, :shape[0], :shape[1], 0] += dictionary_of_images[key]
    detection_model.compile()
    a = detection_model.predict(input_data)
    c = {}
    for i, key in enumerate(dictionary_of_images.keys()):
        c[key] = []
        for j in range(a[i].shape[0]):
            for k in range(a[i].shape[1]):
                if a[i][j][k][0] < threshold and a[i][j][k][1] > threshold:
                    c[key].append([j * 8, k * 8, 40, 100, a[i][j][k][1]])
    num = 4
    '''
    plt.imshow(input_data[num])
    plt.show()
    plt.imshow(a[num, :,:,0], cmap='hot')
    plt.show()
    '''
    return c
    # your code here /\


# =============================== 5 IoU ========================================
def fix_box(first_bbox):
    missed_sq = 0
    if first_bbox[0] + first_bbox[2] <= 0 or first_bbox[1] + first_bbox[3] <= 0:
        for i in range(len(first_bbox)):
            first_bbox[i] = 0
    elif first_bbox[0] < 0 and first_bbox[1] < 0:
        missed_sq = -(first_bbox[2] * first_bbox[1] +
                      first_bbox[3] * first_bbox[0] +
                      first_bbox[0] * first_bbox[1])
        first_bbox[2] += first_bbox[0]
        first_bbox[3] += first_bbox[1]
        first_bbox[0] = 0
        first_bbox[1] = 0
    elif first_bbox[0] < 0:
        missed_sq = -first_bbox[0] * first_bbox[3]
        first_bbox[2] += first_bbox[0]
        first_bbox[0] = 0
    elif first_bbox[1] < 0:
        missed_sq = -first_bbox[1] * first_bbox[2]
        first_bbox[3] += first_bbox[1]
        first_bbox[1] = 0
    return first_bbox, missed_sq

def calc_iou(first_bbox, second_bbox):
    import numpy as np
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    first_bbox = [int(elem) for elem in first_bbox]
    second_bbox = [int(elem) for elem in second_bbox]
    first_bbox, missed_sq = fix_box(first_bbox)
    first = np.zeros((max(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2]),
                      max(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])), dtype='bool')
    second = np.zeros(first.shape, dtype='bool')
    first[first_bbox[0]:first_bbox[0] + first_bbox[2], first_bbox[1]:first_bbox[1] + first_bbox[3]] = True
    second[second_bbox[0]:second_bbox[0] + second_bbox[2], second_bbox[1]:second_bbox[1] + second_bbox[3]] = True
    return np.sum(np.logical_and(first, second)) / (np.sum(np.logical_or(first, second)) + missed_sq)
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    import numpy as np
    iou_thr = 0.5
    num_boxes_gt = 0
    tp = []
    fp = []
    for key in pred_bboxes.keys():

        pred = pred_bboxes[key]
        gt = gt_bboxes[key]
        pred = sorted(pred, reverse=True, key=lambda x: x[-1])
        num_boxes_gt += len(gt)
        for i in range(len(gt)):
            iou = -1
            j_fit = -1
            for j in range(len(pred)):
                
                new_iou = calc_iou(pred[j], gt[i])
                if new_iou >= iou_thr and new_iou > iou:
                    iou = new_iou
                    j_fit = j
            if iou != -1:
                iou = -1
                tp.append(pred[j_fit])
                del pred[j_fit]
                j_fit = -1
        for j in range(len(pred)):
            fp.append(pred[j])

    fp += tp

    fp = sorted(fp, key=lambda x:x[-1])
    tp = sorted(tp, key=lambda x:x[-1])
    fp_cnt = 0
    tp_cnt = 0

    tp_len = len(tp)
    fp_len = len(fp)
    precision_dots = []
    for i in range(fp_len):
        fp_cnt = fp_len - i

        j = tp_len - 1
        tp_cnt = 0
        while j >= 0 and tp[j][-1] >= fp[i][-1]:
            tp_cnt += 1
            j -= 1
        precision_dots.append([tp_cnt / num_boxes_gt, tp_cnt / fp_cnt])
        tp_cnt = 0
    S = 0
    for i in range(0, len(precision_dots) - 1):
        S += (precision_dots[i + 1][1] + precision_dots[i][1]) * abs(precision_dots[i + 1][0] - precision_dots[i][0]) / 2
    S += (precision_dots[-1][1] + 1) / 2 * precision_dots[-1][0]
    return S
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.3):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    nms_dict = {}
    for key in detections_dictionary.keys():

        detections = detections_dictionary[key]
        detections = sorted(detections, reverse=True, key=lambda x: x[-1])
        j = 0
        while j < len(detections):
            i = j + 1
            while i < len(detections):
                if calc_iou(detections[j], detections[i]) >= iou_thr:
                    del detections[i]
                else:
                    i += 1
            j += 1
        nms_dict[key] = detections
    # your code here \/
    return nms_dict
    # your code here /\
