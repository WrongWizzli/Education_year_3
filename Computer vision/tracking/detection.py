import numpy as np
import os

from PIL import Image
from skimage.transform import resize
from skimage import io
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from config import VOC_CLASSES, bbox_util, model
from utils import get_color


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    # Write code here
    # First, convert frame to float and resize to 300x300
    basic_shape = frame.shape
    frame = resize(frame.astype('float32'), (300, 300, 3))
    # Then use preprocess_input, model.predict and bbox_util.detection_out
    frame = preprocess_input(frame)
    frame_wrap = np.zeros((1,) + frame.shape)
    frame_wrap[0] = frame
    results = model.predict(frame_wrap)
    results = bbox_util.detection_out(results)

    # Select detections with confidence > min_confidence

    final_results = []
    mask = np.array([0, 2, 3, 4, 5])
    for result in results[0]:
        if result[1] > min_confidence:
            final_results.append(result[mask])

    results = final_results
    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indeces = [i for i, l in enumerate(result_labels) if VOC_CLASSES[l - 1] in labels]
        results = results[indeces]

    # Remove confidence column from result

    # Resize detection coords to input image shape.
    # Didn't you forget to save it before resize?
    for i in range(len(results)):
        results[i][1] *= basic_shape[1]
        results[i][2] *= basic_shape[0]
        results[i][3] *= basic_shape[1]
        results[i][4] *= basic_shape[0]
    # Return result
    return detection_cast(np.array(results, dtype=np.int32))


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()

    # Write code here

    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, 'data', 'test.png'))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == '__main__':
    main()
