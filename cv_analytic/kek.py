import tensorflow as tf
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
# tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import time
from sklearn.cluster import KMeans
import pandas as pd

tf.config.set_visible_devices([], 'GPU')
model = hub.load(
    "openimages_v4_ssd_mobilenet_v2_1").signatures['default']

allowed_classes = {
    'Window': 0,
    'Toilet': False,
    'Oven': False,
    'Sink': False,
    'Tap': False,
    'Door': 0,
    'Bucket': False,
    'Shelf': False,
    'Chair': False,
}

_allowed_classes = ['Window',
                    'Toilet',
                    'Oven',
                    'Sink',
                    'Tap',
                    'Door',
                    'Bucket',
                    'Shelf',
                    'Chair',
                    ]
seed = 2023


def class_detection(frame: cv.Mat, list_of_rooms_with_windows: list, frame_count: int, threshold: float = 0.3) -> cv.Mat:
    window_count = 0
    new_frame = frame
    new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2RGB)
    new_frame = cv.resize(new_frame, dsize=(
        256, 256), interpolation=cv.INTER_CUBIC)
    new_frame = cv.normalize(new_frame.astype(
        'float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    np_image = np.asarray(new_frame)
    np_image = np.expand_dims(np_image, axis=0)
    np_image = np_image.astype(np.float32)
    tensor1 = tf.convert_to_tensor(np_image)
    result = model(tensor1)
    boxes = result["detection_boxes"]
    classes = result["detection_class_entities"]
    scores = result["detection_scores"]
    for i in range(boxes.shape[0]):
        if scores[i] >= threshold:
            try:
                if classes[i].numpy().decode() == 'Window':
                    window_count += 1
                # Test
                if allowed_classes[classes[i].numpy().decode()] == False:
                    allowed_classes[classes[i].numpy(
                    ).decode()] = True
                    # print(
                    #    f"{str(classes[i].numpy().decode())}: {int(100 * scores[i])}")
                if classes[i].numpy().decode() in _allowed_classes:
                    (left, right, top, bottom) = (int(boxes[i][1] * frame.shape[1]), int(boxes[i][3] * frame.shape[1]),
                                                  int(boxes[i][0] * frame.shape[0]), int(boxes[i][2] * frame.shape[0]))
                    frame = cv.rectangle(
                        frame, (right, top), (left,  bottom), (255, 0, 0), 2)
                    cv.putText(
                        frame, f"{str(classes[i].numpy().decode())}: {int(100 * scores[i])}", (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except KeyError as e:
                ...
                # print(
                #    f"{classes[i].numpy().decode()}, {int(100 * scores[i])} \t not allowed class")
    if window_count == 1:
        list_of_rooms_with_windows.append(
            [frame_count, 100 * scores[i], frame])
    return frame


def check_floor(frame: cv.Mat) -> None:

    ...


def check_walls(frame: cv.Mat) -> None:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # blur
    blur = cv.GaussianBlur(gray, (7, 7), 0)

    # canny
    canny = cv.Canny(blur, 127, 200, apertureSize=3, L2gradient=True)

    # threshold
    # _, threshold = cv.threshold(canny, 127, 200, cv.THRESH_BINARY)

    # contours
    # contours, hierch = cv.findContours(
    #    threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return canny


if __name__ == '__main__':
    cap = cv.VideoCapture('data/4.mp4')
    start = time.time()
    fps = cap.get(cv.CAP_PROP_FPS)
    N_SKIPPED_FRAMES: int = fps if fps < 10 else fps // 6
    frame_count = 0

    lst_of_rooms_with_windows = []

    try:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % N_SKIPPED_FRAMES == 0:
                    '''
                    детекция объектов на кадре
                    '''
                    _ = class_detection(
                        frame, lst_of_rooms_with_windows, frame_count)
                    # cv.imshow("frame", frame)
                    # if cv.waitKey(0) & 0xFF == ord('q'):
                    #    break
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # blur = cv.GaussianBlur(gray, (1, 1), 0)
                canny = cv.Canny(
                    gray, 50, 200, apertureSize=3, L2gradient=True)
                _, threshold = cv.threshold(canny, 127, 200, cv.THRESH_BINARY)
                # frame = check_walls(frame)
                #cv.imshow("origin", frame)
                #cv.imshow("canny", canny)
                #if cv.waitKey(0) & 0xFF == ord('q'):
                #    break
            else:
                break
        for i in range(len(lst_of_rooms_with_windows)):
            print(lst_of_rooms_with_windows[i][0])
    except Exception as e:
        print(e)
    finally:
        print(allowed_classes)
        end = time.time()
        print(f"Time passed:\t{end - start} seconds")
        cap.release()
        cv.destroyAllWindows()
