import tensorflow as tf
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
# tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import typing

tf.config.set_visible_devices([], 'GPU')
model = hub.load(
    "/home/servervf/case-19/cv_analytic/openimages_v4_ssd_mobilenet_v2_1").signatures['default']

allowed_classes = ['Toilet',
                   'Oven',
                   'Sink',
                   'Tap',
                   'Bucket',
                   'Shelf',
                   'Chair',
                   ]

RU_NAMES = {
    'Toilet': 'Туалет',
    'Oven': 'Печь',
    'Sink': 'Кран',
    'Tap': 'Раковина',
    'Shelf': 'Шкаф',
    'Chair': 'Стул'
}

seed = 2023


def class_detection(frame: cv.Mat, list_of_rooms_with_windows: list, lst_of_doors: list, lst_of_objects: list, frame_count: int, threshold: float = 0.2) -> typing.Union[cv.Mat, bool]:
    window_count = 0
    window_score = 0
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
    (left, right, top, bottom) = [0]*4
    for i in range(boxes.shape[0]):
        if scores[i] >= threshold:
            if classes[i].numpy().decode() == 'Window':
                if scores[i] >= 0.4:
                    window_count += 1
                    window_score = int(100 * scores[i])
                    (left, right, top, bottom) = (int(boxes[i][1] * frame.shape[1]), int(boxes[i][3] * frame.shape[1]),
                                                  int(boxes[i][0] * frame.shape[0]), int(boxes[i][2] * frame.shape[0]))
            if classes[i].numpy().decode() == 'Door':
                lst_of_doors.append(
                    [frame_count, int(100 * scores[i])])
            if classes[i].numpy().decode() in allowed_classes:
                lst_of_objects.append(
                    [classes[i].numpy().decode(), frame_count, int(100 * scores[i])])
    if window_count == 1:
        list_of_rooms_with_windows.append(
            [frame_count, window_score])
        return (frame[:top], True) if frame[:top].shape[0] != 0 else (frame, False)
    return (frame, False)


def check_ceiling(frame: cv.Mat, lst_of_rooms_with_windows: list) -> None:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(
        gray, 50, 100, apertureSize=3, L2gradient=True)
    contours, _ = cv.findContours(
        canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    '''
    0 - потолок не готов
    1 - потолок готов
    '''
    lst_of_rooms_with_windows[len(
        lst_of_rooms_with_windows)-1].append(int(len(contours) <= 10))


def check_floor(frame: cv.Mat) -> None:
    ...


def check_walls(frame: cv.Mat) -> None:
    ...


def cv_detection(video: str):
    cap = cv.VideoCapture(video)
    fps = cap.get(cv.CAP_PROP_FPS)
    N_SKIPPED_FRAMES: int = fps if fps < 10 else fps // 6
    frame_count = 0
    READY_PRECENTAGE: int = 90
    lst_of_rooms_with_windows = []
    lst_of_objects = []
    lst_of_doors = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            got_window = False
            ceiling_frame = None
            if ret:
                # Здесь кадры обрабатываются, можешь сюда вставлять свой код
                # Либо сделать обработку асинхронной
                #
                #
                #
                #
                # frame_count: счетчик для кадров
                frame_count += 1
                # frame_count % N_SKIPPED_FRAMES == 0
                if frame_count % N_SKIPPED_FRAMES == 0:
                    '''
                    детекция объектов на кадре
                    '''
                    ceiling_frame, got_window = class_detection(
                        frame, lst_of_rooms_with_windows, lst_of_doors, lst_of_objects, frame_count)
                if got_window:
                    '''
                    Анализ комнаты с окном
                    '''
                    check_ceiling(ceiling_frame, lst_of_rooms_with_windows)
                    got_window = False
            else:
                break
        # Тут чтение видео заканчивается, начинается обработка данных
        
        arr_winds = np.array([np.array([lst_of_rooms_with_windows[i][0], lst_of_rooms_with_windows[i][0]])
                              for i in range(len(lst_of_rooms_with_windows))])

        
        output = AgglomerativeClustering(
            n_clusters=None, distance_threshold=200).fit_predict(arr_winds)
        
        
        for i in range(len(lst_of_rooms_with_windows)):
            lst_of_rooms_with_windows[i].append(output[i])
        
        arr_doors = np.array([np.array([lst_of_doors[i][0], lst_of_doors[i][0]])
                              for i in range(len(lst_of_doors))])
        
        door_flag = True
        if len(arr_doors) != 0:
            output = AgglomerativeClustering(
                n_clusters=None, distance_threshold=100).fit_predict(arr_doors)
            for i in range(len(lst_of_doors)):
                lst_of_doors[i].append(output[i])
            df_doors = pd.DataFrame(data=lst_of_doors, columns=[
                "Frame_count", "Score", "Class"])
        else: door_flag = False
        
        df_wind = pd.DataFrame(data=lst_of_rooms_with_windows, columns=[
            "Frame_count", "Score", "Ceiling", "Class"])

        df_objects = pd.DataFrame(data=lst_of_objects, columns=[
            "Name", "Frame_count", "Score"
        ])

        # dr
        # cr
        # toilet
        # oven

        result = {
            "Door_ready": len(df_doors['Class'].unique()) > len(df_wind['Class'].unique()) if door_flag else False,
            "Ceiling_ready": df_wind['Ceiling'].mean() >= 0.7,
            "Detected_objects": df_objects[df_objects.groupby('Name')['Score'].transform(max) == df_objects['Score']].to_dict('records')
        }

        if not result['Door_ready']:
            READY_PRECENTAGE -= 30
        if not result['Ceiling_ready']:
            READY_PRECENTAGE -= 30

        t_flag = False
        for i in result['Detected_objects']:
            if i['Name'] == 'Toilet':
                t_flag = True
            i['Name'] = RU_NAMES[i['Name']]
        if not t_flag:
            READY_PRECENTAGE -= 30
        result['Ready_precentage'] = 0 if READY_PRECENTAGE == 10 else READY_PRECENTAGE
        return result
    except Exception as e:
        print(e)
        return None
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    output = cv_detection("../data/1.mp4")
