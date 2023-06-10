import os
import typing
import numpy as np
import cv2
import tensorflow as tf
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
# tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from visual_odometry.pose_evaluation_utils import rot2quat

tf.config.set_visible_devices([], 'GPU')
model = hub.load(
    "/home/servervf/case-19/cv_analytic/openimages_v4_ssd_mobilenet_v2_1").signatures['default']

# 
# /home/servervf/case-19/cv_analytic/openimages_v4_ssd_mobilenet_v2_1
ALLOWED_CLASSES = ['Toilet',
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


fx, fy, cx, cy = [0, 0, 240.0, 340]

def check_ceiling(frame: cv2.Mat, lst_of_rooms_with_windows: list) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(
        gray, 50, 100, apertureSize=3, L2gradient=True)
    contours, _ = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    0 - потолок не готов
    1 - потолок готов
    '''
    lst_of_rooms_with_windows[len(
        lst_of_rooms_with_windows)-1].append(int(len(contours) <= 10))

def check_floor(frame: cv2.Mat, lst_of_rooms_with_windows: list, frame_count) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(
        gray, 50, 100, apertureSize=3, L2gradient=True)
    contours, _ = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lst_of_rooms_with_windows[len(
        lst_of_rooms_with_windows)-1].append(int(len(contours) <= 25))
    
def class_detection(frame: cv2.Mat, list_of_rooms_with_windows: list, lst_of_doors: list, lst_of_objects: list, frame_count: int, threshold: float = 0.2) -> typing.Union[cv2.Mat, bool]:
    window_count = 0
    window_score = 0
    new_frame = frame
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    new_frame = cv2.resize(new_frame, dsize=(
        256, 256), interpolation=cv2.INTER_CUBIC)
    new_frame = cv2.normalize(new_frame.astype(
        'float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
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
            if classes[i].numpy().decode() in ALLOWED_CLASSES:
                lst_of_objects.append(
                    [classes[i].numpy().decode(), frame_count, int(100 * scores[i])])
    if window_count == 1:
        if frame[:top].shape[0] != 0: 
            list_of_rooms_with_windows.append(
                [frame_count, window_score])
            return (frame[:top], frame[bottom:], True)
    return (frame, None, False)



def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3, 4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])
        try:
            reprojected_pt = np.matmul(P, pt_3d)
            reprojected_pt /= reprojected_pt[2]
            print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            rep_error.append(pt_2d - reprojected_pt[0:2])
        except Exception as e:
            print(e)


def vo_video(object):
    # создадим объект VideoCapture для захвата видео
    video = cv2.VideoCapture(object)


    # Если не удалось открыть файл, выводим сообщение об ошибке
    if video.isOpened() == False:
        print('Не возможно открыть файл')
        return '', {}

    fps = video.get(cv2.CAP_PROP_FPS)
    N_SKIPPED_FRAMES: int = fps if fps < 10 else fps // 6
    frame_count = 0
    READY_PRECENTAGE: int = 0
    lst_of_rooms_with_windows = []
    lst_of_objects = []
    lst_of_doors = []
    result = {}

    global prev_img
    trajMap = np.zeros((1500, 1500, 3), dtype=np.uint8)
    for i in range(1500):
        for j in range(1500):
            trajMap[i, j] = [255, 255, 255]
    out_pose_file = './' + 'traj_est.txt'

    i = int()

    K = []
    row1 = [200, 0, 240]
    row2 = [0, 300, 340]
    row3 = [0, 0, 1]
    K.append(row1)
    K.append(row2)
    K.append(row3)

    try:
        # Пока файл открыт
        while video.isOpened():
            # поочередно считываем кадры видео
            fl, curr_img = video.read()
            got_window = False
            ceiling_frame = None
            floor_frame = None
            # если кадры закончились, совершаем выход
            if curr_img is None:
                break

            frame_count += 1

            if frame_count % N_SKIPPED_FRAMES == 0:
                    '''
                    детекция объектов на кадре
                    '''
                    ceiling_frame, floor_frame, got_window = class_detection(
                        curr_img, lst_of_rooms_with_windows, lst_of_doors, lst_of_objects, frame_count)
            if got_window:
                '''
                Анализ комнаты с окном
                '''
                check_ceiling(ceiling_frame, lst_of_rooms_with_windows)
                check_floor(floor_frame, lst_of_rooms_with_windows, frame_count)
                got_window = False

            if i == 0:
                curr_R = np.eye(3)
                curr_t = np.array([0, 0, 0])
            else:
                # ====================== Используйте функцию ORB для сопоставления функций =====================#
                # создавать объекты ORB
                # orb = cv2.ORB_create(10000) xfeatures2d.SIFT_create()
                sift = cv2.xfeatures2d.SIFT_create(10000)

                kp1, des1 = sift.detectAndCompute(prev_img, None)
                kp2, des2 = sift.detectAndCompute(curr_img, None)
                # найти ключевые точки и дескрипторы с ORB
                # kp1, des1 = orb.detectAndCompute(prev_img, None)
                # kp2, des2 = orb.detectAndCompute(curr_img, None)

                # использовать подборщик грубой силы
                bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)

                # Совпадение с дескрипторами ORB
                if des1 is None or des2 is None:
                    continue
                matches = bf.match(des1, des2)

                # Сортировать совпадающие ключевые точки в порядке соответствия расстоянию
                # так что лучшие матчи вышли на фронт
                matches = sorted(matches, key=lambda x: x.distance)

                img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[0:5000], None)
                #cv2.imshow('feature matching', img_matching)

                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                if pts1 is None or pts2 is None:
                    continue

                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                try:
                    rows, cols = F.shape
                    if rows != 3 or cols != 3:
                        continue
                except AttributeError:
                    #print("F not found")
                    continue

                E = np.matmul(np.matmul(np.transpose(K), F), K)

                # вычислить основную матрицу
                # E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999,
                #                               threshold=1)
                try:
                    rows, cols = E.shape
                    if rows != 3 or cols != 3:
                        continue
                except AttributeError:
                    print("E not found")
                    continue

                pts1 = pts1[mask.ravel() == 1]
                pts2 = pts2[mask.ravel() == 1]
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

                # получить движение камеры
                R = R.transpose()
                t = -np.matmul(R, t)

                # получаем конечное движение
                if i == 1:
                    curr_R = R
                    curr_t = t
                else:
                    curr_R = np.matmul(prev_R, R)
                    curr_t = np.matmul(prev_R, t) + prev_t

                if i > 10:
                    Rt1 = np.hstack((prev_R, prev_t.reshape(3, 1)))
                    Rt2 = np.hstack((curr_R, curr_t.reshape(3, 1)))

                    # построение облака 3-D точек
                    pts4D = cv2.triangulatePoints(Rt1, Rt2, pts1.T, pts2.T).T

                    # convert from homogeneous coordinates to 3D
                    pts3D = pts4D[:, :3] / np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

                    # plot with matplotlib
                    Ys = pts3D[:, 0]
                    Zs = pts3D[:, 1]
                    Xs = pts3D[:, 2]

                    pts3D = pts3D

                    #for k in pts3D:
                    #    if curr_t[0] - 50 < k[0] + curr_t[0] < curr_t[0] + 50 and -50 < k[1] < 50 and curr_t[2] - 50 < \
                    #            k[2] + curr_t[2] < curr_t[2] + 50:
                    #        trajMap = cv2.circle(trajMap, (int((k[0] + curr_t[0]) * 10) + (int(3000 / 2)),
                    #                                       int((k[2] + curr_t[2]) * 10) + (int(3000 / 2))), 2,
                    #                             (0, 0, 255), -1)

                cur_t = curr_t

                curr_img_kp = cv2.drawKeypoints(curr_img, kp2, None, color=(0, 255, 0), flags=0)
                #cv2.imshow('keypoints from current image', curr_img_kp)

            prev_img = curr_img
            # сохранить текущую позу
            [tx, ty, tz] = [curr_t[0], curr_t[1], curr_t[2]]
            qw, qx, qy, qz = rot2quat(curr_R)
            with open(out_pose_file, 'a') as f:
                f.write('%f %f %f %f %f %f %f %f\n' % (0.0, tx, ty, tz, qx, qy, qz, qw))

            prev_R = curr_R
            prev_t = curr_t

            # нарисуйте предполагаемую траекторию (синий) и траекторию gt (красный)
            offset_draw = (int(1500 / 2))
            cv2.circle(trajMap, (int(curr_t[0] * 10) + offset_draw, int(curr_t[2] * 10) + offset_draw), 1, (255, 0, 0),
                       2)
            #cv2.imshow('Trajectory', trajMap)
            cv2.waitKey(1)
            i = i + 1


        # Ан�
        windows_flag = True
        if len(lst_of_rooms_with_windows) != 0:
            arr_winds = np.array([np.array([lst_of_rooms_with_windows[i][0], lst_of_rooms_with_windows[i][0]])
                              for i in range(len(lst_of_rooms_with_windows))])

        
            output = AgglomerativeClustering(
            n_clusters=None, distance_threshold=200).fit_predict(arr_winds)
        
        
            for i in range(len(lst_of_rooms_with_windows)):
                lst_of_rooms_with_windows[i].append(output[i])
        
        else: windows_flag = False


        door_flag = True
        if len(lst_of_doors) != 0:
            arr_doors = np.array([np.array([lst_of_doors[i][0], lst_of_doors[i][0]])
                              for i in range(len(lst_of_doors))])
        
            output = AgglomerativeClustering(
                n_clusters=None, distance_threshold=100).fit_predict(arr_doors)
            for i in range(len(lst_of_doors)):
                lst_of_doors[i].append(output[i])
            df_doors = pd.DataFrame(data=lst_of_doors, columns=[
                "Frame_count", "Score", "Class"])
        else: door_flag = False
        
        
        df_wind = pd.DataFrame(data=lst_of_rooms_with_windows, columns=[
            "Frame_count", "Score", "Ceiling", "Floor", "Class"])
        
        df_objects = pd.DataFrame(data=lst_of_objects, columns=[
            "Name", "Frame_count", "Score"
        ])

        # dr
        # cr
        # toilet
        # oven

        count_classes = len(df_wind['Class'].unique())
        a = df_wind.groupby('Class')['Ceiling'].mean()
        b = df_wind.groupby('Class')['Floor'].mean()
        ceiling_c = 0
        floor_c = 0
        for i in range(count_classes):
            if a[i] >= 0.7:
                ceiling_c += 1
            if b[i] >= 0.7:
                floor_c += 1
        ceiling_ready_precentage = int((ceiling_c / count_classes) * 100)
        floor_ready_precentage = int((floor_c / count_classes) * 100)
        result = {
            "Door_ready": bool(len(df_doors['Class'].unique()) > len(df_wind['Class'].unique())) if door_flag else False,
            "Ceiling_ready": bool(df_wind['Ceiling'].mean() >= 0.7) if windows_flag else False,
            "Ceiling_ready_precentage": ceiling_ready_precentage,
            "Floor_ready_precentage": floor_ready_precentage,
            "Floor_ready" : bool(df_wind['Floor'].mean() >= 0.7) if windows_flag else False,
            "Detected_objects": df_objects[df_objects.groupby('Name')['Score'].transform(max) == df_objects['Score']].to_dict('records')
        }

        if not result['Door_ready']:
            READY_PRECENTAGE += 22
        if not result['Ceiling_ready']:
            READY_PRECENTAGE += 22 * ceiling_ready_precentage / 100.0
            READY_PRECENTAGE += 22 * floor_ready_precentage / 100.0

        t_flag = False
        for i in result['Detected_objects']:
            if i['Name'] == 'Toilet':
                t_flag = True
            i['Name'] = RU_NAMES[i['Name']]
        if not t_flag:
            READY_PRECENTAGE += 22
        result['Ready_precentage'] = 10 if READY_PRECENTAGE == 0 else READY_PRECENTAGE
    except Exception as e:
        print(e)
    finally:
        video.release()
        cv2.destroyAllWindows()
        print("################################")
        traj_name = 'trajMap.png'
        cv2.imwrite(traj_name, trajMap)
        ret, jpeg = cv2.imencode('.jpg', trajMap)
        result['Trajectory_path'] = jpeg.tobytes()
        return result

if __name__ == '__main__':
    import time
    start = time.time()
    output = vo_video("../data/7.MP4")
    print(time.time() - start)
    print(output)
