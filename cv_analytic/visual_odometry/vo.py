import os
import numpy as np
import cv2

from visual_odometry.pose_evaluation_utils import rot2quat

fx, fy, cx, cy = [200, 300, 240.0, 340]
def vo_video(object):
    # создадим объект VideoCapture для захвата видео
    video = cv2.VideoCapture(object)

    # Если не удалось открыть файл, выводим сообщение об ошибке
    if video.isOpened() == False:
        print('Не возможно открыть файл')

    global prev_img
    trajMap = np.zeros((3000, 3000, 3), dtype=np.uint8)
    out_pose_file = './' + 'traj_est.txt'

    i = int()
    try:
    # Пока файл открыт
        while video.isOpened():
            # поочередно считываем кадры видео
            fl, curr_img = video.read()
            # если кадры закончились, совершаем выход
            if curr_img is None:
                break

            if i == 0:
                curr_R = np.eye(3)
                curr_t = np.array([0, 0, 0])
            else:
                # ====================== Используйте функцию ORB для сопоставления функций =====================#
                # создавать объекты ORB
                orb = cv2.ORB_create(10000)

                # найти ключевые точки и дескрипторы с ORB
                kp1, des1 = orb.detectAndCompute(prev_img, None)
                kp2, des2 = orb.detectAndCompute(curr_img, None)

                # использовать подборщик грубой силы
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                # Совпадение с дескрипторами ORB
                if des1 is None or des2 is None:
                    continue
                matches = bf.match(des1, des2)

                # Сортировать совпадающие ключевые точки в порядке соответствия расстоянию
                # так что лучшие матчи вышли на фронт
                matches = sorted(matches, key=lambda x: x.distance)

                img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[0:500], None)
                #cv2.imshow('feature matching', img_matching)

                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                if pts1 is None or pts2 is None:
                    continue

                # вычислить основную матрицу
                E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999,
                                               threshold=1)
                try:
                    rows, cols = E.shape
                    if rows != 3 or cols != 3:
                        continue
                except AttributeError:
                    print("shape not found")
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

                    for k in pts3D:
                        if curr_t[0]-50 < k[0]+curr_t[0] < curr_t[0]+50 and -50 < k[1] < 50 and curr_t[2]-50 < k[2]+curr_t[2] < curr_t[2]+50:
                            trajMap = cv2.circle(trajMap, (int((k[0]+curr_t[0])*10)+(int(3000 / 2)), int((k[2]+curr_t[2])*10)+(int(3000 / 2))),  2, (0,0,255), -1)

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
            offset_draw = (int(3000 / 2))
            cv2.circle(trajMap, (int(curr_t[0]*10) + offset_draw, int(curr_t[2]*10) + offset_draw), 1, (255, 0, 0), 2)
            cv2.imshow('Trajectory', trajMap)
            cv2.waitKey(1)
            i = i + 1
    except Exception as e:
        print(e)
    finally:
        print("################################")
        cv2.imwrite('trajMap.png', trajMap)
        ret, jpeg = cv2.imencode('.jpg', trajMap)
        return jpeg.tobytes()
