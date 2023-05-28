from ai.cv_ai import cv_detection
from visual_odometry.vo import vo_video



def main(object):
    #image = vo_video(object)
    analytic = cv_detection(object)
    return  analytic


if __name__ == '__main__':
    anal = main('/home/servervf/case-19/cv_analytic/data/3_Trim.mp4')
    print(anal)