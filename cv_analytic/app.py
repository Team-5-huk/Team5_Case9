from visual_odometry.vo import vo_video
import requests
from time import sleep
import json

def main():
    while True:
        print('start...')
        r = requests.get("http://87.244.7.150:8000/api/samolet/get_unchaked/")
        data = r.json()
        print('Новый ивент:')
        print(data)
        if data:
            print('Аналитика:')
            #/home/servervf/case-19/backend/backend/media/checks
            video_path = f"/home/servervf/case-19/backend/backend/media/{data['video'].replace('/backend-media/','')}"
            analytic = vo_video(video_path)
            print('Detected_objects is:')
            print(analytic['Door_ready'])
            print(analytic['Ceiling_ready'])
            print(analytic.keys())
            print(analytic['Detected_objects'])
            ##files = {'analys_image': analytic['Trajectory_path']}
            sleep(3)
            #with  open(r'/home/servervf/case-19/cv_analytic/trajMap.png','rb') as fb:
            #    content_bytestring = fb.read()
            sleep(1)
            files = {'analys_image': open(r'/home/servervf/case-19/cv_analytic/trajMap.png','rb')}

            del analytic['Trajectory_path']
            data_json = {}
            data_json["is_analysed"] = True
            data_json["analys_square"] = 30
            data_json["flat"] = data['flat']
            data_json["analysis"] = {
                'Door_ready': analytic['Door_ready'],
                'Ceiling_ready': analytic['Ceiling_ready'],
                'Ready_precentage': analytic['Ready_precentage'],
                'Detected_objects': analytic['Detected_objects'],
                'Floor_ready': analytic['Floor_ready']
                #'Defects': 0
            }
            print(data_json)
            print('analysis is:')
            print(analytic)
            print('sending data...')
            r = requests.patch(f"http://87.244.7.150:8000/api/samolet/checks/{data['id']}/", json=data_json)
            sleep(5)
            r = requests.patch(f"http://87.244.7.150:8000/api/samolet/checks/{data['id']}/", files=files)
            print('status code is')
            print(r.status_code)
            print(r.json())
        sleep(5)
        print('I am sleeping...')

if __name__ == '__main__':
    main()# запуск