import requests
from jinja2 import Template
import time

samolet_domain = r'https://samolet.ru'
samolet_url_projects = r'https://samolet.ru/api_redesign/projects/'
samolet_url_project_info = Template(r'https://samolet.ru/api_redesign/projects/{{ slug }}/')
samolet_url_project_buildings = Template(r'https://samolet.ru/v1/api/project/{{ priject_pk }}/buildings/')
samolet_url_section_chessboard = Template(r'https://samolet.ru/v1/api/flat/chessboard/?building={{ building_pk }}&')
samolet_url_sction_floors_info = Template(r'https://samolet.ru/v1/api/section/{{ section_pk }}}/floors/')

PROJECTS_DICT = {}

def get_samolet_projects():
    response = requests.get(samolet_url_projects)
    PROJECTS_DICT = {}
    for project_dict in response.json():
        pk = project_dict["pk"]
        name = project_dict["name"]
        slug = project_dict["slug"]
        url = f"{samolet_domain}{project_dict['url']}"
        latitude,longitude = project_dict["coords"]
        image_src = project_dict["image_src"]["default"]
        PROJECTS_DICT[pk] = {
            "pk": pk,
            "name": name,
            "slug": slug,
            "url": url,
            "latitude": latitude,
            "longitude": longitude,
            "image_src": image_src
        }
    return PROJECTS_DICT

def get_project_info(slug):
    response = requests.get(samolet_url_project_info.render(slug=slug)).json()
    flat_count = response["flat_count"]
    genplan_thumb = response["genplan_thumb"]
    return (flat_count,genplan_thumb)

def get_buildings_in_project(priject_pk):
    building_response_set = requests.get(samolet_url_project_buildings.render(priject_pk=priject_pk)).json()["building_set"]
    buildings_dict = []
    for building_dict in  building_response_set:
        sections = []

        chessboard_response = requests.get(samolet_url_section_chessboard.render(building_pk=building_dict["pk"])).json()
        for section in chessboard_response:
            floors = []
            seciton_dict = {
                "pk": section["pk"],
                "flats_on_floor": section["flats_on_floor"],
                "number" : section["number"],
                "floors_total": section["floors_total"],
            }
            floors_response = requests.get(samolet_url_sction_floors_info.render(section_pk=section["pk"])).json()["floor_set"]
            for floor in floors:
                curren_floor = {
                    "id": floor["id"],

                }
            sections.append(seciton_dict)
        buildings_dict.append({
        "pk": building_dict["pk"],
        "name": building_dict["name"],
        "number": building_dict["number"],
        "url": building_dict["url"],
        "plan": building_dict["plan"],
        "section_set": building_dict["section_set"],
        "section_count": building_dict["section_count"],
        "sections": sections,
        })
    return buildings_dict


if __name__ == '__main__':
    PROJECTS_DICT = get_samolet_projects()
    for pk in PROJECTS_DICT.keys():
        flat_count,genplan_thumb = get_project_info(PROJECTS_DICT[pk]["slug"])
        building_dict = get_buildings_in_project(pk)
        PROJECTS_DICT[pk]["flat_count"] = flat_count
        PROJECTS_DICT[pk]["genplan_thumb"] = genplan_thumb
        PROJECTS_DICT[pk]["building_set"] = building_dict
        print(PROJECTS_DICT[pk])
        time.sleep(1)

