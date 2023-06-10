from django.urls import path
from django.urls import path, re_path, include
from rest_framework import routers

from api.views import *

router = routers.DefaultRouter() #SimpleRouter
router.register(r'projects', ProjectApiView)
router.register(r'buildings', BuildingApiView)
router.register(r'sections', SectionApiView)
router.register(r'flats', FlatApiView)
router.register(r'checks', CheckApiView)

urlpatterns = [
    path('projects/<int:samolet_pk>/getbuildings/',  ProjectApiView.as_view({"get": "getbuildings"})),
    path('buildings/<int:samolet_pk>/getsections/',  BuildingApiView.as_view({"get": "getsections"})),
    path('sections/<int:samolet_pk>/getflats/',  SectionApiView.as_view({"get": "getflats"})),
    path('flats/<int:pk>/getchecks/',  FlatApiView.as_view({"get": "getchecks"})),
    path('get_unchaked/', get_unchaked),
    path('', include(router.urls)),
]
