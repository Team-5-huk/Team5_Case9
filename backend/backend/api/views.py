from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.decorators import action
from samolet.models import *
from samolet.serializers import *
from users import authentication
from rest_framework.response import Response
from rest_framework.decorators import api_view

class ProjectApiView(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    authentication_classes = (authentication.CustomUserAuthentication,)
    permission_classes = (permissions.IsAuthenticated,)
    
    @action(methods=['get'], detail=True)
    def getbuildings(self, request, *args, **kwargs):
        pk = kwargs['samolet_pk']
        buildings = Building.objects.filter(project=pk)
        return Response(BuildingSerializer(buildings, many=True).data)

class BuildingApiView(viewsets.ModelViewSet):
    queryset = Building.objects.all()
    serializer_class = BuildingSerializer
    authentication_classes = (authentication.CustomUserAuthentication,)
    permission_classes = (permissions.IsAuthenticated,)

    @action(methods=['get'], detail=True)
    def getsections(self, request, *args, **kwargs):
        pk = kwargs['samolet_pk']
        sections = Section.objects.filter(building=pk)
        return Response(SectionSerializer(sections, many=True).data)

class SectionApiView(viewsets.ModelViewSet):
    queryset = Section.objects.all()
    serializer_class = SectionSerializer
    authentication_classes = (authentication.CustomUserAuthentication,)
    permission_classes = (permissions.IsAuthenticated,)

    @action(methods=['get'], detail=True)
    def getflats(self, request, *args, **kwargs):
        pk = kwargs['samolet_pk']
        flats = Flat.objects.filter(section=pk)
        return Response(FlatSerializer(flats, many=True).data)

class FlatApiView(viewsets.ModelViewSet):
    queryset = Flat.objects.all()
    serializer_class = FlatSerializer
    authentication_classes = (authentication.CustomUserAuthentication,)
    permission_classes = (permissions.IsAuthenticated,)

    @action(methods=['get'], detail=True)
    def getchecks(self, request, *args, **kwargs):
        pk = kwargs['pk']
        checks = Check.objects.filter(flat=pk)
        return Response(CheckSerializer(checks, many=True).data)

class CheckApiView(viewsets.ModelViewSet):
    queryset = Check.objects.all()
    serializer_class = CheckSerializer
    #authentication_classes = (authentication.CustomUserAuthentication,)
    #permission_classes = (permissions.IsAuthenticated,)

@api_view(['GET'])
def get_unchaked(request):
    #authentication_classes = (authentication.CustomUserAuthentication,)
    #authentication_classes = (authentication.AllowAny,)
    #permission_classes = (permissions.IsAuthenticated,)
    if request.method == 'GET':
        check = Check.objects.filter(is_analysed=False, video__isnull=False).exclude(video='').first()
        # TODO: Выбрать последний элемент check, у которого video не пусто
        if check:
            return Response(CheckSerializer(check, many=False).data)
        else:
            return Response({})