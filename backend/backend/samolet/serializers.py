from rest_framework import serializers


from samolet.models import *



class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = "__all__"

class BuildingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Building
        fields = "__all__"
    
class SectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Section
        fields = "__all__"

class FlatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Flat
        fields = "__all__"



class CheckSerializer(serializers.ModelSerializer):
    analys_image_url = serializers.SerializerMethodField('get_analys_image_url', read_only=True)
    analys_video_url = serializers.SerializerMethodField('get_analys_video_url', read_only=True)
    class Meta:
        model = Check
        fields = ['id', 'flat', 'video', 'analysis', 'date', 'is_analysed', 'analys_image','analys_square','analys_image_url','analys_video_url']
        #fields = "__all__"
    def get_analys_image_url(self, obj):
        try:
            return f'http://87.244.7.150:8000{obj.analys_image.url}'
        except:
            return None
    def get_analys_video_url(self, obj):
        try:
            return f'http://87.244.7.150:8000{obj.video.url}'
        except:
            return None

class FlatSerializer_additional(serializers.ModelSerializer):
    #checks =
    checks = CheckSerializer(many=True, read_only=True)

    class Meta:
        model = Flat
        fields = ['id', 'floor','number','section','checks']
        # add extra field