from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.exceptions import ValidationError
import datetime
from django.core.validators import FileExtensionValidator

def validate_file_size(value):
    filesize= value.size
    
    if filesize > 104857600:
        raise ValidationError("You cannot upload file more than 100Mb")
    else:
        return value

class Project(models.Model):
    samolet_pk = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=128)
    url = models.URLField(max_length=256, blank=True, null=True)
    genplan_thumb = models.URLField(max_length=256, blank=True, null=True)
    slug = models.SlugField(null=False, unique=True)
    latitude = models.FloatField(validators=[MinValueValidator(-90),
                                            MaxValueValidator(90)],null=True, blank = True, default=None)
    longitude = models.FloatField(validators=[MinValueValidator(-180),
                                            MaxValueValidator(180)],null=True, blank = True, default=None)
    genplan_thumb = models.URLField(max_length=256, blank=True, null=True)
    card_image = models.URLField(max_length=256, blank=True, null=True)
    developer = models.CharField(max_length=256, blank=True, null=True)
    location_image = models.URLField(max_length=256, blank=True, null=True )
    def __str__(self):
        return self.name



class Building(models.Model):
    samolet_pk = models.IntegerField(unique=True, primary_key=True)
    project = models.ForeignKey(Project, on_delete=models.PROTECT)
    name = models.CharField(max_length=128)
    url = models.URLField(max_length=256, blank=True, null=True)
    number = models.CharField(max_length=8, blank=True, null=True)
    plan = models.URLField(max_length=256, blank=True, null=True)
    section_count = models.IntegerField(blank=True, null=True)
    section_set = models.JSONField(blank=True, null=True, default = list)
    floors_total = models.IntegerField(blank=True, null=True)
    section_total = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.name

class Section(models.Model):
    samolet_pk = models.IntegerField(unique=True, primary_key=True)
    building = models.ForeignKey(Building, on_delete=models.PROTECT,related_name="section_building")
    floors_total = models.IntegerField(blank=True, null=True)
    number = models.IntegerField(blank=True, null=True)
    flats_on_floor = models.IntegerField(blank=True, null=True)


@receiver(post_save, sender=Section)
def create_flats(sender, instance, created, **kwargs):
    if created:
        flat_num = 1
        for floor in range(1, instance.floors_total +1):
            for flat in range(instance.flats_on_floor):
                Flat.objects.create(floor=floor, number=flat_num, section=instance)
                flat_num+=1



'''
class Floor(models.Model):
    samolet_pk = models.IntegerField(unique=True, primary_key=True)
    number = models.IntegerField(blank=True, null=True)
    plan = models.URLField(max_length=256, blank=True, null=True)
'''

class Flat(models.Model):
    floor = models.IntegerField(blank=True, null=True)
    #building = models.ForeignKey(Building, on_delete=models.PROTECT)
    section = models.ForeignKey(Section, on_delete=models.PROTECT)
    number = models.IntegerField(blank=True, null=True)
    Ready_precentage = models.FloatField(default=0.0, blank=True, null=True)
    square = models.FloatField(blank=True, null=True, default=0.0)

class Check(models.Model):
    flat = models.ForeignKey(Flat,related_name='checks', on_delete=models.PROTECT, blank=True, null=True)
    video = models.FileField(blank=True, null=True, validators=[validate_file_size], upload_to='checks/')
    analysis = models.JSONField(blank=True, null=True, default = list)
    date = models.DateField(default=datetime.date.today)
    is_analysed = models.BooleanField(default=False)
    analys_image = models.ImageField(blank=True, null=True, upload_to='analys_images/',validators=[FileExtensionValidator(['jpg', 'png','','*'])])
    analys_square = models.FloatField(blank=True, null=True, default=0.0)


@receiver(post_save, sender=Check)
def update_check(sender, instance, created, **kwargs):
    if instance.analysis:
        try:
            perc = float(instance.analysis["Ready_precentage"])
            flat = Flat.objects.get(pk=instance.flat.pk)
            flat.Ready_precentage = perc
            flat.save()
        except:
            pass