# Generated by Django 4.2.1 on 2023-05-28 14:08

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("samolet", "0002_rename_flats_on_floor_building_floors_total_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="section",
            name="project",
        ),
    ]
