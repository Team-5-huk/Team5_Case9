# Generated by Django 4.2.1 on 2023-06-06 15:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("samolet", "0013_alter_check_flat"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="check",
            name="is_analysed",
        ),
        migrations.AddField(
            model_name="flat",
            name="Ready_precentage",
            field=models.FloatField(blank=True, default=0.0, null=True),
        ),
    ]
