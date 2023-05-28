# Generated by Django 4.2.1 on 2023-05-28 13:33

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Floor",
            fields=[
                (
                    "samolet_pk",
                    models.IntegerField(primary_key=True, serialize=False, unique=True),
                ),
                ("number", models.IntegerField(blank=True, null=True)),
                ("plan", models.URLField(blank=True, max_length=256, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Project",
            fields=[
                ("samolet_pk", models.IntegerField(primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=128)),
                ("url", models.URLField(blank=True, max_length=256, null=True)),
                ("slug", models.SlugField(unique=True)),
                (
                    "latitude",
                    models.FloatField(
                        blank=True,
                        default=None,
                        null=True,
                        validators=[
                            django.core.validators.MinValueValidator(-90),
                            django.core.validators.MaxValueValidator(90),
                        ],
                    ),
                ),
                (
                    "longitude",
                    models.FloatField(
                        blank=True,
                        default=None,
                        null=True,
                        validators=[
                            django.core.validators.MinValueValidator(-180),
                            django.core.validators.MaxValueValidator(180),
                        ],
                    ),
                ),
                (
                    "genplan_thumb",
                    models.URLField(blank=True, max_length=256, null=True),
                ),
                ("card_image", models.URLField(blank=True, max_length=256, null=True)),
                ("developer", models.CharField(blank=True, max_length=256, null=True)),
                (
                    "location_image",
                    models.URLField(blank=True, max_length=256, null=True),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Section",
            fields=[
                (
                    "samolet_pk",
                    models.IntegerField(primary_key=True, serialize=False, unique=True),
                ),
                ("floors_total", models.IntegerField(blank=True, null=True)),
                ("project", models.CharField(max_length=128)),
                ("number", models.IntegerField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Flat",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "floor",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT, to="samolet.floor"
                    ),
                ),
                (
                    "section",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        to="samolet.section",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Check",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("video", models.FileField(blank=True, null=True, upload_to="checks/")),
                ("analysis", models.JSONField(blank=True, default=list, null=True)),
                ("date", models.DateTimeField(auto_now_add=True)),
                (
                    "Flat",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT, to="samolet.flat"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Building",
            fields=[
                (
                    "samolet_pk",
                    models.IntegerField(primary_key=True, serialize=False, unique=True),
                ),
                ("name", models.CharField(max_length=128)),
                ("url", models.URLField(blank=True, max_length=256, null=True)),
                ("number", models.CharField(blank=True, max_length=8, null=True)),
                ("plan", models.URLField(blank=True, max_length=256, null=True)),
                ("section_count", models.IntegerField(blank=True, null=True)),
                ("section_set", models.JSONField(blank=True, default=list, null=True)),
                ("flats_on_floor", models.IntegerField(blank=True, null=True)),
                (
                    "protject",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        to="samolet.project",
                    ),
                ),
            ],
        ),
    ]