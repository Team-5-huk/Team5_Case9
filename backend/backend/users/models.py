from django.db import models
from django.contrib.auth import models as auth_models
from users.managers import UserManager

class User(auth_models.AbstractUser):
    ROLES = (
        ("admin", "Админ"),
        ("user", "Покупатель"),
        ("employee", "Обходчик"),
    )
    first_name = models.CharField(verbose_name="First Name", max_length=255)
    last_name = models.CharField(verbose_name="Last Name", max_length=255)
    email = models.EmailField(verbose_name="Email", max_length=255, unique=True)
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=32, choices = ROLES, default = 'user')

    username = None
    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["first_name", "last_name"]