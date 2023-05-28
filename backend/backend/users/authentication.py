from django.conf import settings
from rest_framework import authentication, exceptions
import jwt

from users import models


class CustomUserAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        token = request.COOKIES.get("jwt")

        if not token:
        # Only IF DEBUG
            token = request.headers.get("Authorization")
            if not token:
                return None
            token = token.replace('Bearer ', '')
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        except:
            raise exceptions.AuthenticationFailed("Unauthorized")
        user = models.User.objects.filter(id=payload["id"]).first()
        return (user, None)
