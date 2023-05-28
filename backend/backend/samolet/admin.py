from django.contrib import admin
from samolet.models import *
# Register your models here.



class FlatAdmin(admin.ModelAdmin):
    list_display = ("id", "section", "floor", "number")


admin.site.register(Flat, FlatAdmin)


admin.site.register(Project)
admin.site.register(Building)
admin.site.register(Section)
#admin.site.register(Floor)
#admin.site.register(Flat)
admin.site.register(Check)



