from .views import *
from django.urls import path


urlpatterns = [
    path('', main_menu, name='main-menu'),
    path('template', download_file, name='template')
]
