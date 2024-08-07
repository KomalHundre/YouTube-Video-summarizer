from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Route for the homepage
    path('transcribe/', views.transcribe, name='transcribe'),  # Route for the transcribe function
]
