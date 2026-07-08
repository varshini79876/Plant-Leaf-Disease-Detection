from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.history, name='history'),
    path('prediction/<int:pk>/', views.prediction_detail, name='prediction_detail'),
    path('about/', views.about, name='about'),
]
