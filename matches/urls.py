from django.urls import path
from . import views

urlpatterns = [
    path("matches/", views.match_list, name="match_list"),
    path("matches/<int:pk>/", views.match_detail, name="match_detail"),
    path("matches/upload/", views.upload_matches_csv, name="upload_matches_csv"),
    path('matches/<int:pk>/predict/', views.generate_prediction, name='generate_prediction'),
]
