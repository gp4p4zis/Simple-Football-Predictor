from django.contrib import admin
from .models import Team, Match, Prediction

@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("id", "name")
    search_fields = ("name",)


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ("id", "home_team", "away_team", "kickoff_at", "status", "home_score", "away_score")
    list_filter = ("status", "home_team", "away_team")
    search_fields = ("home_team__name", "away_team__name")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "match", "predicted_result", "created_at")
    list_filter = ("predicted_result",)
    search_fields = ("match__home_team__name", "match__away_team__name")
