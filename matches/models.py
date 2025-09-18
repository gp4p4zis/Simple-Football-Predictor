from django.db import models

class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Match(models.Model):
    class Status(models.TextChoices):
        UPCOMING = "UPCOMING", "Upcoming"
        FINISHED = "FINISHED", "Finished"

    home_team = models.ForeignKey(Team, related_name="home_matches", on_delete=models.CASCADE)
    away_team = models.ForeignKey(Team, related_name="away_matches", on_delete=models.CASCADE)
    kickoff_at = models.DateTimeField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.UPCOMING)

    # Results
    home_score = models.IntegerField(null=True, blank=True)
    away_score = models.IntegerField(null=True, blank=True)

    # Extra stats
    HS = models.IntegerField(null=True, blank=True)   # Home Shots
    AS = models.IntegerField(null=True, blank=True)   # Away Shots
    HST = models.IntegerField(null=True, blank=True)  # Home Shots on Target
    AST = models.IntegerField(null=True, blank=True)  # Away Shots on Target
    HC = models.IntegerField(null=True, blank=True)   # Home Corners
    AC = models.IntegerField(null=True, blank=True)   # Away Corners
    
    # Odds (Bet365)
    B365H = models.FloatField(null=True, blank=True)
    B365D = models.FloatField(null=True, blank=True)
    B365A = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.home_team} vs {self.away_team} ({self.kickoff_at.date()})"


class Prediction(models.Model):
    match = models.ForeignKey(Match, related_name="predictions", on_delete=models.CASCADE)
    predicted_result = models.CharField(max_length=1, choices=[("H", "Home Win"), ("D", "Draw"), ("A", "Away Win")])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.match} → {self.predicted_result}"
