from django import forms
from .models import Prediction

class MatchCSVUploadForm(forms.Form):
    """
    Upload a CSV with columns:
    date, home_team, away_team, home_score, away_score, HS, AS, HST, AST, HC, AC
    """
    csv_file = forms.FileField()


class PredictionForm(forms.ModelForm):
    """
    Optional: lets staff/admins create/edit predictions manually.
    Mostly useful for testing alongside ML predictions.
    """
    class Meta:
        model = Prediction
        fields = ["predicted_result"]
        widgets = {
            "predicted_result": forms.RadioSelect(choices=[
                ("H", "Home Win"),
                ("D", "Draw"),
                ("A", "Away Win"),
            ]),
        }
