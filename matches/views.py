from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from .forms import MatchCSVUploadForm
from .models import Match, Team
import csv
from io import TextIOWrapper
from django.contrib.admin.views.decorators import staff_member_required
from django.db import transaction
from .ml import train_model, predict_match
from datetime import datetime
from django.views.decorators.http import require_POST


def match_list(request):
    now = timezone.now()
    upcoming = Match.objects.filter(kickoff_at__gte=now).order_by("kickoff_at")[:50]
    recent = Match.objects.filter(kickoff_at__lt=now).order_by("-kickoff_at")[:20]
    return render(request, "matches/match_list.html", {"upcoming": upcoming, "recent": recent})


def match_detail(request, pk):
    match = get_object_or_404(Match, pk=pk)
    model_prediction = request.session.pop("model_prediction", None)

    if match.status == Match.Status.UPCOMING:
        model, acc, profiles, imputer, scaler = train_model()  # <-- unpack scaler
        if model:
            prediction = predict_match(
                model,
                match.home_team.name,
                match.away_team.name,
                profiles,
                imputer,
                scaler  # <-- pass scaler here
            )
            model_prediction = {
                "probs": prediction["probs"],
                "odds": prediction["odds"],
                "acc": round(acc * 100, 1) if acc else None
            }

    return render(
        request,
        "matches/match_detail.html",
        {
            "match": match,
            "model_prediction": model_prediction,
        },
    )



@staff_member_required
def upload_matches_csv(request):
    """
    Upload CSV with columns:
    date,home_team,away_team,home_score,away_score,HS,AS,HST,AST,HC,AC,B365H,B365D,B365A
    Overwrites existing matches with the new dataset.
    """
    def parse_int(val):
        if not val:
            return 0
        try:
            return int(float(str(val).replace(",", ".")))
        except ValueError:
            return 0
    
    def parse_float(val):
        """Convert to float safely (handles commas, bad values)."""
        if not val:
            return None
        try:
            return float(str(val).replace(",", "."))
        except ValueError:
            return None
        
    if request.method == "POST":
        form = MatchCSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = TextIOWrapper(request.FILES["csv_file"].file, encoding="utf-8")
            reader = csv.DictReader(csv_file)

            created = 0
            with transaction.atomic():
                # 🔥 Clear old dataset before import
                Match.objects.all().delete()

                for row in reader:
                    home, _ = Team.objects.get_or_create(name=row["home_team"])
                    away, _ = Team.objects.get_or_create(name=row["away_team"])
                    kickoff_naive = datetime.strptime(row["date"], "%Y-%m-%d")
                    kickoff_aware = timezone.make_aware(kickoff_naive)

                    Match.objects.create(
                        home_team=home,
                        away_team=away,
                        kickoff_at=kickoff_aware,
                        status=Match.Status.FINISHED,
                        home_score=int(row["home_score"]),
                        away_score=int(row["away_score"]),
                        HS=parse_int(row.get("HS")),
                        AS=parse_int(row.get("AS")),
                        HST=parse_int(row.get("HST")),
                        AST=parse_int(row.get("AST")),
                        HC=parse_int(row.get("HC")),
                        AC=parse_int(row.get("AC")),
                        B365H=parse_float(row.get("B365H")),
                        B365D=parse_float(row.get("B365D")),
                        B365A=parse_float(row.get("B365A")),
                    )
                    created += 1

            messages.success(request, f"Imported {created} matches (previous dataset overwritten).")
            return redirect("match_list")
    else:
        form = MatchCSVUploadForm()

    return render(request, "matches/upload_csv.html", {"form": form})



@require_POST
def generate_prediction(request, pk):
    match = get_object_or_404(Match, pk=pk)
    model, acc, profiles, imputer = train_model()
    if model:
        probs = predict_match(model, match.home_team.name, match.away_team.name, profiles, imputer)
        request.session["model_prediction"] = {"probs": probs, "acc": round(acc * 100, 1) if acc else None}
    else:
        request.session["model_prediction"] = None
    return redirect("match_detail", pk=match.pk)
