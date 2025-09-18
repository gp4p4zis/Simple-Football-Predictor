# Football Match Predictor

A machine learning-based football match outcome prediction tool using historical match data and betting odds. Built with Django, the app predicts match results (home win, draw, away win) and generates bookmaker-style odds.

---

## Features

* Predict outcomes for upcoming football matches.
* Compute team performance profiles based on historical match data.
* Handle multiple seasons with shrinkage to account for teams with fewer games.
* Incorporate betting odds (B365) in predictions.
* Upload CSV files to update match database.
* Simple web interface using Django.

---

## Technologies

* **Backend:** Django 5.2
* **Machine Learning:** scikit-learn (Logistic Regression, SimpleImputer, StandardScaler)
* **Data Handling:** pandas, numpy
* **Frontend:** Django templates

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Simple-Football-Predictor.git
cd Simple-Football-Predictor
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Apply migrations:

```bash
python manage.py migrate
```

5. Create a superuser for admin access:

```bash
python manage.py createsuperuser
```

6. Run the development server:

```bash
python manage.py runserver
```

---

## Usage

1. Visit 127.0.0.1:8000/matches/upload. You will be prompted to login with your superuser.
2. (optional, if you want new data) Upload match data CSVs via the **Upload Matches CSV** form.

   * CSV should include columns:
     `date, home_team, away_team, home_score, away_score, HS, AS, HST, AST, HC, AC, B365H, B365D, B365A`
3. Visit 127.0.0.1:8000/admin/ and click on Matchs -> ADD MATCH. Then, select home/away teams and enter a kickoff time, then save. The match needs to be "Upcoming".
4. Visit 127.0.0.1:8000/matches. You can view your upcoming matches there and by clicking on them, the prediction is made.

---

## Model Details

* **Features:** Goals scored/conceded, shots, shot accuracy, corners, betting odds, and games played per team.
* **Shrinkage:** Teams with fewer games are pulled toward the league average to account for limited data.
* **Algorithm:** Logistic Regression (multinomial) with scaled features.
* **Odds Conversion:** Probabilities are converted to odds with a bookmaker margin of 5%, including draw adjustment.


\*You can add screensho
