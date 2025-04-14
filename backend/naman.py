import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading match data...")
df = pd.read_csv("processed_matches.csv")

# Drop unused columns before a match
cols_to_drop = ['season', 'match_name', 'date', 'home_score', 'away_score', 'h_match_points', 'a_match_points', 'winner']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Fill missing values
df.fillna(-33, inplace=True)

# Get dummies for categorical variables
df_dum = pd.get_dummies(df)

# Set up features and scaler
X = df_dum.drop(columns=['home_team_Arsenal'], errors='ignore')  # placeholder drop for key column to simulate trained model features
y = np.zeros(X.shape[0])  # dummy y for structure
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Simulated trained model (Logistic Regression-style with dummy logic)
def fake_model_predict_proba(X):
    np.random.seed(0)
    return np.random.dirichlet(np.ones(3), size=X.shape[0])  # Fake 3-class probs

# Ask for user input
home_team = input("Enter home team: ").strip()
away_team = input("Enter away team: ").strip()
year = input("Enter year of the match (e.g. 2021): ").strip()

# Filter original CSV again to match by user input
df_raw = pd.read_csv("processed_matches.csv")
match_row = df_raw[(df_raw['home_team'].str.lower() == home_team.lower()) &
                   (df_raw['away_team'].str.lower() == away_team.lower()) &
                   (df_raw['date'].str.startswith(year))].copy()

if match_row.empty:
    print("Match not found. Check team names and year.")
else:
    # Drop and prepare match row for prediction
    drop_cols = ['season', 'match_name', 'date', 'home_score', 'away_score', 'h_match_points', 'a_match_points', 'winner']
    match_row.drop(columns=[col for col in drop_cols if col in match_row.columns], inplace=True)
    match_row.fillna(-33, inplace=True)
    match_row = pd.get_dummies(match_row)

    for col in X.columns:
        if col not in match_row.columns:
            match_row[col] = 0
    match_row = match_row[X.columns]  # ensure column order
    match_scaled = scaler.transform(match_row)

    # Fake prediction using dummy model
    probs = fake_model_predict_proba(match_scaled)[0]
    labels = ['Draw', 'Away Win', 'Home Win']

    print("\n--- Match Prediction Probabilities ---")
    for label, prob in zip(labels, probs):
        print(f"{label}: {prob:.2%}")

    # Visualization
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=probs)
    plt.title(f"Prediction Odds: {home_team} vs {away_team} ({year})")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.show()