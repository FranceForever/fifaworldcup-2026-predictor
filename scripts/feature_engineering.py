#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ============================
# Part 1: Load Filtered Match Data
# ============================
filtered_file = 'data/filtered_competitive_matches.csv'
if not os.path.exists(filtered_file):
    print(f"Error: Filtered dataset not found at {filtered_file}. Run filter_data.py first.")
    exit(1)

df = pd.read_csv(filtered_file, parse_dates=['date'])
df.sort_values('date', inplace=True)

# ============================
# Part 2: Compute Basic Team Features
# ============================
def compute_recent_win_rate(team, df, n=10):
    """Compute the win rate for a team over its last n matches."""
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    team_matches.sort_values('date', inplace=True)
    recent_matches = team_matches.tail(n)
    if recent_matches.empty:
        return np.nan
    wins = 0
    for _, match in recent_matches.iterrows():
        if match['home_team'] == team and match['home_score'] > match['away_score']:
            wins += 1
        elif match['away_team'] == team and match['away_score'] > match['home_score']:
            wins += 1
    return wins / len(recent_matches)

def compute_goal_diff_trend(team, df, n=10):
    """Compute the average goal difference for a team over its last n matches."""
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    team_matches.sort_values('date', inplace=True)
    recent_matches = team_matches.tail(n)
    if recent_matches.empty:
        return np.nan
    goal_diffs = []
    for _, match in recent_matches.iterrows():
        if match['home_team'] == team:
            goal_diff = match['home_score'] - match['away_score']
        else:
            goal_diff = match['away_score'] - match['home_score']
        goal_diffs.append(goal_diff)
    return np.mean(goal_diffs)

# Compute team-level features and save to file
teams = pd.concat([df['home_team'], df['away_team']]).unique()
features_data = []
for team in teams:
    win_rate = compute_recent_win_rate(team, df, n=10)
    goal_diff_trend = compute_goal_diff_trend(team, df, n=10)
    features_data.append({
        'team': team,
        'win_rate_last10': win_rate,
        'avg_goal_diff_last10': goal_diff_trend
    })
df_team_features = pd.DataFrame(features_data)
df_team_features.to_csv('data/team_features.csv', index=False)
print("Team features saved to data/team_features.csv")

# ============================
# Part 3: Compute Head-to-Head Statistics
# ============================
def compute_head2head(team, opponent, data):
    """
    Compute head-to-head win rate and average goal difference for team against opponent.
    """
    matches = data[
        ((data['home_team'] == team) & (data['away_team'] == opponent)) |
        ((data['home_team'] == opponent) & (data['away_team'] == team))
    ]
    if matches.empty:
        return np.nan, np.nan
    wins = 0
    goal_diff_total = 0
    for _, match in matches.iterrows():
        if match['home_team'] == team:
            score_diff = match['home_score'] - match['away_score']
            if score_diff > 0:
                wins += 1
        else:
            score_diff = match['away_score'] - match['home_score']
            if score_diff > 0:
                wins += 1
        goal_diff_total += score_diff
    head2head_win_rate = wins / len(matches)
    head2head_avg_goal_diff = goal_diff_total / len(matches)
    return head2head_win_rate, head2head_avg_goal_diff

# Precompute head-to-head lookup table for all pairs of teams
teams_list = teams.tolist()
head2head_data = []
for i, team in enumerate(teams_list):
    for opponent in teams_list[i+1:]:
        win_rate, avg_goal_diff = compute_head2head(team, opponent, df)
        head2head_data.append({
            'team': team,
            'opponent': opponent,
            'head2head_win_rate': win_rate,
            'head2head_avg_goal_diff': avg_goal_diff
        })
        # Inverse stats for opponent vs. team
        if not np.isnan(win_rate):
            head2head_data.append({
                'team': opponent,
                'opponent': team,
                'head2head_win_rate': 1 - win_rate,
                'head2head_avg_goal_diff': -avg_goal_diff
            })
df_head2head = pd.DataFrame(head2head_data)
df_head2head.to_csv('data/head2head_stats.csv', index=False)
print("Head-to-head lookup table saved to data/head2head_stats.csv")

# ============================
# Part 4: Ranking Data Integration and Trend Computation
# ============================
# Load ranking data
ranking_file = 'data/fifa_ranking-2024-06-20.csv'
if not os.path.exists(ranking_file):
    print(f"Error: Ranking data file not found at {ranking_file}.")
    exit(1)
rankings = pd.read_csv(ranking_file)

# Rename columns to match our expected names:
if 'country_full' in rankings.columns:
    rankings.rename(columns={'country_full': 'team'}, inplace=True)
if 'total_points' in rankings.columns:
    rankings.rename(columns={'total_points': 'ranking_score'}, inplace=True)

# Convert the date column; the CSV uses 'rank_date' for ranking dates
rankings['ranking_date'] = pd.to_datetime(rankings['rank_date'], errors='coerce')

# Optionally, you can drop unused columns if needed
# rankings = rankings[['team', 'ranking_score', 'ranking_date', ...]]

# Standardize team names in rankings and match data (update mapping as needed)
team_mapping = {'USA': 'United States', 'ENG': 'England'}  # Extend as necessary
rankings['team'] = rankings['team'].replace(team_mapping)
df['home_team'] = df['home_team'].replace(team_mapping)
df['away_team'] = df['away_team'].replace(team_mapping)

# Function to get the nearest ranking record for a team before the match date
def get_nearest_ranking(team, match_date, ranking_data):
    team_rankings = ranking_data[ranking_data['team'] == team]
    team_rankings = team_rankings[team_rankings['ranking_date'] <= match_date]
    if team_rankings.empty:
        return None
    nearest_record = team_rankings.loc[team_rankings['ranking_date'].idxmax()]
    return nearest_record

# Add ranking data to each match in the dataset
home_ranking_scores = []
away_ranking_scores = []
ranking_diffs = []

for _, match in df.iterrows():
    match_date = match['date']
    home_team = match['home_team']
    away_team = match['away_team']
    
    home_rank = get_nearest_ranking(home_team, match_date, rankings)
    away_rank = get_nearest_ranking(away_team, match_date, rankings)
    
    if home_rank is not None and away_rank is not None:
        home_ranking_scores.append(home_rank['ranking_score'])
        away_ranking_scores.append(away_rank['ranking_score'])
        ranking_diffs.append(home_rank['ranking_score'] - away_rank['ranking_score'])
    else:
        home_ranking_scores.append(np.nan)
        away_ranking_scores.append(np.nan)
        ranking_diffs.append(np.nan)

df['home_ranking_score'] = home_ranking_scores
df['away_ranking_score'] = away_ranking_scores
df['ranking_diff'] = ranking_diffs

# Compute ranking trends (momentum)
def compute_ranking_trend(team, match_date, ranking_data, offset=3):
    team_rankings = ranking_data[ranking_data['team'] == team]
    team_rankings = team_rankings[team_rankings['ranking_date'] <= match_date]
    if len(team_rankings) < offset + 1:
        return np.nan
    team_rankings = team_rankings.sort_values('ranking_date')
    current_score = team_rankings.iloc[-1]['ranking_score']
    previous_score = team_rankings.iloc[-(offset + 1)]['ranking_score']
    return current_score - previous_score

home_ranking_trends = []
away_ranking_trends = []

for _, match in df.iterrows():
    match_date = match['date']
    home_team = match['home_team']
    away_team = match['away_team']
    
    home_trend = compute_ranking_trend(home_team, match_date, rankings)
    away_trend = compute_ranking_trend(away_team, match_date, rankings)
    
    home_ranking_trends.append(home_trend)
    away_ranking_trends.append(away_trend)

df['home_ranking_trend'] = home_ranking_trends
df['away_ranking_trend'] = away_ranking_trends
df['ranking_trend_diff'] = df['home_ranking_trend'] - df['away_ranking_trend']

# Save the dataset with ranking information
df.to_csv('data/filtered_with_ranking_trends.csv', index=False)
print("Filtered dataset with ranking trends saved to data/filtered_with_ranking_trends.csv")

# ============================
# Part 5: Merge Features and Scale Numerical Features
# ============================
# Adjust numerical_cols based on the features you plan to use in your model.
scaler = StandardScaler()
numerical_cols = ['ranking_diff', 'ranking_trend_diff']  # Extend this list as needed

df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the scaler for future inference
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the final, feature-rich and scaled dataset
final_output = 'data/final_features_dataset.csv'
df_scaled.to_csv(final_output, index=False)
print(f"Final feature-rich dataset saved to {final_output}")