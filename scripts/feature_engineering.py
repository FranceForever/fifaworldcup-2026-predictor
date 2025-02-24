#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Define team name mapping (extend as needed)
team_mapping = {'USA': 'United States', 'ENG': 'England'}

# Define tournament importance mapping
tournament_importance = {
    'FIFA World Cup': 1.0,
    'FIFA World Cup qualification': 0.7,
    'UEFA Euro': 0.8,
    'UEFA Euro qualification': 0.6,
    'Copa América': 0.75,
    'Copa América qualification': 0.6,
    'UEFA Nations League': 0.65,
    'CONMEBOL Nations League': 0.65
}

# ============================
# Part 1: Load and Preprocess Filtered Match Data
# ============================
filtered_file = 'data/filtered_competitive_matches.csv'
if not os.path.exists(filtered_file):
    print(f"Error: Filtered dataset not found at {filtered_file}. Run filter_data.py first.")
    exit(1)

df = pd.read_csv(filtered_file, parse_dates=['date'])
# Standardize date format (normalize to midnight)
df['date'] = pd.to_datetime(df['date']).dt.normalize()
df.sort_values('date', inplace=True)

# Unify team names in match data
df['home_team'] = df['home_team'].replace(team_mapping)
df['away_team'] = df['away_team'].replace(team_mapping)

# Map tournament importance: create a new column based on the tournament type
df['tournament_importance'] = df['tournament'].map(tournament_importance)

# Add recency features: days since last match for home and away teams
df['home_last_date'] = df.groupby('home_team')['date'].shift(1)
df['home_recency'] = (df['date'] - df['home_last_date']).dt.days
df['away_last_date'] = df.groupby('away_team')['date'].shift(1)
df['away_recency'] = (df['date'] - df['away_last_date']).dt.days

# Fill NaN recency with the median value (avoid chained assignment warnings)
df['home_recency'] = df['home_recency'].fillna(df['home_recency'].median())
df['away_recency'] = df['away_recency'].fillna(df['away_recency'].median())

# Create weighted recency features by multiplying recency with tournament importance
df['home_weighted_recency'] = df['home_recency'] * df['tournament_importance']
df['away_weighted_recency'] = df['away_recency'] * df['tournament_importance']

# Add seasonality features (if desired)
df['match_month'] = df['date'].dt.month
df['match_day_of_week'] = df['date'].dt.dayofweek

# ===== New Feature: Recency Relative to Current Date =====
# Compute the number of days between today's date (normalized) and the match date.
current_date = pd.Timestamp('now').normalize()
df['days_since_match'] = (current_date - df['date']).dt.days
# Create a weighted version using tournament importance
df['weighted_days_since_match'] = df['days_since_match'] * df['tournament_importance']

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
# Part 3: Compute Head-to-Head Statistics (All-Time and Last 5 Meetings)
# ============================
def compute_head2head_stats(team, opponent, data, last_n=None):
    """
    Compute head-to-head statistics for team vs opponent.
    If last_n is specified, use only the last n matches.
    
    Returns a dict with:
      - win_rate: fraction of matches won by 'team'
      - avg_goal_diff: average goal difference for 'team'
      - num_matches: total matches considered
    """
    matches = data[
        ((data['home_team'] == team) & (data['away_team'] == opponent)) |
        ((data['home_team'] == opponent) & (data['away_team'] == team))
    ]
    if last_n is not None and not matches.empty:
        matches = matches.sort_values('date').tail(last_n)
    if matches.empty:
        return {'win_rate': np.nan, 'avg_goal_diff': np.nan, 'num_matches': 0}
    wins = 0
    goal_diff_total = 0
    for _, match in matches.iterrows():
        if match['home_team'] == team:
            score_diff = match['home_score'] - match['away_score']
        else:
            score_diff = match['away_score'] - match['home_score']
        if score_diff > 0:
            wins += 1
        goal_diff_total += score_diff
    num_matches = len(matches)
    win_rate = wins / num_matches if num_matches > 0 else np.nan
    avg_goal_diff = goal_diff_total / num_matches if num_matches > 0 else np.nan
    return {'win_rate': win_rate, 'avg_goal_diff': avg_goal_diff, 'num_matches': num_matches}

teams_list = sorted(teams)
head2head_records = []
for i, team in enumerate(teams_list):
    for opponent in teams_list[i+1:]:
        all_time_stats = compute_head2head_stats(team, opponent, df, last_n=None)
        last5_stats = compute_head2head_stats(team, opponent, df, last_n=5)
        
        all_time_stats_inv = {
            'win_rate': 1 - all_time_stats['win_rate'] if not np.isnan(all_time_stats['win_rate']) else np.nan,
            'avg_goal_diff': -all_time_stats['avg_goal_diff'] if not np.isnan(all_time_stats['avg_goal_diff']) else np.nan,
            'num_matches': all_time_stats['num_matches']
        }
        last5_stats_inv = {
            'win_rate': 1 - last5_stats['win_rate'] if not np.isnan(last5_stats['win_rate']) else np.nan,
            'avg_goal_diff': -last5_stats['avg_goal_diff'] if not np.isnan(last5_stats['avg_goal_diff']) else np.nan,
            'num_matches': last5_stats['num_matches']
        }
        
        head2head_records.append({
            'team': team,
            'opponent': opponent,
            'all_time_win_rate': all_time_stats['win_rate'],
            'all_time_avg_goal_diff': all_time_stats['avg_goal_diff'],
            'all_time_num_matches': all_time_stats['num_matches'],
            'last5_win_rate': last5_stats['win_rate'],
            'last5_avg_goal_diff': last5_stats['avg_goal_diff'],
            'last5_num_matches': last5_stats['num_matches']
        })
        head2head_records.append({
            'team': opponent,
            'opponent': team,
            'all_time_win_rate': all_time_stats_inv['win_rate'],
            'all_time_avg_goal_diff': all_time_stats_inv['avg_goal_diff'],
            'all_time_num_matches': all_time_stats_inv['num_matches'],
            'last5_win_rate': last5_stats_inv['win_rate'],
            'last5_avg_goal_diff': last5_stats_inv['avg_goal_diff'],
            'last5_num_matches': last5_stats_inv['num_matches']
        })

df_head2head = pd.DataFrame(head2head_records)
df_head2head = df_head2head[df_head2head['all_time_num_matches'] > 0]
df_head2head = df_head2head.sort_values(['team', 'opponent'])
df_head2head.to_csv('data/head2head_stats.csv', index=False)
print("Head-to-head stats (all-time and last 5 meetings) saved to data/head2head_stats.csv")

# ============================
# Part 4: Ranking Data Integration and Trend Computation
# ============================
ranking_file = 'data/fifa_ranking-2024-06-20.csv'
if not os.path.exists(ranking_file):
    print(f"Error: Ranking data file not found at {ranking_file}.")
    exit(1)
rankings = pd.read_csv(ranking_file)

if 'country_full' in rankings.columns:
    rankings.rename(columns={'country_full': 'team'}, inplace=True)
if 'total_points' in rankings.columns:
    rankings.rename(columns={'total_points': 'ranking_score'}, inplace=True)
if 'rank' in rankings.columns:
    rankings.rename(columns={'rank': 'team_rank'}, inplace=True)

rankings['ranking_date'] = pd.to_datetime(rankings['rank_date'], errors='coerce')

rankings['team'] = rankings['team'].replace(team_mapping)
df['home_team'] = df['home_team'].replace(team_mapping)
df['away_team'] = df['away_team'].replace(team_mapping)

def get_nearest_ranking(team, match_date, ranking_data):
    team_rankings = ranking_data[ranking_data['team'] == team]
    team_rankings = team_rankings[team_rankings['ranking_date'] <= match_date]
    if team_rankings.empty:
        return None
    nearest_record = team_rankings.loc[team_rankings['ranking_date'].idxmax()]
    return nearest_record

home_ranking_scores = []
away_ranking_scores = []
home_ranks = []
away_ranks = []
ranking_diffs = []

for _, match in df.iterrows():
    match_date = match['date']
    home_team = match['home_team']
    away_team = match['away_team']
    
    home_rank_record = get_nearest_ranking(home_team, match_date, rankings)
    away_rank_record = get_nearest_ranking(away_team, match_date, rankings)
    
    if home_rank_record is not None and away_rank_record is not None:
        home_ranking_scores.append(home_rank_record['ranking_score'])
        away_ranking_scores.append(away_rank_record['ranking_score'])
        ranking_diffs.append(home_rank_record['ranking_score'] - away_rank_record['ranking_score'])
        home_ranks.append(home_rank_record['team_rank'])
        away_ranks.append(away_rank_record['team_rank'])
    else:
        home_ranking_scores.append(np.nan)
        away_ranking_scores.append(np.nan)
        ranking_diffs.append(np.nan)
        home_ranks.append(np.nan)
        away_ranks.append(np.nan)

df['home_ranking_score'] = home_ranking_scores
df['away_ranking_score'] = away_ranking_scores
df['ranking_diff'] = ranking_diffs
df['home_rank'] = home_ranks
df['away_rank'] = away_ranks

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

# ----------------------------
# Filtering Matches Based on Rankings
# ----------------------------
# Remove matches if any team has a rank worse than 175 (i.e. team_rank > 175)
df = df[(df['home_rank'] <= 175) & (df['away_rank'] <= 175)]
print("Filtered matches: Keeping only matches where both teams are ranked 175 or better.")

df.to_csv('data/filtered_with_ranking_trends.csv', index=False)
print("Filtered dataset with ranking trends saved to data/filtered_with_ranking_trends.csv")

# ============================
# Part 5: Merge Features and Scale Numerical Features
# ============================
# Before scaling, remove seasonality features that we don't want
df.drop(['match_month', 'match_day_of_week'], axis=1, inplace=True)

# Optionally, also drop raw 'date' if you don't want to feed it into the model,
# since we now have recency features.
df.drop(['date'], axis=1, inplace=True)

scaler = StandardScaler()
numerical_cols = ['ranking_diff', 'ranking_trend_diff']  # Extend this list as needed

df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

final_output = 'data/final_features_dataset.csv'
df_scaled.to_csv(final_output, index=False)
print(f"Final feature-rich dataset saved to {final_output}")