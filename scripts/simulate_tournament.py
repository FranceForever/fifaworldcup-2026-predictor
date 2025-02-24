#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import random

# ----------------------------
# Function for User Group Assignment
# ----------------------------
def get_user_groups():
    print("Would you like to manually assign teams to groups? (y/n): ", end="")
    user_choice = input().strip().lower()
    if user_choice == 'y':
        groups = {}
        for i in range(8):
            group_name = f"Group {chr(65 + i)}"  # Group A, B, ...
            while True:
                teams_input = input(f"Enter exactly 4 team names for {group_name} (separated by commas): ")
                teams = [team.strip() for team in teams_input.split(",") if team.strip()]
                if len(teams) == 4:
                    groups[group_name] = teams
                    break
                else:
                    print("Error: You must enter exactly 4 teams. Please try again.")
        return groups
    else:
        return None

# ----------------------------
# Load Pre-trained Model, Mappings, and Team Features
# ----------------------------
model_file = 'final_model.pt'
if not os.path.exists(model_file):
    print(f"Error: Model file not found at {model_file}.")
    exit(1)

mapping_file = 'non_numeric_mappings.pkl'
if not os.path.exists(mapping_file):
    print(f"Error: Mapping file not found at {mapping_file}.")
    exit(1)
with open(mapping_file, 'rb') as f:
    non_numeric_mappings = pickle.load(f)
# Assume team mapping is stored under 'home_team'
team_mapping = non_numeric_mappings.get('home_team', {})

team_features_file = 'data/team_features.csv'
if not os.path.exists(team_features_file):
    print(f"Error: Team features file not found at {team_features_file}.")
    exit(1)
df_team = pd.read_csv(team_features_file)
# Expected columns: 'team', 'win_rate_last10', 'avg_goal_diff_last10'

# ----------------------------
# Define Neural Network Architecture (must match training)
# ----------------------------
class FootballNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(FootballNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # 256//2 = 128
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        # Residual connection: add a slice of x1 to x3
        x3 = self.relu(x3 + x1[:, :x3.shape[1]])
        x3 = self.dropout(x3)
        x4 = self.fc4(x3)
        return x4

input_dim = 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_model = FootballNN(input_dim=input_dim, hidden_dim=256, output_dim=3)
sim_model.load_state_dict(torch.load(model_file, map_location=device))
sim_model.to(device)
sim_model.eval()

# ----------------------------
# Helper Functions for Simulation
# ----------------------------
def construct_match_feature(home_team, away_team):
    """
    Construct a 6-dimensional feature vector using team-level features:
      [home_win_rate_last10, home_avg_goal_diff_last10,
       away_win_rate_last10, away_avg_goal_diff_last10,
       difference in win_rate, difference in avg_goal_diff].
    Then pad with zeros to reach 21 dimensions.
    """
    try:
        home_feat = df_team[df_team['team'] == home_team].iloc[0]
    except IndexError:
        home_feat = {'win_rate_last10': 0.5, 'avg_goal_diff_last10': 0.0}
    try:
        away_feat = df_team[df_team['team'] == away_team].iloc[0]
    except IndexError:
        away_feat = {'win_rate_last10': 0.5, 'avg_goal_diff_last10': 0.0}
    
    feat = np.array([
        home_feat['win_rate_last10'],
        home_feat['avg_goal_diff_last10'],
        away_feat['win_rate_last10'],
        away_feat['avg_goal_diff_last10'],
        home_feat['win_rate_last10'] - away_feat['win_rate_last10'],
        home_feat['avg_goal_diff_last10'] - away_feat['avg_goal_diff_last10']
    ], dtype=np.float32)
    padded_feat = np.zeros(input_dim, dtype=np.float32)
    padded_feat[:6] = feat
    return padded_feat

def predict_match(home_team, away_team):
    """
    Predict the outcome of a match between home_team and away_team.
    Returns:
      pred_class: 0 = Away win, 1 = Draw, 2 = Home win.
      probs: array of probabilities for each class.
    """
    feat = construct_match_feature(home_team, away_team)
    feat_tensor = torch.tensor(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        output = sim_model(feat_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
    return pred_class, probs

def simulate_group_match(home_team, away_team):
    """
    Simulate a group stage match between home_team and away_team.
    If the model predicts a draw, compare the teams’ win rates:
      If the difference (home_wr - away_wr) >= 0.05, award a home win.
      If <= -0.05, award an away win.
      Otherwise, keep it a draw.
    Then simulate an exact score:
      - For a win, simulate a margin: base margin (1-3 goals) plus an adjustment proportional to the win rate difference.
      - For a draw, randomly choose a tied score: 0-0, 1-1, or 2-2.
    Returns home_score, away_score, final_pred_class, probs.
    """
    pred_class, probs = predict_match(home_team, away_team)
    # Retrieve team win rates
    try:
        home_wr = df_team[df_team['team'] == home_team].iloc[0]['win_rate_last10']
    except IndexError:
        home_wr = 0.5
    try:
        away_wr = df_team[df_team['team'] == away_team].iloc[0]['win_rate_last10']
    except IndexError:
        away_wr = 0.5
    diff = home_wr - away_wr
    # If draw and significant difference, break the tie for group stage if desired.
    if pred_class == 1:
        if diff >= 0.05:
            pred_class = 2
        elif diff <= -0.05:
            pred_class = 0
    # Simulate exact score based on final prediction.
    if pred_class == 2:  # Home win
        base = np.random.randint(1, 4)  # 1 to 3 goals margin
        adjustment = int(diff * 2)  # if positive, add margin
        margin = max(base + adjustment, 1)
        home_score, away_score = margin, 0
    elif pred_class == 0:  # Away win
        base = np.random.randint(1, 4)
        adjustment = int(-diff * 2)
        margin = max(base + adjustment, 1)
        home_score, away_score = 0, margin
    else:  # Draw
        tie_score = np.random.choice([0, 1, 2])
        home_score = away_score = tie_score
    return home_score, away_score, pred_class, probs

def simulate_knockout_match(team1, team2):
    """
    Simulate a knockout match between team1 and team2.
    If the model predicts a draw, simulate a tied score (e.g., 1-1) and then break the tie:
      The team with the better group stage ranking wins with a 60% chance.
    Returns home_score, away_score, final_winner, final_pred_class, probs.
    """
    pred_class, probs = predict_match(team1, team2)
    if pred_class == 2:  # Team1 wins
        base = np.random.randint(1, 4)
        home_score, away_score = base, 0
        final_winner = team1
    elif pred_class == 0:  # Team2 wins
        base = np.random.randint(1, 4)
        home_score, away_score = 0, base
        final_winner = team2
    else:  # Draw predicted
        # Simulate a tie score (e.g., 1-1)
        home_score, away_score = 1, 1
        # Use group stage rankings for tie-breaker: lower rank (from group ranking) is better.
        rank1 = rank2 = None
        for grp, ranking in group_rankings.items():
            if team1 in ranking:
                rank1 = ranking.index(team1) + 1
            if team2 in ranking:
                rank2 = ranking.index(team2) + 1
        if rank1 is not None and rank2 is not None:
            # The better-ranked team wins with a 60% chance.
            if rank1 < rank2:
                final_winner = team1 if random.random() < 0.6 else team2
            elif rank2 < rank1:
                final_winner = team2 if random.random() < 0.6 else team1
            else:
                final_winner = random.choice([team1, team2])
        else:
            final_winner = random.choice([team1, team2])
        # Adjust score: winner gets an extra goal
        if final_winner == team1:
            home_score, away_score = 2, 1
            pred_class = 2
        else:
            home_score, away_score = 1, 2
            pred_class = 0
    return home_score, away_score, final_winner, pred_class, probs

# ----------------------------
# Get Group Assignments from User (8 groups, 4 teams each)
# ----------------------------
user_groups = get_user_groups()
if user_groups is not None:
    groups = user_groups
    print("\nUsing user-defined groups:")
    print(groups)
else:
    groups = {
        "Group A": ["Brazil", "Germany", "Japan", "Nigeria"],
        "Group B": ["Argentina", "France", "South Korea", "Egypt"],
        "Group C": ["Mexico", "Belgium", "Iran", "South Africa"],
        "Group D": ["Uruguay", "Spain", "Saudi Arabia", "Cameroon"],
        "Group E": ["Colombia", "England", "Tunisia", "Canada"],
        "Group F": ["Ecuador", "Italy", "Australia", "Morocco"],
        "Group G": ["Peru", "Netherlands", "USA", "Ghana"],
        "Group H": ["Chile", "Portugal", "Costa Rica", "Ivory Coast"]
    }
    print("\nUsing default groups:")

# ----------------------------
# Group Stage Simulation
# ----------------------------
print("\n--- Group Stage Simulation ---")
group_stage_results = {}
group_points = {}
group_goal_diff = {}
group_rankings = {}

# Simulate each match with exact scores.
for group, teams in groups.items():
    print(f"\nSimulating matches for {group}")
    points = {team: 0 for team in teams}
    goal_diff = {team: 0 for team in teams}
    match_results = []
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            home_team = teams[i]
            away_team = teams[j]
            home_score, away_score, final_pred, probs = simulate_group_match(home_team, away_team)
            print(f"  {home_team} vs {away_team} => Score: {home_score}-{away_score}, Prediction: {final_pred}")
            if home_score > away_score:
                points[home_team] += 3
            elif away_score > home_score:
                points[away_team] += 3
            else:
                points[home_team] += 1
                points[away_team] += 1
            goal_diff[home_team] += (home_score - away_score)
            goal_diff[away_team] += (away_score - home_score)
            match_results.append({
                "group": group,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "predicted_class": final_pred,
                "probabilities": probs
            })
    group_stage_results[group] = match_results
    group_points[group] = points
    group_goal_diff[group] = goal_diff
    ranking = sorted(teams, key=lambda t: (points[t], goal_diff[t]), reverse=True)
    group_rankings[group] = ranking
    print(f"{group} Points: {points}")
    print(f"{group} Goal Difference: {goal_diff}")
    print(f"{group} Ranking: {ranking}")

# ----------------------------
# Knockout Stage Simulation
# ----------------------------
# Advance top 2 teams from each group.
advancing_teams = []
for group in groups:
    ranking = group_rankings[group]
    advancing_teams.extend(ranking[:2])
print("\nAdvancing Teams (Round of 16):")
print(advancing_teams)

def simulate_knockout_round(teams):
    winners = []
    round_matches = []
    for i in range(0, len(teams), 2):
        team1 = teams[i]
        team2 = teams[i+1]
        home_score, away_score, winner, final_pred, probs = simulate_knockout_match(team1, team2)
        winners.append(winner)
        round_matches.append({
            "team1": team1,
            "team2": team2,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "predicted_class": final_pred,
            "probabilities": probs
        })
        print(f"Knockout Match: {team1} vs {team2} => Score: {home_score}-{away_score}, Winner: {winner}")
    return winners, round_matches

current_round = advancing_teams
knockout_bracket = []
round_number = 1
while len(current_round) > 1:
    print(f"\n--- Knockout Round {round_number}: {len(current_round)} teams ---")
    winners, matches = simulate_knockout_round(current_round)
    knockout_bracket.append(matches)
    current_round = winners
    round_number += 1

print("\nTournament Champion:", current_round[0])

# ----------------------------
# Save Results to CSV
# ----------------------------
df_matches = pd.DataFrame([m for group, matches in group_stage_results.items() for m in matches])
df_matches.to_csv("group_matches.csv", index=False)
print("Group stage match details saved to group_matches.csv")

summary = []
for group in groups:
    for team in groups[group]:
        summary.append({
            "group": group,
            "team": team,
            "points": group_points[group][team],
            "goal_difference": group_goal_diff[group][team],
            "group_rank": group_rankings[group].index(team) + 1
        })
df_summary = pd.DataFrame(summary)
df_summary.to_csv("group_points.csv", index=False)
print("Group stage summary saved to group_points.csv")