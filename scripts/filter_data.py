import os
import pandas as pd

# Define the file path to the results CSV relative to the scripts folder
results_file = 'data/results.csv'

# Check if the file exists
if not os.path.exists(results_file):
    print(f"Error: The file '{results_file}' does not exist. Please check the file name and path.")
    exit(1)

# Load the match results dataset from the data folder
df = pd.read_csv(results_file)

# Convert the 'date' column to datetime format (adjust the column name if needed)
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Filter for matches from 2010 onward
df = df[df['date'] >= '2010-01-01']

# Define the list of competitive tournaments to include
competitive_tournaments = [
    'FIFA World Cup',
    'FIFA World Cup qualification',
    'UEFA Euro',
    'UEFA Euro qualification',
    'Copa América',
    'Copa América qualification',
    'UEFA Nations League',
    'CONMEBOL Nations League'
]

# Filter the DataFrame to only include matches from the competitive tournaments
df_filtered = df[df['tournament'].isin(competitive_tournaments)]

# Exclude any matches that might be friendlies (if any appear)
df_filtered = df_filtered[~df_filtered['tournament'].str.contains('Friendly', case=False, na=False)]

# Reset the index and save the filtered dataset to a new CSV file
df_filtered.reset_index(drop=True, inplace=True)
output_file = 'data/filtered_competitive_matches.csv'
df_filtered.to_csv(output_file, index=False)

print(f"Filtered dataset saved to {output_file}")