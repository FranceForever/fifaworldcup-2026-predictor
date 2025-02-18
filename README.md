Data Collection & Preprocessing
For this project, I use the "International Football Results from 1872 to 2017" dataset from Kaggle as the primary data source. The dataset was downloaded using the Kaggle CLI and contains several CSV files, including results.csv, which holds match-level data (dates, teams, scores, tournaments, etc.).

To focus on competitive international matches relevant to predicting the World Cup 2026, I implemented a data filtering pipeline that:

Converts dates to a standardized datetime format.
Filters out matches prior to 2010 to ensure the dataset reflects recent performance.
Selects only competitive tournaments such as FIFA World Cup, World Cup qualification, UEFA Euro (and its qualifiers), Copa América (and its qualifiers), UEFA Nations League, and CONMEBOL Nations League.
Excludes friendlies, ensuring that only competitive matches are considered.
Saves the cleaned data as filtered_competitive_matches.csv for further feature engineering and model training.
The filtering script (scripts/filter_data.py) verifies the presence of the dataset file, applies these filters, and outputs the processed data, which forms the foundation for subsequent model development.