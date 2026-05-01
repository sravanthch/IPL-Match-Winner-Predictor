import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("Loading dataset...")
# Load the ball-by-ball dataset from the Dataset folder
# Update 'IPL.csv' if your file has a different name
df = pd.read_csv(r"Dataset/IPL.csv", low_memory=False)

print("Extracting match-level data...")
# We need to get match level information.
# For each match_id, we can just grab the first row to get the match result, toss info, and venue
match_df = df.drop_duplicates(subset=['match_id']).copy()

# The first row of each match will give us the batting and bowling team for the 1st innings.
# Let's call them Team 1 and Team 2.
match_df['team1'] = match_df['batting_team']
match_df['team2'] = match_df['bowling_team']

# Features we want to use
features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
target = 'match_won_by'

# Prune missing target values (e.g. no result, ties if not decided)
match_df = match_df.dropna(subset=[target] + features)

# Fix team names (e.g., Delhi Daredevils -> Delhi Capitals, Kings XI Punjab -> Punjab Kings)
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans', # Roughly
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants'
}

for col in ['team1', 'team2', 'toss_winner', 'match_won_by']:
    match_df[col] = match_df[col].replace(team_mapping)

# Fix duplicate and legacy venue names
venue_mapping = {
    'Arun Jaitley Stadium': 'Arun Jaitley Stadium, Delhi',
    'Feroz Shah Kotla': 'Arun Jaitley Stadium, Delhi',
    'Brabourne Stadium': 'Brabourne Stadium, Mumbai',
    'Dr DY Patil Sports Academy': 'Dr DY Patil Sports Academy, Mumbai',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
    'Eden Gardens': 'Eden Gardens, Kolkata',
    'Himachal Pradesh Cricket Association Stadium': 'Himachal Pradesh Cricket Association Stadium, Dharamsala',
    'M Chinnaswamy Stadium, Bengaluru': 'M. Chinnaswamy Stadium, Bengaluru',
    'M Chinnaswamy Stadium': 'M. Chinnaswamy Stadium, Bengaluru',
    'M.Chinnaswamy Stadium': 'M. Chinnaswamy Stadium, Bengaluru',
    'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chennai',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chennai',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium, Chennai',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh',
    'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium, Pune',
    'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium, Pune',
    'Narendra Modi Stadium': 'Narendra Modi Stadium, Ahmedabad',
    'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium, Ahmedabad',
    'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association Stadium, Mohali',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali',
    'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Hyderabad',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium, Hyderabad',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium, Hyderabad',
    'Sawai Mansingh Stadium': 'Sawai Mansingh Stadium, Jaipur',
    'Wankhede Stadium': 'Wankhede Stadium, Mumbai',
    'Sheikh Zayed Stadium': 'Sheikh Zayed Stadium, Abu Dhabi',
    'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium, Abu Dhabi'
}

match_df['venue'] = match_df['venue'].replace(venue_mapping)

# Prepare encoding
print("Encoding categorical features...")
encoders = {}
for col in features:
    encoder = LabelEncoder()
    # Fit on all possible values so we don't miss anything
    all_values = match_df[col].unique().tolist()
    # It's possible some teams/venues never won or something, so better to be safe
    encoder.fit(all_values)
    match_df[col + '_encoded'] = encoder.transform(match_df[col])
    encoders[col] = encoder

# Encode target
target_encoder = LabelEncoder()
match_df[target] = target_encoder.fit_transform(match_df[target])
encoders['target'] = target_encoder

encoded_features = [f + '_encoded' for f in features]
X = match_df[encoded_features]
y = match_df[target]

print(f"Training model on {len(X)} matches...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f"Model Accuracy on Test Set: {accuracy_score(y_test, y_pred):.2f}")

# Save the model and encoders
print("Saving model and encoders...")
joblib.dump(model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("Done! model.pkl and encoders.pkl have been created.")
