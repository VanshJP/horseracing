import pandas as pd

# Read the CSV file
df = pd.read_csv('data/all_tracks_hackathon.csv')

# Get unique trainers and sort them alphabetically
unique_trainers = sorted(df['horse_name'].unique())

# Write trainers to a text file
with open('cool_info/horses.txt', 'w') as f:
    for trainer in unique_trainers:
        f.write(f"{trainer}\n")

print(f"Successfully wrote {len(unique_trainers)} unique trainers to trainers.txt") 