# Horse Racing Prediction System

An XGBoost-based prediction system for horse racing that analyzes how horses perform under different conditions to predict race outcomes.

## Directory Structure

```
horseracing/
├── data/                        # Data directory
│   ├── all_tracks_hackathon.csv # Historical race data (training data)
│   ├── keeneland_entries.csv    # Tomorrow's race entries (prediction targets)
│   └── data_dictionary.xlsx     # Descriptions of each column and abbreviations
│
├── xgb/                         # XGBoost prediction system
│   ├── models/                  # Trained models and feature lists
│   │   ├── race_predictor.json  # Saved model file
│   │   └── feature_list.txt     # List of features used by the model
│   │
│   ├── output/                  # Prediction outputs
│   │   └── predictions.csv      # Race predictions
│   │
│   ├── utils/                   # Utility modules
│   │   ├── data_processing.py   # Data preprocessing and feature engineering
│   │   └── evaluation.py        # Scoring and performance tracking
│   │
│   ├── models/                  # Model definitions
│   │   └── xgb_model.py         # XGBoost model implementation
│   │
│   └── predict.py               # Main prediction script
│
└── README.md                    # This file
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Make sure your data files are in the correct locations:
   - `data/all_tracks_hackathon.csv` - Historical race data for training
   - `data/keeneland_entries.csv` - Tomorrow's race entries for prediction
   - `data/data_dictionary.xlsx` - Column descriptions and abbreviations

## XGBoost Prediction System

This system predicts the top 3 finishers in horse races by analyzing how horses perform in specific conditions. It's optimized for the scoring system that awards:
- +1 point for each correct pick in the top 3 (regardless of order)
- +1 point for each correct pick in the correct position
- +1 bonus point if all top 3 picks are in the correct order

### How It Works

The system creates performance metrics for each horse under various conditions:
- How well they perform on different surfaces (dirt, turf)
- Performance in different track conditions (fast, muddy, sloppy)
- Performance at different distances
- Success with specific jockeys
- Track-specific advantages
- Recent form

### Usage

#### Making Predictions

To train the model and make predictions for tomorrow's races:

```bash
python xgb/predict.py
```

This will:
1. Load historical data from `data/all_tracks_hackathon.csv`
2. Train an XGBoost model on horse performance metrics
3. Load tomorrow's races from `data/keeneland_entries.csv`
4. Make predictions for each race
5. Save predictions to `xgb/output/predictions.csv`

#### Understanding Output

The prediction output includes:
- Race information (date, track, race number)
- Top 3 predicted finishers for each race
- Confidence scores for each prediction
- Key factors behind each prediction (e.g., "Win rate on dirt: 0.33")

### Model Details

The `xgb/models/feature_list.txt` file contains all features used to train the model. These features are engineered from the raw data to capture how horses perform in different conditions.

### Troubleshooting

- If you see an error about missing data files, check that your data is in the correct location (`data/` directory)
- For large CSV files, the system uses `low_memory=False` to prevent data type issues
- If you get an error about unsupported data types, check the data types in your CSV files

All Tracks Data:

race_number,race_type,purse,distance,distance_unit,course,surface,track_condition,weather,post_time,win_time,horse_name,breed,weight,age,sex,medication,program_num,post_position,finish,comment,jockey,trainer,owner,last_race_track,last_race_date,last_race_number,last_race_finish,track_code,track_name,race_date,dollar_odds,num_past_starts,num_past_wins,num_past_seconds,num_past_thirds