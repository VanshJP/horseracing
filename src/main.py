import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
import shap
import os

# Create predictions folder if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Load the data
race_data = pd.read_csv('data/all_tracks_hackathon.csv', low_memory=False)

# Define target variable: predict if horse finishes first
race_data['target_win'] = (race_data['finish'] == 1).astype(int)

# Drop unnecessary columns
drop_cols = [
    'finish',          # leaking target
    'horse_name',      # not useful for model generalization
    'track_name',      # redundant with track_code
    'race_date',       # could be used but not critical
    'post_time',       # avoid time leakage
    'dollar_odds',     # optional for now
    'comment'          # free text, skip for now
]

features = race_data.drop(columns=drop_cols + ['target_win'])
target = race_data['target_win']

# Encode categorical variables
for col in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))

# Fill missing values
features = features.fillna(-999)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save XGBoost feature importance plot
plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=20, importance_type='weight', height=0.8)
plt.title("XGBoost Feature Importance (Weight)")
plt.tight_layout()
plt.savefig('predictions/xgboost_feature_importance.png')
plt.close()

# SHAP values for better interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Save SHAP summary plot
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.savefig('predictions/shap_feature_importance.png')
plt.close()

print("✅ Saved both graphs to the 'predictions' folder!")
