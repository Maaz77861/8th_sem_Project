import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_and_preprocess():
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

    # Load JSON
    with open(os.path.join(PROJECT_ROOT, "data/fake.json")) as f:
        fake_data = json.load(f)
    with open(os.path.join(PROJECT_ROOT, "data/real.json")) as f:
        real_data = json.load(f)

    # Create DataFrame
    df_fake = pd.DataFrame(fake_data)
    df_real = pd.DataFrame(real_data)
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Feature engineering
    df["follower_following_ratio"] = df["userFollowerCount"] / (
        df["userFollowingCount"] + 1
    )
    df["engagement_rate"] = df["userMediaCount"] / (df["userFollowerCount"] + 1)

    # Select features
    features = [
        "userFollowerCount",
        "userFollowingCount",
        "userBiographyLength",
        "userMediaCount",
        "userHasProfilPic",
        "userIsPrivate",
        "usernameDigitCount",
        "usernameLength",
        "follower_following_ratio",
        "engagement_rate",
    ]

    # Split data
    X = df[features]
    y = df["isFake"].values.reshape(-1, 1)  # Reshape labels to 2D

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    # Save artifacts
    joblib.dump(scaler, os.path.join(PROJECT_ROOT, "models/scaler.joblib"))
    np.save(os.path.join(PROJECT_ROOT, "models/X_train.npy"), X_train)
    np.save(os.path.join(PROJECT_ROOT, "models/X_test.npy"), X_test)
    np.save(os.path.join(PROJECT_ROOT, "models/y_train.npy"), y_train)
    np.save(os.path.join(PROJECT_ROOT, "models/y_test.npy"), y_test)


if __name__ == "__main__":
    load_and_preprocess()
