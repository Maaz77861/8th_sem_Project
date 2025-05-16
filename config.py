# Configuration parameters
HYPERPARAMS = {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_units": [128, 64],
    "dropout_rate": 0.3,
    "max_bio_length": 50,
    "text_vocab_size": 2000,
}

FEATURES = [
    "follower_ratio",
    "profile_completeness",
    "account_age_normalized",
    "post_count",
    "username_digits",
]
