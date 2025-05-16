import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential


def create_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.F1Score(name="f1_score", threshold=0.5, average="micro"),
        ],
    )

    return model
