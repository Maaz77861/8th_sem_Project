import numpy as np
import tensorflow as tf
from models import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def plot_learning_curves(history, fold=None):
    metrics = ["loss", "accuracy", "precision", "recall", "auc", "f1_score"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Learning Curves {"- Fold "+str(fold) if fold is not None else ""}')

    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3

        axes[row, col].plot(history.history[metric], label="Training")
        axes[row, col].plot(history.history[f"val_{metric}"], label="Validation")
        axes[row, col].set_title(f"{metric.capitalize()} over epochs")
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].set_ylabel(metric.capitalize())
        axes[row, col].legend()
        axes[row, col].grid(True)

    plt.tight_layout()
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            plots_dir,
            f'learning_curves{"_fold_"+str(fold) if fold is not None else ""}.png',
        )
    )
    plt.close()


def train_model():
    # Create directory for model checkpoints
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load preprocessed data
    X_train = np.load(os.path.join(models_dir, "X_train.npy"))
    y_train = np.load(os.path.join(models_dir, "y_train.npy"))
    X_test = np.load(os.path.join(models_dir, "X_test.npy"))
    y_test = np.load(os.path.join(models_dir, "y_test.npy"))

    # K-fold cross validation
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")

        # Split data
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        # Create model
        model = create_model((X_train.shape[1],))

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                os.path.join(models_dir, f"best_model_fold_{fold}.h5"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ]

        # Train
        history = model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        # Plot learning curves for this fold
        plot_learning_curves(history, fold)

        # Evaluate on validation set
        val_scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        fold_scores.append(val_scores)

        # Print fold results
        print(f"\nFold {fold + 1} Results:")
        for metric, score in zip(model.metrics_names, val_scores):
            print(f"{metric}: {score:.4f}")

    # Print average results across folds
    print("\nAverage Results Across Folds:")
    mean_scores = np.mean(fold_scores, axis=0)
    std_scores = np.std(fold_scores, axis=0)
    for i, metric in enumerate(model.metrics_names):
        print(f"{metric}: {mean_scores[i]:.4f} ± {std_scores[i]:.4f}")

    # Train final model on all training data
    final_model = create_model((X_train.shape[1],))
    final_callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(models_dir, "best_model_final.h5"),
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    final_history = final_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=final_callbacks,
        verbose=1,
    )

    # Plot final learning curves
    plot_learning_curves(final_history)

    # Evaluate on test set
    test_results = final_model.evaluate(X_test, y_test, verbose=0)
    print("\nFinal Model Test Results:")
    for metric, score in zip(final_model.metrics_names, test_results):
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    train_model()
