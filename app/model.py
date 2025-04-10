#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras import backend as K


# Custom loss function for cleavage site prediction
def masked_mse_loss(y_true, y_pred):
    """
    Custom MSE loss that ignores examples with zero values in the target

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE loss
    """
    # Create a mask for non-zero values in y_true (assuming 0 is the mask value)
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())

    # Apply the mask to both y_true and y_pred
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    # Calculate MSE only on unmasked values
    squared_difference = K.square(y_true_masked - y_pred_masked)

    # Sum of the mask gives us the number of non-masked values
    n = K.sum(mask) + K.epsilon()  # Add epsilon to avoid division by zero

    # Return the mean squared error
    return K.sum(squared_difference) / n


# Custom metrics for cleavage site prediction
def masked_mae(y_true, y_pred):
    """
    Custom MAE that ignores examples with zero values in the target

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE metric
    """
    # Create a mask for non-zero values in y_true
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())

    # Apply the mask to both y_true and y_pred
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    # Calculate MAE only on unmasked values
    absolute_difference = K.abs(y_true_masked - y_pred_masked)

    # Sum of the mask gives us the number of non-masked values
    n = K.sum(mask) + K.epsilon()  # Add epsilon to avoid division by zero

    # Return the mean absolute error
    return K.sum(absolute_difference) / n


class SignalPeptideModel:
    def __init__(self, max_length=100, n_features=20, n_kingdoms=5, n_sp_classes=8):
        """
        Initialize the Signal Peptide model

        Args:
            max_length: Maximum sequence length
            n_features: Number of features per amino acid (one-hot encoding)
            n_kingdoms: Number of kingdom categories
            n_sp_classes: Number of signal peptide classes
        """
        self.max_length = max_length
        self.n_features = n_features
        self.n_kingdoms = n_kingdoms
        self.n_sp_classes = n_sp_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Build and compile the multi-output model for signal peptide prediction

        Returns:
            Compiled Keras model
        """
        # Sequence input
        seq_input = Input(
            shape=(self.max_length, self.n_features), name="sequence_input"
        )

        # Kingdom input
        kingdom_input = Input(shape=(self.n_kingdoms,), name="kingdom_input")

        # Convolutional layers for sequence processing
        x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(
            seq_input
        )
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)

        # Flatten and concatenate with kingdom input
        x = Concatenate()([x, kingdom_input])

        # Shared fully connected layers
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)

        # Branch for SP classification
        sp_branch = Dense(64, activation="relu")(x)
        sp_output = Dense(1, activation="sigmoid", name="sp_prediction")(sp_branch)

        # Branch for SP class prediction
        sp_class_branch = Dense(64, activation="relu")(x)
        sp_class_output = Dense(
            self.n_sp_classes, activation="softmax", name="sp_class_prediction"
        )(sp_class_branch)

        # Branch for cleavage site prediction - using sigmoid and constraint to ensure output between 0 and 1
        cleavage_branch = Dense(64, activation="relu")(x)
        # Use sigmoid activation to constrain output to [0, 1] (will be denormalized later)
        cleavage_output = Dense(1, activation="sigmoid", name="cleavage_prediction")(
            cleavage_branch
        )

        # Create model with three outputs
        model = Model(
            inputs=[seq_input, kingdom_input],
            outputs=[sp_output, sp_class_output, cleavage_output],
        )

        # Compile model with appropriate loss functions
        model.compile(
            optimizer="adam",
            loss={
                "sp_prediction": "binary_crossentropy",
                "sp_class_prediction": "categorical_crossentropy",
                "cleavage_prediction": masked_mse_loss,  # Custom loss that ignores zero values
            },
            metrics={
                "sp_prediction": "accuracy",
                "sp_class_prediction": "accuracy",
                "cleavage_prediction": masked_mae,  # Custom metric that ignores zero values
            },
            # Weigh losses
            loss_weights={
                "sp_prediction": 1.0,
                "sp_class_prediction": 1.0,
                "cleavage_prediction": 0.5,
            },
        )

        return model

    def train(
        self,
        X_seq,
        X_kingdom,
        y_has_sp,
        y_sp_class,
        y_cleavage_normalized,
        batch_size=32,
        epochs=100,
        model_path="models",
        validation_split=0.2,
    ):
        """
        Train the model

        Args:
            X_seq: Encoded sequences
            X_kingdom: Encoded kingdom information
            y_has_sp: Signal peptide binary labels
            y_sp_class: One-hot encoded SP class labels
            batch_size: Training batch size
            epochs: Number of epochs
            model_path: Path to save model files
            validation_split: Fraction of data to use for validation

        Returns:
            Training history
        """
        # Create model directory
        os.makedirs(model_path, exist_ok=True)

        # Split data for training and validation
        indices = np.arange(len(X_seq))
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_split, random_state=42
        )

        X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
        X_kingdom_train, X_kingdom_val = X_kingdom[train_idx], X_kingdom[val_idx]
        y_has_sp_train, y_has_sp_val = y_has_sp[train_idx], y_has_sp[val_idx]
        y_sp_class_train, y_sp_class_val = y_sp_class[train_idx], y_sp_class[val_idx]
        y_cleavage_train, y_cleavage_val = (
            y_cleavage_normalized[train_idx],
            y_cleavage_normalized[val_idx],
        )

        # Set up callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(model_path, "sp_model_best.keras"),
            monitor="val_sp_prediction_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        early_stopping = EarlyStopping(
            monitor="val_sp_prediction_accuracy",
            patience=10,
            restore_best_weights=True,
            mode="max",
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
        )

        # Train the model
        history = self.model.fit(
            [X_seq_train, X_kingdom_train],
            [y_has_sp_train, y_sp_class_train, y_cleavage_train],
            validation_data=(
                [X_seq_val, X_kingdom_val],
                [y_has_sp_val, y_sp_class_val, y_cleavage_val],
            ),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1,
        )

        # Save final model
        self.model.save(os.path.join(model_path, "sp_model.keras"))

        # Save and plot training history
        self._save_training_history(history, model_path)

        return history

    def _save_training_history(self, history, model_path):
        """
        Save and plot training history

        Args:
            history: Training history object
            model_path: Path to save history files
        """
        # Save history to CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(model_path, "training_history.csv"), index=False)

        # Plot training metrics
        plt.figure(figsize=(15, 10))

        # Plot SP prediction accuracy
        plt.subplot(2, 3, 1)
        plt.plot(history.history["sp_prediction_accuracy"])
        plt.plot(history.history["val_sp_prediction_accuracy"])
        plt.title("SP Prediction Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")

        # Plot SP class prediction accuracy
        plt.subplot(2, 3, 2)
        plt.plot(history.history["sp_class_prediction_accuracy"])
        plt.plot(history.history["val_sp_class_prediction_accuracy"])
        plt.title("SP Class Prediction Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")

        # Plot cleavage site prediction MAE
        plt.subplot(2, 3, 3)
        plt.plot(history.history["cleavage_prediction_masked_mae"])
        plt.plot(history.history["val_cleavage_prediction_masked_mae"])
        plt.title("Cleavage Site Prediction MAE")
        plt.ylabel("MAE")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        # Plot SP prediction loss
        plt.subplot(2, 3, 4)
        plt.plot(history.history["sp_prediction_loss"])
        plt.plot(history.history["val_sp_prediction_loss"])
        plt.title("SP Prediction Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        # Plot SP class prediction loss
        plt.subplot(2, 3, 5)
        plt.plot(history.history["sp_class_prediction_loss"])
        plt.plot(history.history["val_sp_class_prediction_loss"])
        plt.title("SP Class Prediction Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        # Plot cleavage site prediction loss
        plt.subplot(2, 3, 6)
        plt.plot(history.history["cleavage_prediction_loss"])
        plt.plot(history.history["val_cleavage_prediction_loss"])
        plt.title("Cleavage Site Prediction Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(model_path, "training_history.png"))
        plt.close()

    def predict(self, X_seq, X_kingdom, max_length=100):
        """
        Make predictions with the model

        Args:
            X_seq: Encoded sequences
            X_kingdom: Encoded kingdom information
            max_length: Maximum sequence length used for denormalizing cleavage positions

        Returns:
            Dictionary with predictions
        """
        # Get model predictions
        sp_pred, sp_class_pred, cleavage_pred = self.model.predict([X_seq, X_kingdom])

        # Convert SP prediction to binary
        sp_binary = (sp_pred > 0.5).astype(int).flatten()

        # Get the most likely SP class
        sp_class_indices = np.argmax(sp_class_pred, axis=1)

        # Denormalize cleavage site predictions (from [0,1] to actual positions)
        cleavage_positions = np.round(cleavage_pred * max_length).astype(int).flatten()

        # Set cleavage position to 0 for sequences predicted to have no signal peptide
        cleavage_positions[sp_binary == 0] = 0

        return {
            "sp_prob": sp_pred.flatten(),
            "sp_binary": sp_binary,
            "sp_class_prob": sp_class_pred,
            "sp_class_indices": sp_class_indices,
            "cleavage_positions": cleavage_positions,
        }

    def load_model(self, model_path):
        """
        Load a pretrained model

        Args:
            model_path: Path to saved model
        """
        # Register custom loss and metric
        custom_objects = {"masked_mse_loss": masked_mse_loss, "masked_mae": masked_mae}

        self.model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects
        )
        return self


def parse_fasta_for_prediction(file_path):
    """
    Parse a FASTA file for prediction (without requiring annotation lines)

    Format:
    >Uniprot_AC|Kingdom|Type
    amino-acid sequence

    Also supports simpler FASTA formats:
    >SequenceID
    amino-acid sequence

    Returns:
        DataFrame with parsed data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    records = []
    with open(file_path, "r") as f:
        current_id = None
        current_seq = ""

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Process previous record if exists
                if current_id is not None and current_seq:
                    records.append(
                        {
                            "protein_id": protein_id,
                            "kingdom": kingdom,
                            "type": sp_type,
                            "sequence": current_seq,
                        }
                    )

                # Start new record
                current_id = line[1:]  # Remove '>' prefix

                # Extract protein ID, kingdom and type from header
                header_parts = current_id.split("|")
                protein_id = header_parts[0] if len(header_parts) > 0 else current_id
                kingdom = (
                    header_parts[1] if len(header_parts) > 1 else "EUKARYA"
                )  # Default to EUKARYA if not specified
                sp_type = header_parts[2] if len(header_parts) > 2 else "unknown"

                current_seq = ""
            else:
                # This is the amino acid sequence
                current_seq += line

        # Process the last record if exists
        if current_id is not None and current_seq:
            header_parts = current_id.split("|")
            protein_id = header_parts[0] if len(header_parts) > 0 else current_id
            kingdom = (
                header_parts[1] if len(header_parts) > 1 else "EUKARYA"
            )  # Default to EUKARYA if not specified
            sp_type = header_parts[2] if len(header_parts) > 2 else "unknown"

            records.append(
                {
                    "protein_id": protein_id,
                    "kingdom": kingdom,
                    "type": sp_type,
                    "sequence": current_seq,
                }
            )

    # If no records were found, check if the file might contain just a sequence without a proper header
    if not records and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            content = f.read().strip()
            if not content.startswith(">"):
                # Treat the whole content as a sequence
                records.append(
                    {
                        "protein_id": "seq1",
                        "kingdom": "EUKARYA",
                        "type": "unknown",
                        "sequence": content,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Ensure we have the required columns
    if len(df) == 0:
        df = pd.DataFrame({"protein_id": [], "kingdom": [], "type": [], "sequence": []})

    return df


def prepare_data_for_prediction(data_df, max_length=100):
    """
    Prepare data for prediction (no signal peptide annotations required)

    Args:
        data_df: DataFrame with protein sequences
        max_length: Maximum sequence length

    Returns:
        Dictionary with encoded sequences
    """
    from data_preprocessing import encode_sequences, encode_kingdom

    # Encode sequences
    X_seq = encode_sequences(data_df["sequence"].tolist(), max_length=max_length)

    # Encode kingdom information
    X_kingdom = encode_kingdom(data_df["kingdom"].tolist())

    return {
        "X_seq": X_seq,
        "X_kingdom": X_kingdom,
        "protein_ids": data_df["protein_id"].tolist(),
    }


def load_model_and_predict(model_path, X_seq, X_kingdom, max_length=100):
    """
    Load a trained model and make predictions

    Args:
        model_path: Path to the trained model
        X_seq: Encoded sequences
        X_kingdom: Encoded kingdom information
        max_length: Maximum sequence length

    Returns:
        Dictionary with predictions
    """
    model = SignalPeptideModel(max_length=max_length)
    model.load_model(model_path)
    return model.predict(X_seq, X_kingdom, max_length=max_length)


def predict_from_file(fasta_file, model_path, max_length=100):
    """
    Make predictions directly from a FASTA file

    Args:
        fasta_file: Path to FASTA file with sequences to predict
        model_path: Path to the trained model
        max_length: Maximum sequence length

    Returns:
        DataFrame with prediction results
    """
    # Parse FASTA file (without requiring annotation lines)
    data_df = parse_fasta_for_prediction(fasta_file)

    # Prepare data for prediction
    data = prepare_data_for_prediction(data_df, max_length=max_length)

    # Load model and make predictions
    predictions = load_model_and_predict(
        model_path, data["X_seq"], data["X_kingdom"], max_length
    )

    # Format results
    results_df = pd.DataFrame(
        {
            "ID": data["protein_ids"],
            "Has_Signal_Peptide": predictions["sp_binary"],
            "Signal_Peptide_Type": [
                (
                    ["Sec/SPI", "Tat/SPI", "Sec/SPII", "Other"][i]
                    if predictions["sp_binary"][j] == 1
                    else "None"
                )
                for j, i in enumerate(predictions["sp_class_indices"])
            ],
            "Cleavage_Position": predictions["cleavage_positions"],
        }
    )

    return results_df


def train_model_from_file(
    fasta_file, model_path="models", max_length=100, batch_size=32, epochs=100
):
    """
    Train a signal peptide prediction model from a FASTA file

    Args:
        fasta_file: Path to FASTA file with training data
        model_path: Path to save model files
        max_length: Maximum sequence length
        batch_size: Training batch size
        epochs: Number of epochs

    Returns:
        Trained model
    """
    from data_preprocessing import parse_fasta_file, prepare_data_for_training

    # Create directories
    os.makedirs(model_path, exist_ok=True)

    # Load and prepare data
    print(f"Loading data from {fasta_file}...")
    data_df = parse_fasta_file(fasta_file)

    print(f"Loaded {len(data_df)} sequences")
    print(f"Sequences with signal peptides: {data_df['has_sp'].sum()}")
    print(
        f"Sequences without signal peptides: {len(data_df) - data_df['has_sp'].sum()}"
    )

    # Print distribution of kingdoms and SP classes
    print("\nKingdom distribution:")
    print(data_df["kingdom"].value_counts())

    print("\nSignal peptide class distribution:")
    print(data_df["sp_class"].value_counts())

    # Prepare data for training
    print("\nPreparing data for training...")
    data = prepare_data_for_training(data_df, max_length=max_length)

    # Initialize and train model
    print("\nInitializing model...")
    model = SignalPeptideModel(max_length=max_length)

    print("\nTraining model...")
    history = model.train(
        X_seq=data["X_seq"],
        X_kingdom=data["X_kingdom"],
        y_has_sp=data["y_has_sp"],
        y_sp_class=data["y_sp_class"],
        y_cleavage_normalized=data["y_cleavage_normalized"],
        batch_size=batch_size,
        epochs=epochs,
        model_path=model_path,
    )

    print(f"\nTraining complete! Model saved to {model_path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train signal peptide prediction model"
    )
    parser.add_argument(
        "--train-fasta", default="data/train.fasta", help="Path to training FASTA file"
    )
    parser.add_argument(
        "--max-length", type=int, default=100, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--model-path", default="models", help="Path to save trained models"
    )

    args = parser.parse_args()

    train_model_from_file(
        fasta_file=args.train_fasta,
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
