import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Optional, Any, Dict, Union
import pandas as pd
import numpy as np
import os

class RandomForestModel:
    """
    Class for the scikit-learn RandomForestClassifier.

    Provides methods for training, prediction, probability prediction,
    and saving/loading the model.
    """
    MODEL_TYPE = "RandomForestClassifier"

    def __init__(self, **kwargs: Any):
        """
        Initialize the RandomForestModel and the underlying scikit-learn model.
        :param kwargs: Arbitrary keyword arguments passed directly to the
        scikit-learn RandomForestClassifier constructor.
        Examples: n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1, etc.
        If 'random_state' is not provided, it defaults to 42 for reproducibility.
        """
        # Ensure reproducibility if random_state is not explicitly set
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42

        # Store parameters used for initialization
        self.params: Dict[str, Any] = kwargs

        # Initialize scikit-learn model
        self.model: Optional[RandomForestClassifier] = RandomForestClassifier(**self.params)
        self._is_trained: bool = False

    def train(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the RandomForest model using the provided training data.
        :param X_train: Training features (pandas DataFrame or NumPy array)
        :param y_train: Training labels (pandas Series or NumPy array)
        :return:
        """
        self.model.fit(X_train, y_train)
        self._is_trained = True


    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:

        """
        Make predictions on class labels on new data using the trained model.
        :param X: Features for prediction (pandas DataFrame or NumPy array),
        must have the same features as the training data.
        :return: NumPy array of predicted class labels, or None if prediction fails.
        """
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:

        """
        Predicts class probabilities for new data using the trained model.
        :param X: Features for prediction (pandas DataFrame or NumPy array),
        must have the same features as the training data.
        :return: Numpy array of shape [n_samples, n_classes] with class probabilities,
        or None if prediction fails. Typically, column 0 is class 0, column 1
        is class 1
        """
        probabilities = self.model.predict_proba(X)
        return probabilities

    def save_model(self, filepath: str) -> None:
        """
        Saves the trained scikit-learn model object to a file using joblib.
        :param filepath: The path to save the model to.
        :return:
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestModel':
        """
        Loads a pre-trained scikit-learn RandomForestClassifier model from a
        file using joblib and wraps it in a RandomForestModel instance.
        :param filepath: The path to the saved model file
        :return: An instance of RandomForestModel containing the loaded model.
        """
        loaded_sklearn_model = joblib.load(filepath)

        # Create a new instance of our wrapper
        model = cls(**loaded_sklearn_model.get_params())

        # Assign loaded model and set trained flag
        model.model = loaded_sklearn_model
        model._is_trained = True

        return model


# --- Example Usage ---
if __name__ == '__main__':
    # Load dataset
    train_path = "../storage/datasets/dataset_train_2025-04-29-20-21-59.csv"
    test_path = "../storage/datasets/dataset_test_2025-04-29-20-21-59.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Get features and target
    y_train = train_df['is_fraud']
    X_train = train_df.drop(columns=['is_fraud'])
    y_test = test_df['is_fraud']
    X_test = test_df.drop(columns=['is_fraud'])

    # Remove cc_num from features
    X_train = X_train.drop(columns=['cc_num'], errors='ignore')
    X_test = X_test.drop(columns=['cc_num'], errors='ignore')

    # Add this line
    final_training_features = X_train.columns.tolist()
    print(final_training_features)
    print(f"Number of features: {len(final_training_features)}")

    # Initialize custom RandomForestModel
    print("\nInitializing RandomForestModel...")
    rf_wrapper = RandomForestModel(n_estimators=120, max_depth=15, class_weight='balanced_subsample', n_jobs=-1, random_state=123)

    # Train model
    print("\nTraining model...")
    rf_wrapper.train(X_train, y_train)


    # Make predictions and get probabilities on test set
    print("\nMaking predictions on test data...")

    test_predictions = rf_wrapper.predict(X_test)
    test_probabilities = rf_wrapper.predict_proba(X_test)
    # Probability of the positive class (fraud) is typically the second column
    test_fraud_probabilities = test_probabilities[:, 1] if test_probabilities is not None else None

    # Evaluate model
    print("\n5. Evaluating the model...")
    if test_predictions is not None and test_fraud_probabilities is not None:
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions)
        recall = recall_score(y_test, test_predictions)
        auc = roc_auc_score(y_test, test_fraud_probabilities)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
    else:
        print("Evaluation skipped due to prediction errors.")


    # Save the trained model
    print("\nSaving the trained model...")
    model_directory = "saved_models"
    model_filename = "securebank_rf.joblib"
    model_filepath = os.path.join(model_directory, model_filename)

    rf_wrapper.save_model(model_filepath)

    # Load the model from the file into a new instance
    print("\nLoading the model...")

    loaded_rf_wrapper = RandomForestModel.load_model(model_filepath)

    # Verify the loaded model by making predictions again
    print("\nVerifying loaded model...")
    loaded_test_predictions = loaded_rf_wrapper.predict(X_test)

    if loaded_test_predictions is not None:
        loaded_accuracy = accuracy_score(y_test, loaded_test_predictions)
        print(f"Accuracy of loaded model: {loaded_accuracy:.4f}")
        # Simple check if predictions match
        if np.array_equal(test_predictions, loaded_test_predictions):
            print("Verification successful: Predictions match original model.")
        else:
            print("Verification failed: Predictions differ from original model.")
    else:
        print("Verification skipped due to prediction error with loaded model.")
