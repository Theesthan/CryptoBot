# src/model_manager.py

import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import os


from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_SAVE_PATH, CONFIDENCE_THRESHOLD

# Global variable to store the reverse mapper for signal predictions
reverse_mapper = {0: -1, 1: 0, 2: 1} # Default, will be set during training if needed


def prepare_model_data(df, feature_cols, target_col=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE, use_adasyn=True):
    """
    Prepare data for model training with validation, stratified split, SMOTE and ADASYN.
    
    Args:
        use_adasyn (bool): If True, applies both SMOTE and ADASYN. If False, only SMOTE.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input df must be a pandas DataFrame')

    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found in DataFrame')

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Feature columns not found in DataFrame: {missing_cols}')

    try:
        X = df[feature_cols]
        y = df[target_col].astype(int)

        unique_classes = set(y.unique())
        expected_classes = {-1, 0, 1}
        if not unique_classes.issubset(expected_classes):
            logging.warning(f'Invalid target values found: {unique_classes}. Expected: {expected_classes}')

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logging.info(f"Data split successfully. Training: {X_train.shape}, Testing: {X_test.shape}")
        
        # Handle NaN values (time series features can create NaNs for first few rows)
        # Fill NaN with forward fill first, then backward fill, then with 0
        X_train = X_train.ffill().bfill().fillna(0)
        X_test = X_test.ffill().bfill().fillna(0)
        
        logging.info(f"NaN values handled. Training set shape: {X_train.shape}")
        logging.info(f"Class distribution before resampling (Train):\n{pd.Series(y_train).value_counts()}")

        # Apply SMOTE first
        smote = SMOTE(random_state=random_state, n_jobs=1)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logging.info(f"Class distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

        # Apply ADASYN if enabled
        if use_adasyn:
            try:
                adasyn = ADASYN(random_state=random_state, n_jobs=1)
                X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_smote, y_train_smote)
                logging.info(f"Class distribution after ADASYN:\n{pd.Series(y_train_balanced).value_counts()}")
            except Exception as e:
                logging.warning(f"ADASYN failed: {e}. Using SMOTE results only.")
                X_train_balanced, y_train_balanced = X_train_smote, y_train_smote
        else:
            X_train_balanced, y_train_balanced = X_train_smote, y_train_smote

        return X_train_balanced, X_test, y_train_balanced, y_test
    except Exception as e:
        logging.error(f'Error preparing model data: {e}', exc_info=True)
        raise


def train_xgboost_model(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE):
    """
    Trains and optimizes an XGBoost model using RandomizedSearchCV followed by GridSearchCV,
    evaluates it, and saves the best model.
    """
    global reverse_mapper

    try:
        # Map labels from [-1, 0, 1] -> [0, 1, 2]
        y_train_mapped = (y_train + 1).astype(int)
        y_test_mapped = (y_test + 1).astype(int)

        # Keep mapping consistent
        reverse_mapper = {0: -1, 1: 0, 2: 1}

        # Stage 1: RandomizedSearchCV for broad exploration (optimized for ~5 min runtime)
        param_distributions = {
            'max_depth': [3, 5],              # Reduced from 4 to 2 options
            'learning_rate': [0.05, 0.1],     # Reduced from 4 to 2 options
            'n_estimators': [100, 200],       # Reduced from 3 to 2 options
            'min_child_weight': [1, 3],       # Reduced from 3 to 2 options
            'gamma': [0, 0.1],                # Reduced from 3 to 2 options
            'subsample': [0.8],               # Fixed at optimal value
            'colsample_bytree': [0.8],        # Fixed at optimal value
            'reg_alpha': [0.1],               # Fixed at optimal value
            'reg_lambda': [1.0]               # Fixed at optimal value
        }

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores for faster training
        )

        logging.info("Stage 1: Starting RandomizedSearchCV for hyperparameter exploration (optimized for speed)...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=10,  # Reduced from 20 to 10 iterations
            scoring='f1_macro',
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),  # Reduced from 5 to 3 folds
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )

        random_search.fit(X_train, y_train_mapped)
        best_params_random = random_search.best_params_
        logging.info(f'RandomizedSearchCV completed. Best params: {best_params_random}')
        logging.info(f'Best F1 Macro score: {random_search.best_score_:.4f}')

        # Stage 2: GridSearchCV for fine-tuning around best parameters (fast)
        logging.info("Stage 2: Starting GridSearchCV for fine-tuning...")
        
        # Create minimal parameter grid for fast fine-tuning
        param_grid = {}
        for param, value in best_params_random.items():
            if param == 'learning_rate':
                # Only fine-tune learning rate
                lower = max(0.001, value * 0.9)
                upper = value * 1.1
                param_grid[param] = sorted(list(set([lower, value, upper])))
        
        logging.info(f'GridSearchCV parameter grid (fast): {param_grid}')
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )

        grid_search.fit(X_train, y_train_mapped)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        logging.info(f'GridSearchCV completed. Best params: {best_params}')
        logging.info(f'Best F1 Macro score: {grid_search.best_score_:.4f}')

        # Evaluate on test set
        y_pred_proba = best_model.predict_proba(X_test)
        y_pred_mapped = np.argmax(y_pred_proba, axis=1)
        max_probs = np.max(y_pred_proba, axis=1)

        # Below-threshold -> hold (mapped value 1)
        y_pred_mapped[max_probs < CONFIDENCE_THRESHOLD] = 1
        y_pred = pd.Series(y_pred_mapped, index=X_test.index).map(reverse_mapper)

        accuracy = accuracy_score(y_test, y_pred)
        logging.info('XGBoost model training completed.')
        logging.info(f'Final best parameters: {best_params}')
        logging.info(f'Test accuracy: {accuracy:.4f}')
        logging.info(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')

        # Save the best model and feature columns
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        if hasattr(best_model, "save_model"):
            best_model.save_model(MODEL_SAVE_PATH)
            logging.info(f"Best XGBoost model saved to {MODEL_SAVE_PATH}")
            
            # Save feature columns used for training
            feature_columns = X_train.columns.tolist()
            feature_path = MODEL_SAVE_PATH.replace('.json', '_features.pkl')
            import pickle
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_columns, f)
            logging.info(f"Feature columns saved to {feature_path}")
        else:
            logging.warning("Best model does not implement save_model(); skipping save.")

        return best_model, best_params
    except Exception as e:
        logging.error(f'Error training XGBoost model: {e}', exc_info=True)
        raise

def load_trained_model(model_path=MODEL_SAVE_PATH):
    """Loads a pre-trained XGBoost model and its feature columns."""
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logging.info(f"XGBoost model loaded from {model_path}")
        
        # Load feature columns if available
        feature_path = model_path.replace('.json', '_features.pkl')
        if os.path.exists(feature_path):
            import pickle
            with open(feature_path, 'rb') as f:
                feature_columns = pickle.load(f)
            logging.info(f"Feature columns loaded from {feature_path}: {len(feature_columns)} features")
            return model, feature_columns
        else:
            logging.warning(f"Feature columns file not found at {feature_path}. Model will use current config features.")
            return model, None
    except Exception as e:
        logging.error(f"Error loading XGBoost model from {model_path}: {str(e)}")
        raise

def make_predictions(model, X_data, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Generates trading signals from a trained model with a confidence threshold.
    Returns signals mapped back to -1, 0, 1.
    """
    global reverse_mapper

    if X_data.empty:
        logging.warning("Input data for prediction is empty.")
        return pd.Series([], dtype=int), np.array([])

    # Capability-based guard: enables unit tests with mocks
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba().")

    try:
        pred_proba = model.predict_proba(X_data)
        predictions_mapped = np.argmax(pred_proba, axis=1)
        max_probs = np.max(pred_proba, axis=1)

        # Below-threshold -> hold (mapped value 1)
        predictions_mapped[max_probs < confidence_threshold] = 1

        predictions = pd.Series(predictions_mapped, index=X_data.index).map(reverse_mapper)

        logging.info(f"Predictions generated with confidence threshold {confidence_threshold}.")
        logging.info(f"Prediction distribution:\n{predictions.value_counts()}")

        return predictions, max_probs
    except Exception as e:
        logging.error(f"Error making predictions: {e}", exc_info=True)
        raise