"""
Model definitions per Amazon Species Risk Predictor.
XGBoost, Random Forest, Logistic Regression.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


class SpeciesRiskPredictor:
    """Ensemble di modelli per predire rischio di estinzione."""

    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Addestra XGBoost (modello principale)."""
        print(" Training XGBoost...")
        
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            num_class=6,
            eval_metric="mlogloss",
            verbose=0
        )

        self.xgb_model.fit(X_train, y_train)

        y_pred = self.xgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    XGBoost trained")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-score: {f1:.4f}")

        return {"accuracy": acc, "f1": f1}

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Addestra Random Forest."""
        print(" Training Random Forest...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    Random Forest trained")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-score: {f1:.4f}")

        return {"accuracy": acc, "f1": f1}

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Addestra Logistic Regression (baseline)."""
        print(" Training Logistic Regression...")
        
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            multi_class="multinomial"
        )

        self.lr_model.fit(X_train, y_train)

        y_pred = self.lr_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    Logistic Regression trained")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-score: {f1:.4f}")

        return {"accuracy": acc, "f1": f1}

    def get_feature_importance(self, feature_names):
        """Estrai feature importance da XGBoost."""
        if self.xgb_model is None:
            return {}

        importances = self.xgb_model.feature_importances_
        feature_importance = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }

        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        print(f"\n Top 10 feature importance:")
        for name, imp in top_features:
            print(f"   {name}: {imp:.4f}")

        return feature_importance

    def save_model(self, model_name: str = "xgboost", path: str = "models"):
        """Salva modello."""
        if model_name == "xgboost":
            model = self.xgb_model
        elif model_name == "random_forest":
            model = self.rf_model
        elif model_name == "logistic_regression":
            model = self.lr_model
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if model is None:
            print(f"  {model_name} non addestrato")
            return

        filepath = f"{path}/{model_name}_v1.pkl"
        joblib.dump(model, filepath)
        print(f" {model_name} salvato in {filepath}")

    def load_model(self, model_name: str = "xgboost", path: str = "models"):
        """Carica modello salvato."""
        filepath = f"{path}/{model_name}_v1.pkl"
        if model_name == "xgboost":
            self.xgb_model = joblib.load(filepath)
        elif model_name == "random_forest":
            self.rf_model = joblib.load(filepath)
        elif model_name == "logistic_regression":
            self.lr_model = joblib.load(filepath)
        
        print(f" {model_name} caricato da {filepath}")

    def predict(self, X, model_name: str = "xgboost"):
        """Predici con modello scelto."""
        if model_name == "xgboost":
            return self.xgb_model.predict(X)
        elif model_name == "random_forest":
            return self.rf_model.predict(X)
        elif model_name == "logistic_regression":
            return self.lr_model.predict(X)

    def predict_proba(self, X, model_name: str = "xgboost"):
        """Predici probabilità."""
        if model_name == "xgboost":
            return self.xgb_model.predict_proba(X)
        elif model_name == "random_forest":
            return self.rf_model.predict_proba(X)
        elif model_name == "logistic_regression":
            return self.lr_model.predict_proba(X)
