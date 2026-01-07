"""
Training script principale per Amazon Species Risk Predictor.
Versione standalone, NO import data.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib
from xgboost import XGBClassifier


class AmazonPreprocessor:
    """Pipeline preprocessing."""

    NUMERIC_FEATURES = [
        "population_size",
        "habitat_fragmentation",
        "climate_vulnerability",
        "illegal_hunting_pressure",
        "conservation_efforts_index",
    ]

    CATEGORICAL_FEATURES = [
        "habitat",
        "breeding_program_exists",
        "legal_protection",
    ]

    TARGET = "iucn_category_code"

    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def create_preprocessor(self):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="first"
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
            ]
        )
        return self.preprocessor

    def fit_and_transform(self, df: pd.DataFrame, test_size: float = 0.2):
        self.create_preprocessor()

        X = df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        y = df[self.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        numeric_names = self.NUMERIC_FEATURES
        categorical_names = (
            self.preprocessor.named_transformers_["cat"]
            .get_feature_names_out(self.CATEGORICAL_FEATURES)
            .tolist()
        )
        self.feature_names = numeric_names + categorical_names

        print(f" Preprocessing completato")
        print(f"   Train shape: {X_train_transformed.shape}")
        print(f"   Test shape:  {X_test_transformed.shape}")
        print(f"   Features: {len(self.feature_names)}")

        return (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            self.feature_names,
        )

    def save_preprocessor(self, path: str = "models/preprocessor.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)
        print(f" Preprocessor salvato in {path}")


class SpeciesRiskPredictor:
    """Ensemble di modelli."""

    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("🎯 Training XGBoost...")

        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            verbose=0,
            objective="multi:softprob",
        )

        self.xgb_model.fit(X_train, y_train)

        y_pred = self.xgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    XGBoost trained")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-score: {f1:.4f}")

        return {"accuracy": float(acc), "f1": float(f1)}

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print(" Training Random Forest...")

        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    Random Forest trained")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-score: {f1:.4f}")

        return {"accuracy": float(acc), "f1": float(f1)}

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        print(" Training Logistic Regression...")

        self.lr_model = LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, multi_class="multinomial"
        )

        self.lr_model.fit(X_train, y_train)

        y_pred = self.lr_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    Logistic Regression trained")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-score: {f1:.4f}")

        return {"accuracy": float(acc), "f1": float(f1)}

    def get_feature_importance(self, feature_names):
        if self.xgb_model is None:
            return {}

        importances = self.xgb_model.feature_importances_
        feature_importance = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }

        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]

        print(f"\n Top 10 feature importance:")
        for name, imp in top_features:
            print(f"   {name}: {imp:.4f}")

        return feature_importance

    def save_model(self, model_name: str = "xgboost", path: str = "models"):
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

        Path(path).mkdir(exist_ok=True)
        filepath = f"{path}/{model_name}_v1.pkl"
        joblib.dump(model, filepath)
        print(f" {model_name} salvato in {filepath}")


def main():
    print("=" * 70)
    print(" AMAZON RAINFOREST SPECIES RISK PREDICTOR - TRAINING PIPELINE")
    print("=" * 70)

    # ===== STEP 1: Carica Dataset =====
    print("\n STEP 1: Caricamento Dataset")
    print("-" * 70)

    csv_path = Path("data/raw/amazon_species.csv")
    if not csv_path.exists():
        print(" Dataset non trovato!")
        print("Esegui prima: python data/synthetic_generator.py")
        return

    df = pd.read_csv(csv_path)

    # AGGIUNGI QUESTE RIGHE:
    # Normalizza classi da [2,3,4,5] a [0,1,2,3]
    df["iucn_category_code"] = df["iucn_category_code"] - 2

    print(f"✅ Dataset caricato: {len(df)} specie, {len(df.columns)} colonne")
    print(f"   Classi IUCN: {sorted(df['iucn_category_code'].unique())}")

    # ===== STEP 2: Preprocessing =====
    print("\n STEP 2: Preprocessing")
    print("-" * 70)

    preprocessor = AmazonPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.fit_and_transform(df)

    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    preprocessor.save_preprocessor(str(model_path / "preprocessor.pkl"))

    # ===== STEP 3: Model Training =====
    print("\n STEP 3: Training Modelli ML")
    print("-" * 70)

    predictor = SpeciesRiskPredictor()

    xgb_results = predictor.train_xgboost(X_train, y_train, X_test, y_test)
    rf_results = predictor.train_random_forest(X_train, y_train, X_test, y_test)
    lr_results = predictor.train_logistic_regression(X_train, y_train, X_test, y_test)

    # ===== STEP 4: Feature Importance =====
    print("\n STEP 4: Feature Importance")
    print("-" * 70)

    feature_importance = predictor.get_feature_importance(feature_names)

    # ===== STEP 5: Salva Modelli =====
    print("\n STEP 5: Salvataggio Modelli")
    print("-" * 70)

    predictor.save_model("xgboost", str(model_path))
    predictor.save_model("random_forest", str(model_path))
    predictor.save_model("logistic_regression", str(model_path))

    # ===== STEP 6: Metadata =====
    metadata = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "iucn_categories": [
            "Least Concern",
            "Vulnerable",
            "Endangered",
            "Critically Endangered",
        ],
        "models": {
            "xgboost": xgb_results,
            "random_forest": rf_results,
            "logistic_regression": lr_results,
        },
    }

    metadata_path = model_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f" Metadata salvato in {metadata_path}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETATO!")
    print("=" * 70)
    print(f"\n Artifacts salvati:")
    print(f"   - Dataset: {csv_path}")
    print(f"   - Modelli: {model_path}/")
    print(f"   - Metadata: {metadata_path}")
    print(f"\n Prossimo step: Crea FastAPI app!")


if __name__ == "__main__":
    main()
