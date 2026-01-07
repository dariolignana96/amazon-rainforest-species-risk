"""
Preprocessing pipeline per Amazon Species Risk Predictor.
Scaling, encoding, train/test split.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class AmazonPreprocessor:
    """Pipeline di preprocessing per dataset specie amazzoniche."""

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
        """Crea sklearn ColumnTransformer per preprocessing."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first"
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
            ]
        )
        return self.preprocessor

    def fit_and_transform(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """
        Fit preprocessor e splitta train/test.
        
        Returns:
            (X_train_transformed, X_test_transformed, y_train, y_test, feature_names)
        """
        self.create_preprocessor()

        X = df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        y = df[self.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        numeric_names = self.NUMERIC_FEATURES
        categorical_names = self.preprocessor.named_transformers_["cat"].get_feature_names_out(
            self.CATEGORICAL_FEATURES
        ).tolist()
        self.feature_names = numeric_names + categorical_names

        print(f" Preprocessing completato")
        print(f"   Train shape: {X_train_transformed.shape}")
        print(f"   Test shape:  {X_test_transformed.shape}")
        print(f"   Features: {len(self.feature_names)}")

        return X_train_transformed, X_test_transformed, y_train, y_test, self.feature_names

    def save_preprocessor(self, path: str = "models/preprocessor.pkl"):
        """Salva preprocessor per inference."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)
        print(f" Preprocessor salvato in {path}")

    def load_preprocessor(self, path: str = "models/preprocessor.pkl"):
        """Carica preprocessor salvato."""
        self.preprocessor = joblib.load(path)
        print(f" Preprocessor caricato da {path}")
        return self.preprocessor


if __name__ == "__main__":
    from data.synthetic_generator import AmazonSpeciesGenerator
    
    gen = AmazonSpeciesGenerator(n_species=100, seed=42)
    df = gen.generate()
    
    preprocessor = AmazonPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.fit_and_transform(df)
    
    print(f"\n Preprocessing test completato!")
