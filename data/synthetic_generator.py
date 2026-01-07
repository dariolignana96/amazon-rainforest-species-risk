"""
Amazon Rainforest Species Synthetic Data Generator
Dataset 100% originale di specie amazzoniche.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class AmazonSpeciesGenerator:
    def __init__(self, n_species: int = 1000, seed: int = 42):
        self.n_species = n_species
        np.random.seed(seed)

    def generate(self):
        print(f"🌿 Generando {self.n_species} specie...")

        # Genera nomi
        prefixes = ["Morpho", "Jaguar", "Toucan", "Sloth", "Parrot"] * 200
        epithets = ["azul", "mont", "flor", "virid", "aure"] * 200
        names = [
            f"{np.random.choice(prefixes)} {np.random.choice(epithets)}"
            for _ in range(self.n_species)
        ]

        # Genera dati (tutte le colonne hanno ESATTAMENTE self.n_species elementi)
        data_dict = {
            "species_id": np.arange(1, self.n_species + 1),
            "species_name": names,
            "habitat": np.random.choice(["Canopy", "Floor", "River"], self.n_species),
            "population_size": np.random.exponential(10000, self.n_species).astype(int)
            + 10,
            "habitat_fragmentation": np.random.uniform(0, 1, self.n_species),
            "climate_vulnerability": np.random.uniform(0, 1, self.n_species),
            "illegal_hunting_pressure": np.random.uniform(0, 1, self.n_species),
            "conservation_efforts_index": np.random.uniform(0, 1, self.n_species),
            "breeding_program_exists": np.random.choice(
                [0, 1], self.n_species, p=[0.7, 0.3]
            ),
            "legal_protection": np.random.choice([0, 1], self.n_species, p=[0.6, 0.4]),
        }

        # Crea DataFrame
        df = pd.DataFrame(data_dict)

        # Calcola categoria IUCN
        def assign_risk(row):
            score = 0
            if row["population_size"] < 100:
                score += 3
            elif row["population_size"] < 500:
                score += 2
            elif row["population_size"] < 5000:
                score += 1

            if row["habitat_fragmentation"] > 0.7:
                score += 3
            elif row["habitat_fragmentation"] > 0.5:
                score += 2

            if row["climate_vulnerability"] > 0.7:
                score += 2

            if row["illegal_hunting_pressure"] > 0.6:
                score += 2

            score -= int(row["conservation_efforts_index"] * 2)

            if score <= 1:
                return "Least Concern"
            if score <= 3:
                return "Vulnerable"
            if score <= 4:
                return "Endangered"
            return "Critically Endangered"

        df["iucn_category"] = df.apply(assign_risk, axis=1)
        df["iucn_category_code"] = df["iucn_category"].map(
            {
                "Least Concern": 2,
                "Vulnerable": 3,
                "Endangered": 4,
                "Critically Endangered": 5,
            }
        )

        print(f"✅ Dataset generato: {len(df)} specie")
        print("📊 Distribuzione:")
        print(df["iucn_category"].value_counts())

        return df

    def save(self, df, path="data/raw/amazon_species.csv"):
        path_obj = Path(path)
        path_obj.parent.mkdir(exist_ok=True)
        df.to_csv(path_obj, index=False)
        print(f"💾 Salvato in {path_obj}")


if __name__ == "__main__":
    gen = AmazonSpeciesGenerator(n_species=1000)
    df = gen.generate()
    gen.save(df)
    print("\nPrime 5:")
    print(df.head())
