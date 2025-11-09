"""
Data Preprocessing Module
Contient toutes les fonctions de preprocessing pour le pipeline ML
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict
import joblib


class RealEstatePreprocessor:
    """
    Classe pour préprocesser les données immobilières.
    """

    def __init__(self):
        self.scaler = None
        self.energy_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        self.fitted = False

    def cap_outliers_iqr(self, df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
        """
        Cap les outliers en utilisant la méthode IQR.

        Args:
            df: DataFrame
            column: nom de la colonne
            factor: multiplicateur IQR (défaut: 1.5)

        Returns:
            DataFrame avec outliers cappés
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée toutes les features engineered.

        Args:
            df: DataFrame brut

        Returns:
            DataFrame avec nouvelles features
        """
        df = df.copy()

        # Features numériques
        df['price_per_m2'] = df['price'] / df['surface_m2']
        df['surface_per_room'] = df['surface_m2'] / df['rooms']
        df['room_density'] = df['rooms'] / df['surface_m2']
        df['log_surface'] = np.log1p(df['surface_m2'])
        df['surface_squared'] = df['surface_m2'] ** 2

        # Features catégorielles
        df['age_category'] = pd.cut(df['age_years'],
                                    bins=[-1, 5, 15, 30, 100],
                                    labels=['Neuf', 'Récent', 'Moyen', 'Ancien'])

        df['surface_category'] = pd.cut(df['surface_m2'],
                                        bins=[0, 40, 70, 100, 1000],
                                        labels=['Studio', 'Moyen', 'Grand', 'Très_grand'])

        df['floor_category'] = df['floor'].apply(
            lambda x: 'RDC' if x == 0 else ('Bas' if x <= 2 else ('Moyen' if x <= 5 else 'Haut'))
        )

        # Score de confort
        df['comfort_score'] = (df['has_elevator'].astype(int) +
                               df['has_parking'].astype(int) +
                               df['has_balcony'].astype(int))

        # Interaction ville × surface
        df['city_surface_interaction'] = df['city'] + '_' + df['surface_category'].astype(str)

        return df

    def encode_features(self, df: pd.DataFrame, target_encoding_map: Dict = None) -> pd.DataFrame:
        """
        Encode les variables catégorielles.

        Args:
            df: DataFrame avec features
            target_encoding_map: mapping pour target encoding (optionnel)

        Returns:
            DataFrame encodé
        """
        df = df.copy()

        # One-Hot Encoding pour 'city'
        city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True)
        df = pd.concat([df, city_dummies], axis=1)

        # Ordinal Encoding pour 'energy_class'
        df['energy_class_encoded'] = df['energy_class'].map(self.energy_mapping)

        # One-Hot Encoding pour les catégories
        for col in ['age_category', 'surface_category', 'floor_category']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

        # Target Encoding pour interaction (si mapping fourni)
        if target_encoding_map is not None:
            df['city_surface_encoded'] = df['city_surface_interaction'].map(target_encoding_map)

        # Supprimer les colonnes originales
        cols_to_drop = ['city', 'energy_class', 'age_category', 'surface_category',
                        'floor_category', 'city_surface_interaction']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Normalise les features numériques.

        Args:
            df: DataFrame
            fit: Si True, fit le scaler (mode training). Si False, utilise scaler existant (mode inference)

        Returns:
            DataFrame avec features normalisées
        """
        df = df.copy()

        # Colonnes à scaler
        numeric_cols = ['surface_m2', 'rooms', 'age_years', 'floor', 'comfort_score',
                        'surface_per_room', 'room_density', 'log_surface', 'surface_squared',
                        'energy_class_encoded']

        # Filter pour les colonnes qui existent
        cols_to_scale = [col for col in numeric_cols if col in df.columns]

        if fit:
            self.scaler = RobustScaler()
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            self.fitted = True
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df

    def preprocess(self, df: pd.DataFrame, target_col: str = 'price',
                   fit: bool = False, target_encoding_map: Dict = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline complet de preprocessing.

        Args:
            df: DataFrame brut
            target_col: nom de la colonne cible
            fit: Si True, mode training. Si False, mode inference.
            target_encoding_map: mapping pour target encoding

        Returns:
            X (features), y (target)
        """
        # Traiter les outliers (seulement en training)
        if fit:
            for col in ['price', 'surface_m2', 'age_years']:
                if col in df.columns:
                    df = self.cap_outliers_iqr(df, col)

        # Feature engineering
        df = self.create_features(df)

        # Encoding
        df = self.encode_features(df, target_encoding_map)

        # Séparer X et y
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col, 'price_per_m2'], errors='ignore')
        else:
            y = None
            X = df.drop(columns=['price_per_m2'], errors='ignore')

        # Scaling
        X = self.scale_features(X, fit=fit)

        return X, y

    def save_scaler(self, filepath: str):
        """Sauvegarde le scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet.")
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath: str):
        """Charge un scaler sauvegardé."""
        self.scaler = joblib.load(filepath)
        self.fitted = True


# Fonctions utilitaires standalone

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.

    Args:
        filepath: chemin vers le fichier CSV

    Returns:
        DataFrame
    """
    return pd.read_csv(filepath)


def train_test_split_custom(df: pd.DataFrame, test_size: float = 0.2,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train/test personnalisé.

    Args:
        df: DataFrame
        test_size: proportion du test set
        random_state: seed

    Returns:
        train_df, test_df
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df
