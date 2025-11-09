"""
Models Module
Contient les wrappers et utilities pour l'entraînement des modèles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import joblib
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class ModelTrainer:
    """
    Classe pour entraîner et comparer plusieurs modèles.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def add_model(self, name: str, model: Any):
        """
        Ajoute un modèle à la collection.

        Args:
            name: nom du modèle
            model: instance du modèle sklearn-compatible
        """
        self.models[name] = model

    def train_model(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Entraîne un modèle spécifique.

        Args:
            name: nom du modèle
            X_train: features d'entraînement
            y_train: target d'entraînement

        Returns:
            modèle entraîné
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Add it first with add_model().")

        print(f"Training {name}...")
        self.models[name].fit(X_train, y_train)
        return self.models[name]

    def evaluate_model(self, name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Évalue un modèle sur le test set.

        Args:
            name: nom du modèle
            X_test: features de test
            y_test: target de test

        Returns:
            dictionnaire de métriques
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")

        y_pred = self.models[name].predict(X_test)

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,
            'R2': r2_score(y_test, y_pred)
        }

        self.results[name] = metrics
        return metrics

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Entraîne et évalue tous les modèles.

        Args:
            X_train, y_train: données d'entraînement
            X_test, y_test: données de test

        Returns:
            DataFrame avec résultats comparatifs
        """
        for name in self.models.keys():
            self.train_model(name, X_train, y_train)
            self.evaluate_model(name, X_test, y_test)

            # Afficher résultats
            print(f"\n{name}:")
            for metric, value in self.results[name].items():
                if metric == 'R2':
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value:,.2f}")

        # Créer DataFrame de résultats
        results_df = pd.DataFrame(self.results).T.sort_values('R2', ascending=False)

        # Identifier le meilleur modèle
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]

        print(f"\n[BEST] Best Model: {self.best_model_name}")
        print(f"   R²: {results_df.iloc[0]['R2']:.4f}")

        return results_df

    def optimize_hyperparameters(self, model_name: str, param_grid: Dict,
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 cv: int = 5, scoring: str = 'r2',
                                 search_type: str = 'grid') -> Any:
        """
        Optimise les hyperparamètres d'un modèle.

        Args:
            model_name: nom du modèle
            param_grid: grille de paramètres
            X_train, y_train: données d'entraînement
            cv: nombre de folds pour cross-validation
            scoring: métrique d'optimisation
            search_type: 'grid' ou 'random'

        Returns:
            meilleur modèle trouvé
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        base_model = self.models[model_name]

        if search_type == 'grid':
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring,
                                 n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(base_model, param_grid, cv=cv, scoring=scoring,
                                       n_jobs=-1, verbose=1, n_iter=50,
                                       random_state=self.random_state)

        print(f"\nOptimizing {model_name}...")
        search.fit(X_train, y_train)

        print(f"\nBest parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"\nBest CV score ({scoring}): {search.best_score_:.4f}")

        # Remplacer le modèle par la version optimisée
        optimized_name = f"{model_name} (Optimized)"
        self.models[optimized_name] = search.best_estimator_

        return search.best_estimator_

    def save_model(self, model_name: str, filepath: str):
        """Sauvegarde un modèle."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        joblib.dump(self.models[model_name], filepath)
        print(f"✅ Model saved: {filepath}")

    def load_model(self, filepath: str, model_name: str = None):
        """Charge un modèle sauvegardé."""
        model = joblib.load(filepath)
        if model_name:
            self.models[model_name] = model
        return model

    def save_results(self, filepath: str):
        """Sauvegarde les résultats en CSV."""
        if not self.results:
            raise ValueError("No results to save. Train models first.")
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(filepath)
        print(f"✅ Results saved: {filepath}")


class ModelEvaluator:
    """
    Classe pour évaluation avancée des modèles.
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule toutes les métriques de régression.

        Args:
            y_true: valeurs réelles
            y_pred: prédictions

        Returns:
            dictionnaire de métriques
        """
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R2': r2_score(y_true, y_pred),
            'Max_Error': np.max(np.abs(y_true - y_pred)),
            'Mean_Error': np.mean(y_true - y_pred)
        }

    @staticmethod
    def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                            cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Validation croisée avec plusieurs métriques.

        Args:
            model: modèle à évaluer
            X, y: données
            cv: nombre de folds

        Returns:
            dictionnaire de scores
        """
        scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        scores = {}

        for metric in scoring:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            scores[metric] = cv_scores

        return scores

    @staticmethod
    def get_feature_importance(model: Any, feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
        """
        Extrait l'importance des features.

        Args:
            model: modèle entraîné avec feature_importances_
            feature_names: noms des features
            top_n: nombre de top features à retourner

        Returns:
            DataFrame avec importance triée
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute.")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)


# Fonction pour créer des modèles pré-configurés

def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Retourne un dictionnaire de modèles pré-configurés.

    Args:
        random_state: seed

    Returns:
        dictionnaire {nom: modèle}
    """
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_state=random_state,
            verbose=0
        )
    }

    return models


def get_xgboost_param_grid() -> Dict[str, List]:
    """
    Retourne une grille de paramètres pour XGBoost.

    Returns:
        dictionnaire de paramètres
    """
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
