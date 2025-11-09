"""
Script de test du pipeline ML complet.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.data_preprocessing import RealEstatePreprocessor, load_data
from src.models import ModelTrainer, get_default_models
from src.evaluation import generate_evaluation_report
import warnings
warnings.filterwarnings('ignore')


def test_preprocessing():
    """
    Teste le module de preprocessing.
    """
    print("="*80)
    print("TEST 1: PREPROCESSING")
    print("="*80)

    # Charger les données
    print("\n1. Chargement des donnees...")
    df = load_data('data/train_data_clean.csv')
    print(f"   [OK] {len(df)} echantillons charges")

    # Initialiser le preprocessor
    print("\n2. Initialisation du preprocessor...")
    preprocessor = RealEstatePreprocessor()
    print("   [OK] Preprocessor initialise")

    # Preprocesser les données
    print("\n3. Preprocessing des donnees...")
    X_train, y_train = preprocessor.preprocess(df, target_col='price', fit=True)
    print(f"   [OK] X_train shape: {X_train.shape}")
    print(f"   [OK] y_train shape: {y_train.shape}")

    # Vérifier les features
    print("\n4. Verification des features...")
    print(f"   Nombre de features: {X_train.shape[1]}")
    print(f"   Features: {X_train.columns.tolist()[:10]}...")  # Afficher les 10 premières

    # Vérifier qu'il n'y a pas de NaN
    if X_train.isnull().sum().sum() == 0:
        print("   [OK] Aucune valeur manquante dans X_train")
    else:
        print(f"   [WARNING] {X_train.isnull().sum().sum()} valeurs manquantes detectees")

    # Test sur données de test
    print("\n5. Test sur donnees de test...")
    df_test = load_data('data/test_data_clean.csv')
    X_test, y_test = preprocessor.preprocess(df_test, target_col='price', fit=False)
    print(f"   [OK] X_test shape: {X_test.shape}")
    print(f"   [OK] y_test shape: {y_test.shape}")

    print("\n[SUCCESS] Test preprocessing termine avec succes!")

    return X_train, y_train, X_test, y_test, preprocessor


def test_model_training(X_train, y_train, X_test, y_test):
    """
    Teste l'entraînement des modèles.
    """
    print("\n" + "="*80)
    print("TEST 2: ENTRAINEMENT DES MODELES")
    print("="*80)

    # Initialiser le trainer
    print("\n1. Initialisation du trainer...")
    trainer = ModelTrainer()
    print("   [OK] Trainer initialise")

    # Ajouter quelques modèles (pas tous pour gagner du temps)
    print("\n2. Ajout des modeles...")
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10,
                                              random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                                random_state=42, n_jobs=-1)
    }

    for name, model in models.items():
        trainer.add_model(name, model)
        print(f"   [OK] Modele ajoute: {name}")

    # Entraîner les modèles
    print("\n3. Entrainement des modeles...")
    results = trainer.train_all(X_train, y_train, X_test, y_test)

    print("\n4. Resultats:")
    results_df = pd.DataFrame(results).T if isinstance(results, dict) else results
    # Tri par R2 (nom de colonne flexible)
    r2_col = 'R2' if 'R2' in results_df.columns else 'R2_test'
    results_df = results_df.sort_values(r2_col, ascending=False)
    print(results_df)

    # Sauvegarder les résultats
    results_df.to_csv('models/test_results.csv')
    print("\n   [OK] Resultats sauvegardes: models/test_results.csv")

    # Récupérer le meilleur modèle
    best_model_name = results_df.index[0]
    best_model = trainer.models[best_model_name]

    rmse_col = 'RMSE' if 'RMSE' in results_df.columns else 'RMSE_test'

    print(f"\n[SUCCESS] Meilleur modele: {best_model_name}")
    print(f"   R2: {results_df.loc[best_model_name, r2_col]:.4f}")
    print(f"   RMSE: {results_df.loc[best_model_name, rmse_col]:,.2f}")

    return best_model, results


def test_prediction(best_model, X_test, y_test):
    """
    Teste les prédictions.
    """
    print("\n" + "="*80)
    print("TEST 3: PREDICTIONS")
    print("="*80)

    # Faire des prédictions
    print("\n1. Generation des predictions...")
    y_pred = best_model.predict(X_test)
    print(f"   [OK] {len(y_pred)} predictions generees")

    # Évaluer
    print("\n2. Evaluation...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"   RMSE: {rmse:,.2f} EUR")
    print(f"   MAE: {mae:,.2f} EUR")
    print(f"   R2: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    # Afficher quelques exemples
    print("\n3. Exemples de predictions:")
    comparison = pd.DataFrame({
        'Prix_Reel': y_test.values[:10],
        'Prix_Predit': y_pred[:10],
        'Erreur_Abs': np.abs(y_test.values[:10] - y_pred[:10]),
        'Erreur_Pct': np.abs((y_test.values[:10] - y_pred[:10]) / y_test.values[:10]) * 100
    })
    print(comparison.to_string())

    print("\n[SUCCESS] Test predictions termine avec succes!")


def run_all_tests():
    """
    Exécute tous les tests du pipeline.
    """
    print("\n" + "="*80)
    print("TESTS DU PIPELINE ML COMPLET")
    print("="*80)

    try:
        # Test 1: Preprocessing
        X_train, y_train, X_test, y_test, preprocessor = test_preprocessing()

        # Test 2: Training
        best_model, results = test_model_training(X_train, y_train, X_test, y_test)

        # Test 3: Prediction
        test_prediction(best_model, X_test, y_test)

        # Résumé final
        print("\n" + "="*80)
        print("RESUME FINAL")
        print("="*80)
        print("[SUCCESS] Tous les tests ont reussi!")
        print("\nLe pipeline est pret a etre utilise:")
        print("  1. Preprocessing: OK")
        print("  2. Training: OK")
        print("  3. Prediction: OK")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test echoue: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()

    if success:
        print("\n" + "="*80)
        print("PIPELINE VALIDE ET PRET!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("PIPELINE NON VALIDE - CORRECTIONS NECESSAIRES")
        print("="*80)
