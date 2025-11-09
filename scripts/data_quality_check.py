"""
Script de contrôle qualité et nettoyage des données immobilières.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_and_validate_data(file_path='data/real_estate_data.csv'):
    """
    Charge et valide les données.

    Args:
        file_path: chemin vers le fichier de données

    Returns:
        DataFrame validé
    """
    print("="*80)
    print("CHARGEMENT ET VALIDATION DES DONNEES")
    print("="*80)

    # Charger les données
    print(f"\nChargement depuis: {file_path}")
    df = pd.read_csv(file_path)

    print(f"Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"\nColonnes: {df.columns.tolist()}")

    return df


def check_data_quality(df):
    """
    Vérifie la qualité des données.

    Args:
        df: DataFrame à vérifier

    Returns:
        dict avec les résultats des vérifications
    """
    print("\n" + "="*80)
    print("CONTROLES DE QUALITE")
    print("="*80)

    results = {}

    # 1. Valeurs manquantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    results['missing'] = missing[missing > 0]

    print("\n1. VALEURS MANQUANTES")
    if len(results['missing']) == 0:
        print("   [OK] Aucune valeur manquante detectee")
    else:
        print(f"   [WARNING] {len(results['missing'])} colonnes avec valeurs manquantes:")
        for col, count in results['missing'].items():
            pct = missing_pct[col]
            print(f"      - {col}: {count} ({pct:.2f}%)")

    # 2. Doublons
    duplicates = df.duplicated().sum()
    results['duplicates'] = duplicates

    print("\n2. DOUBLONS")
    if duplicates == 0:
        print("   [OK] Aucun doublon detecte")
    else:
        print(f"   [WARNING] {duplicates} lignes dupliquees detectees")

    # 3. Valeurs aberrantes
    print("\n3. VALEURS ABERRANTES")

    # Prix
    price_issues = []
    if df['price'].min() < 1000:
        price_issues.append(f"Prix minimum tres bas: {df['price'].min():,.2f}")
    if df['price'].max() > 10_000_000:
        price_issues.append(f"Prix maximum tres eleve: {df['price'].max():,.2f}")

    results['price_issues'] = price_issues
    if len(price_issues) == 0:
        print("   Prix: [OK]")
    else:
        print("   Prix: [WARNING]")
        for issue in price_issues:
            print(f"      - {issue}")

    # Surface
    surface_issues = []
    if df['surface_m2'].min() < 9:
        surface_issues.append(f"Surface minimum tres petite: {df['surface_m2'].min()}")
    if df['surface_m2'].max() > 500:
        surface_issues.append(f"Surface maximum tres grande: {df['surface_m2'].max()}")

    results['surface_issues'] = surface_issues
    if len(surface_issues) == 0:
        print("   Surface: [OK]")
    else:
        print("   Surface: [WARNING]")
        for issue in surface_issues:
            print(f"      - {issue}")

    # Chambres
    rooms_issues = []
    if df['rooms'].min() < 1:
        rooms_issues.append(f"Nombre de pieces minimum: {df['rooms'].min()}")
    if df['rooms'].max() > 10:
        rooms_issues.append(f"Nombre de pieces maximum: {df['rooms'].max()}")

    results['rooms_issues'] = rooms_issues
    if len(rooms_issues) == 0:
        print("   Pieces: [OK]")
    else:
        print("   Pieces: [WARNING]")
        for issue in rooms_issues:
            print(f"      - {issue}")

    # Âge
    age_issues = []
    if df['age_years'].min() < 0:
        age_issues.append(f"Age negatif detecte: {df['age_years'].min()}")
    if df['age_years'].max() > 200:
        age_issues.append(f"Age tres eleve: {df['age_years'].max()}")

    results['age_issues'] = age_issues
    if len(age_issues) == 0:
        print("   Age: [OK]")
    else:
        print("   Age: [WARNING]")
        for issue in age_issues:
            print(f"      - {issue}")

    # 4. Cohérence des données
    print("\n4. COHERENCE DES DONNEES")

    coherence_issues = []

    # Prix au m² incohérent
    df_temp = df.copy()
    df_temp['price_per_m2'] = df_temp['price'] / df_temp['surface_m2']

    # Prix au m² trop bas ou trop élevé
    low_price_m2 = df_temp[df_temp['price_per_m2'] < 500].shape[0]
    high_price_m2 = df_temp[df_temp['price_per_m2'] > 30000].shape[0]

    if low_price_m2 > 0:
        coherence_issues.append(f"{low_price_m2} biens avec prix/m2 < 500 EUR")
    if high_price_m2 > 0:
        coherence_issues.append(f"{high_price_m2} biens avec prix/m2 > 30000 EUR")

    # Surface par pièce incohérente
    df_temp['surface_per_room'] = df_temp['surface_m2'] / df_temp['rooms']
    low_surface_room = df_temp[df_temp['surface_per_room'] < 8].shape[0]

    if low_surface_room > 0:
        coherence_issues.append(f"{low_surface_room} biens avec surface/piece < 8 m2")

    results['coherence_issues'] = coherence_issues
    if len(coherence_issues) == 0:
        print("   [OK] Aucun probleme de coherence detecte")
    else:
        print("   [WARNING] Problemes de coherence detectes:")
        for issue in coherence_issues:
            print(f"      - {issue}")

    # 5. Distributions
    print("\n5. DISTRIBUTIONS")
    print(f"   Prix moyen: {df['price'].mean():,.2f} EUR")
    print(f"   Prix median: {df['price'].median():,.2f} EUR")
    print(f"   Surface moyenne: {df['surface_m2'].mean():.1f} m2")
    print(f"   Surface mediane: {df['surface_m2'].median():.1f} m2")

    return results


def clean_data(df, remove_duplicates=True, remove_outliers=True):
    """
    Nettoie les données.

    Args:
        df: DataFrame à nettoyer
        remove_duplicates: supprimer les doublons
        remove_outliers: supprimer les outliers

    Returns:
        DataFrame nettoyé
    """
    print("\n" + "="*80)
    print("NETTOYAGE DES DONNEES")
    print("="*80)

    df_clean = df.copy()
    initial_size = len(df_clean)

    # 1. Supprimer les doublons
    if remove_duplicates:
        duplicates_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_size - len(df_clean)
        print(f"\n1. Doublons supprimes: {duplicates_removed}")

    # 2. Supprimer les valeurs manquantes
    missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.dropna()
    missing_removed = initial_size - duplicates_removed - len(df_clean)
    print(f"2. Lignes avec valeurs manquantes supprimees: {missing_removed}")

    # 3. Supprimer les valeurs aberrantes
    if remove_outliers:
        # Prix aberrants (< 5000 ou > 5M)
        price_mask = (df_clean['price'] >= 5000) & (df_clean['price'] <= 5_000_000)

        # Surface aberrante (< 10 ou > 500)
        surface_mask = (df_clean['surface_m2'] >= 10) & (df_clean['surface_m2'] <= 500)

        # Âge aberrant (< 0 ou > 150)
        age_mask = (df_clean['age_years'] >= 0) & (df_clean['age_years'] <= 150)

        # Nombre de pièces aberrant (>= 1 et <= 10)
        rooms_mask = (df_clean['rooms'] >= 1) & (df_clean['rooms'] <= 10)

        # Combiner tous les masques
        all_masks = price_mask & surface_mask & age_mask & rooms_mask
        outliers_removed = len(df_clean) - all_masks.sum()

        df_clean = df_clean[all_masks]
        print(f"3. Outliers supprimes: {outliers_removed}")

    # 4. Vérifier la cohérence
    df_clean['price_per_m2'] = df_clean['price'] / df_clean['surface_m2']

    # Supprimer les prix au m² incohérents
    coherence_mask = (df_clean['price_per_m2'] >= 500) & (df_clean['price_per_m2'] <= 30000)
    coherence_removed = len(df_clean) - coherence_mask.sum()
    df_clean = df_clean[coherence_mask]
    print(f"4. Incoherences supprimees: {coherence_removed}")

    # Supprimer la colonne temporaire
    df_clean = df_clean.drop(columns=['price_per_m2'])

    # Résumé
    final_size = len(df_clean)
    total_removed = initial_size - final_size
    pct_removed = (total_removed / initial_size) * 100

    print("\n" + "="*80)
    print("RESUME DU NETTOYAGE")
    print("="*80)
    print(f"Taille initiale: {initial_size}")
    print(f"Taille finale: {final_size}")
    print(f"Lignes supprimees: {total_removed} ({pct_removed:.2f}%)")

    return df_clean


def save_cleaned_data(df, output_file='data/real_estate_data_clean.csv'):
    """
    Sauvegarde les données nettoyées.

    Args:
        df: DataFrame nettoyé
        output_file: fichier de sortie
    """
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Donnees nettoyees sauvegardees: {output_file}")


if __name__ == "__main__":
    # Charger et valider
    df = load_and_validate_data('data/real_estate_data.csv')

    # Contrôle qualité
    results = check_data_quality(df)

    # Nettoyer
    df_clean = clean_data(df, remove_duplicates=True, remove_outliers=True)

    # Vérifier à nouveau après nettoyage
    print("\n" + "="*80)
    print("VERIFICATION POST-NETTOYAGE")
    print("="*80)
    results_after = check_data_quality(df_clean)

    # Sauvegarder
    save_cleaned_data(df_clean, 'data/real_estate_data_clean.csv')

    # Diviser en train/test avec les données nettoyées
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

    train_df.to_csv('data/train_data_clean.csv', index=False)
    test_df.to_csv('data/test_data_clean.csv', index=False)

    print(f"[OK] Train set nettoye: {len(train_df)} echantillons -> data/train_data_clean.csv")
    print(f"[OK] Test set nettoye: {len(test_df)} echantillons -> data/test_data_clean.csv")

    print("\n" + "="*80)
    print("TERMINE!")
    print("="*80)
    print("Fichiers disponibles:")
    print("  - data/real_estate_data_clean.csv (dataset complet nettoye)")
    print("  - data/train_data_clean.csv (train set nettoye)")
    print("  - data/test_data_clean.csv (test set nettoye)")
