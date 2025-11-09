"""
Script pour générer un dataset complet pour le pipeline ML immobilier.
Ce script crée des features réalistes à partir des prix fournis.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def generate_complete_dataset(prices_file='data/Actual Y train.csv', output_file='data/real_estate_data.csv'):
    """
    Génère un dataset complet avec des features réalistes basées sur les prix fournis.

    Args:
        prices_file: chemin vers le fichier de prix
        output_file: chemin vers le fichier de sortie
    """

    # Charger les prix
    print(f"Chargement des prix depuis {prices_file}...")
    df_prices = pd.read_csv(prices_file)

    # Extraire la colonne de prix
    if 'Actual price' in df_prices.columns:
        prices = df_prices['Actual price'].values
    else:
        # Si le nom de colonne est différent, prendre la première colonne
        prices = df_prices.iloc[:, 0].values

    n_samples = len(prices)
    print(f"Nombre d'échantillons: {n_samples}")

    # Initialiser un random seed pour la reproductibilité
    np.random.seed(42)

    # Villes françaises avec des prix moyens au m²
    cities = ['Paris', 'Lyon', 'Marseille', 'Bordeaux', 'Lille', 'Nice', 'Toulouse', 'Nantes']
    city_prices_m2 = [10000, 4500, 3800, 4200, 3500, 5000, 4000, 4300]

    # Générer les features en fonction des prix
    data = []

    for price in prices:
        # Estimation de la ville basée sur le prix
        # Plus le prix est élevé, plus la probabilité d'être à Paris est grande
        price_factor = price / np.percentile(prices, 75)
        city_probs = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.05])

        # Ajuster les probabilités en fonction du prix
        if price_factor > 2:  # Prix très élevé
            city_probs = np.array([0.5, 0.1, 0.05, 0.1, 0.05, 0.15, 0.03, 0.02])
        elif price_factor < 0.5:  # Prix très bas
            city_probs = np.array([0.1, 0.15, 0.2, 0.15, 0.2, 0.05, 0.1, 0.05])

        city_probs = city_probs / city_probs.sum()
        city_idx = np.random.choice(len(cities), p=city_probs)
        city = cities[city_idx]
        city_price_m2 = city_prices_m2[city_idx]

        # Calculer la surface approximative basée sur le prix et la ville
        # prix = price_m2 * surface * facteurs (age, amenities, energy_class ≈ 0.8-1.2)
        # surface ≈ prix / (price_m2 * facteurs)

        # Surface approximative avec facteurs réalistes
        quality_factor = np.random.uniform(0.75, 1.3)  # Variation due à l'état, amenités, etc.
        base_surface = price / (city_price_m2 * quality_factor)

        # Ajouter du bruit pour rendre plus réaliste
        surface_noise = np.random.normal(1, 0.1)
        surface = base_surface * surface_noise

        # Limiter à des valeurs réalistes
        surface = max(18, min(300, surface))

        # Nombre de pièces basé sur la surface
        if surface < 30:
            rooms = np.random.choice([1, 2], p=[0.7, 0.3])
        elif surface < 50:
            rooms = np.random.choice([2, 3], p=[0.6, 0.4])
        elif surface < 80:
            rooms = np.random.choice([3, 4], p=[0.6, 0.4])
        elif surface < 120:
            rooms = np.random.choice([4, 5], p=[0.5, 0.5])
        else:
            rooms = np.random.choice([5, 6, 7], p=[0.5, 0.3, 0.2])

        # Âge du bien (impacte le prix)
        # Les biens plus chers ont tendance à être plus récents ou rénovés
        if price > np.percentile(prices, 75):
            age = np.random.exponential(scale=10)  # Plutôt récent
        else:
            age = np.random.exponential(scale=25)  # Plus ancien
        age = min(100, age)

        # Étage
        floor = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                 p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

        # Ascenseur (plus probable dans les étages élevés et les biens chers)
        has_elevator_prob = 0.3 if floor <= 2 else 0.8
        if price > np.percentile(prices, 60):
            has_elevator_prob += 0.2
        has_elevator = 1 if np.random.random() < has_elevator_prob else 0

        # Parking (plus probable dans les biens chers)
        has_parking_prob = 0.3 if price < np.percentile(prices, 50) else 0.6
        has_parking = 1 if np.random.random() < has_parking_prob else 0

        # Balcon
        has_balcony = 1 if np.random.random() < 0.45 else 0

        # Classe énergétique (impact sur le prix)
        # Les biens chers ont tendance à avoir de meilleures classes énergétiques
        if price > np.percentile(prices, 75) and age < 20:
            energy_class = np.random.choice(['A', 'B', 'C', 'D'], p=[0.2, 0.4, 0.3, 0.1])
        elif price > np.percentile(prices, 50):
            energy_class = np.random.choice(['B', 'C', 'D', 'E'], p=[0.15, 0.35, 0.35, 0.15])
        else:
            energy_class = np.random.choice(['C', 'D', 'E', 'F', 'G'], p=[0.1, 0.3, 0.3, 0.2, 0.1])

        # Ajouter le record
        data.append({
            'city': city,
            'surface_m2': round(surface, 1),
            'rooms': int(rooms),
            'age_years': round(age, 1),
            'floor': int(floor),
            'has_elevator': int(has_elevator),
            'has_parking': int(has_parking),
            'has_balcony': int(has_balcony),
            'energy_class': energy_class,
            'price': round(price, 2)
        })

    # Créer le DataFrame
    df_complete = pd.DataFrame(data)

    # Statistiques
    print("\n" + "="*80)
    print("STATISTIQUES DU DATASET GÉNÉRÉ")
    print("="*80)
    print(f"Nombre total d'échantillons: {len(df_complete)}")
    print(f"\nRépartition par ville:")
    print(df_complete['city'].value_counts())
    print(f"\nStatistiques des prix:")
    print(df_complete['price'].describe())
    print(f"\nStatistiques de surface:")
    print(df_complete['surface_m2'].describe())
    print(f"\nRépartition classe énergétique:")
    print(df_complete['energy_class'].value_counts().sort_index())

    # Vérifications de qualité
    print("\n" + "="*80)
    print("VÉRIFICATIONS DE QUALITÉ")
    print("="*80)
    print(f"Valeurs manquantes: {df_complete.isnull().sum().sum()}")
    print(f"Doublons: {df_complete.duplicated().sum()}")
    print(f"Prix minimum: {df_complete['price'].min():,.2f} €")
    print(f"Prix maximum: {df_complete['price'].max():,.2f} €")
    print(f"Surface minimum: {df_complete['surface_m2'].min()} m²")
    print(f"Surface maximum: {df_complete['surface_m2'].max()} m²")

    # Sauvegarder
    df_complete.to_csv(output_file, index=False)
    print(f"\n[OK] Dataset sauvegarde: {output_file}")

    # Afficher un aperçu
    print("\n" + "="*80)
    print("APERÇU DES DONNÉES")
    print("="*80)
    print(df_complete.head(10))

    return df_complete


def split_train_test(input_file='data/real_estate_data.csv',
                     train_output='data/train_data.csv',
                     test_output='data/test_data.csv',
                     test_size=0.2,
                     random_state=42):
    """
    Divise le dataset en train et test.

    Args:
        input_file: fichier d'entrée
        train_output: fichier de sortie pour train
        test_output: fichier de sortie pour test
        test_size: proportion du test set
        random_state: seed pour la reproductibilité
    """
    from sklearn.model_selection import train_test_split

    print(f"\nDivision train/test (test_size={test_size})...")
    df = pd.read_csv(input_file)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"[OK] Train set: {len(train_df)} echantillons -> {train_output}")
    print(f"[OK] Test set: {len(test_df)} echantillons -> {test_output}")


if __name__ == "__main__":
    import os

    # Créer le dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Générer le dataset complet
    df = generate_complete_dataset(
        prices_file='data/Actual Y train.csv',
        output_file='data/real_estate_data.csv'
    )

    # Diviser en train/test
    split_train_test(
        input_file='data/real_estate_data.csv',
        train_output='data/train_data.csv',
        test_output='data/test_data.csv',
        test_size=0.2,
        random_state=42
    )

    print("\n" + "="*80)
    print("TERMINÉ!")
    print("="*80)
    print("Vous pouvez maintenant utiliser les fichiers:")
    print("  - data/real_estate_data.csv (dataset complet)")
    print("  - data/train_data.csv (ensemble d'entraînement)")
    print("  - data/test_data.csv (ensemble de test)")
