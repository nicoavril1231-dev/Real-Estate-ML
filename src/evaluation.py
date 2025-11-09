"""
Evaluation Module
Fonctions pour évaluer et visualiser les performances des modèles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               title: str = "Predictions vs Actual",
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: str = None):
    """
    Scatter plot des prédictions vs valeurs réelles.

    Args:
        y_true: valeurs réelles
        y_pred: prédictions
        title: titre du graphique
        figsize: taille de la figure
        save_path: chemin pour sauvegarder (optionnel)
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.5, s=30)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            'r--', lw=2, label='Perfect prediction')

    ax.set_xlabel('Actual Price (€)', fontsize=11)
    ax.set_ylabel('Predicted Price (€)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ajouter R²
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str = "Residuals Distribution",
                  figsize: Tuple[int, int] = (12, 5),
                  save_path: str = None):
    """
    Visualise la distribution des erreurs (résidus).

    Args:
        y_true: valeurs réelles
        y_pred: prédictions
        title: titre
        figsize: taille
        save_path: chemin de sauvegarde
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel('Residuals (€)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Residuals Distribution', fontsize=11, fontweight='bold')
    axes[0].legend()

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=11, fontweight='bold')

    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                           figsize: Tuple[int, int] = (14, 5),
                           save_path: str = None):
    """
    Visualise la distribution des erreurs absolues et relatives.

    Args:
        y_true: valeurs réelles
        y_pred: prédictions
        figsize: taille
        save_path: chemin de sauvegarde
    """
    absolute_errors = np.abs(y_true - y_pred)
    relative_errors = (absolute_errors / y_true) * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Erreurs absolues
    axes[0].hist(absolute_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(absolute_errors), color='red', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(absolute_errors):,.0f}€')
    axes[0].set_xlabel('Absolute Error (€)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Absolute Errors Distribution', fontsize=11, fontweight='bold')
    axes[0].legend()

    # Erreurs relatives
    axes[1].hist(relative_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(np.median(relative_errors), color='red', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(relative_errors):.1f}%')
    axes[1].axvline(10, color='green', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='10% threshold')
    axes[1].set_xlabel('Relative Error (%)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Relative Errors Distribution', fontsize=11, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_model_comparison(results_df: pd.DataFrame,
                         figsize: Tuple[int, int] = (16, 10),
                         save_path: str = None):
    """
    Visualise la comparaison de plusieurs modèles.

    Args:
        results_df: DataFrame avec colonnes ['RMSE', 'MAE', 'MAPE', 'R2']
        figsize: taille
        save_path: chemin de sauvegarde
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # R²
    results_df['R2'].sort_values(ascending=True).plot(
        kind='barh', ax=axes[0, 0], color='skyblue', edgecolor='black'
    )
    axes[0, 0].set_title('R² Score (higher = better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('R²')
    axes[0, 0].axvline(0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    axes[0, 0].legend()

    # RMSE
    results_df['RMSE'].sort_values(ascending=False).plot(
        kind='barh', ax=axes[0, 1], color='coral', edgecolor='black'
    )
    axes[0, 1].set_title('RMSE (lower = better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('RMSE (€)')

    # MAE
    results_df['MAE'].sort_values(ascending=False).plot(
        kind='barh', ax=axes[1, 0], color='lightgreen', edgecolor='black'
    )
    axes[1, 0].set_title('MAE (lower = better)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('MAE (€)')

    # MAPE
    results_df['MAPE'].sort_values(ascending=False).plot(
        kind='barh', ax=axes[1, 1], color='gold', edgecolor='black'
    )
    axes[1, 1].set_title('MAPE (lower = better)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('MAPE (%)')
    axes[1, 1].axvline(10, color='green', linestyle='--', alpha=0.5, label='Excellent (<10%)')
    axes[1, 1].legend()

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 12),
                           save_path: str = None):
    """
    Visualise l'importance des features.

    Args:
        importance_df: DataFrame avec colonnes ['feature', 'importance']
        top_n: nombre de features à afficher
        figsize: taille
        save_path: chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=figsize)

    top_features = importance_df.head(top_n)

    ax.barh(range(len(top_features)), top_features['importance'],
            edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def generate_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model") -> Dict[str, Any]:
    """
    Génère un rapport d'évaluation complet.

    Args:
        y_true: valeurs réelles
        y_pred: prédictions
        model_name: nom du modèle

    Returns:
        dictionnaire avec métriques et statistiques
    """
    errors = y_true - y_pred
    absolute_errors = np.abs(errors)
    relative_errors = (absolute_errors / y_true) * 100

    report = {
        'model_name': model_name,
        'metrics': {
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'MAPE': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            'R2': float(r2_score(y_true, y_pred)),
            'Max_Error': float(np.max(absolute_errors))
        },
        'error_statistics': {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'mean_absolute_error': float(np.mean(absolute_errors)),
            'median_absolute_error': float(np.median(absolute_errors))
        },
        'relative_errors': {
            'mean': float(np.mean(relative_errors)),
            'median': float(np.median(relative_errors)),
            'pct_within_10pct': float((relative_errors <= 10).sum() / len(relative_errors) * 100),
            'pct_within_20pct': float((relative_errors <= 20).sum() / len(relative_errors) * 100)
        },
        'sample_size': len(y_true)
    }

    return report


def print_evaluation_report(report: Dict[str, Any]):
    """
    Affiche un rapport d'évaluation de manière formatée.

    Args:
        report: dictionnaire de rapport généré par generate_evaluation_report
    """
    print("=" * 80)
    print(f"EVALUATION REPORT: {report['model_name']}")
    print("=" * 80)

    print("\n📊 METRICS:")
    for metric, value in report['metrics'].items():
        if metric == 'R2':
            print(f"  {metric:20s}: {value:.4f}")
        else:
            print(f"  {metric:20s}: {value:,.2f}")

    print("\n📈 ERROR STATISTICS:")
    for stat, value in report['error_statistics'].items():
        print(f"  {stat:30s}: {value:,.2f}")

    print("\n📉 RELATIVE ERRORS:")
    for stat, value in report['relative_errors'].items():
        if 'pct' in stat:
            print(f"  {stat:30s}: {value:.1f}%")
        else:
            print(f"  {stat:30s}: {value:.2f}%")

    print(f"\n📦 Sample size: {report['sample_size']}")
    print("=" * 80)


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                    filepath: str, additional_data: pd.DataFrame = None):
    """
    Sauvegarde les prédictions dans un CSV.

    Args:
        y_true: valeurs réelles
        y_pred: prédictions
        filepath: chemin de sauvegarde
        additional_data: données supplémentaires à inclure
    """
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': y_true - y_pred,
        'absolute_error': np.abs(y_true - y_pred),
        'relative_error_pct': (np.abs(y_true - y_pred) / y_true) * 100
    })

    if additional_data is not None:
        df = pd.concat([df, additional_data.reset_index(drop=True)], axis=1)

    df.to_csv(filepath, index=False)
    print(f"✅ Predictions saved: {filepath}")
