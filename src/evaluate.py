"""
Script d'evaluation du modele
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from data import get_data_generators

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def evaluate_model(model_path, data_path='.', save_figures=True):
    """
    Evalue le modele sur le test set
    
    Args:
        model_path: Chemin vers le modele
        data_path: Chemin vers les donnees
        save_figures: Sauvegarder les figures
    """
    print("="*70)
    print("EVALUATION DU MODELE")
    print("="*70)
    
    # Charger le modele
    print(f"\n1. Chargement du modele: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Charger les donnees de test
    print("\n2. Chargement des donnees de test...")
    _, _, test_gen = get_data_generators(data_path)
    print(f"   Test: {test_gen.samples} images")
    
    # Evaluation
    print("\n3. Evaluation...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=1)
    
    print(f"\n{'='*70}")
    print("METRIQUES GLOBALES:")
    print(f"{'='*70}")
    print(f"   Accuracy:  {test_acc*100:.2f}%")
    print(f"   Precision: {test_prec*100:.2f}%")
    print(f"   Recall:    {test_rec*100:.2f}%")
    print(f"   Loss:      {test_loss:.4f}")
    
    # Predictions
    print("\n4. Generation des predictions...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    # Rapport de classification
    print(f"\n{'='*70}")
    print("RAPPORT PAR CLASSE:")
    print(f"{'='*70}")
    print(classification_report(true_classes, predicted_classes, 
                                target_names=CLASSES, digits=3))
    
    # Matrice de confusion
    print("\n5. Generation de la matrice de confusion...")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    if save_figures:
        os.makedirs('reports/figures', exist_ok=True)
        
        # Matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    cbar_kws={'label': 'Nombre de predictions'})
        plt.title('Matrice de Confusion', fontsize=16, weight='bold')
        plt.ylabel('Vraie Classe', fontsize=12)
        plt.xlabel('Classe Predite', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("   Sauvegarde: reports/figures/confusion_matrix.png")
        plt.close()
        
        # Graphique des metriques par classe
        report_dict = classification_report(true_classes, predicted_classes,
                                           target_names=CLASSES, output_dict=True)
        
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(CLASSES))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [report_dict[cls][metric] for cls in CLASSES]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metriques par Classe', fontsize=16, weight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(CLASSES)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/metrics_by_class.png', dpi=300, bbox_inches='tight')
        print("   Sauvegarde: reports/figures/metrics_by_class.png")
        plt.close()
    
    print(f"\n{'='*70}")
    print("EVALUATION TERMINEE")
    print(f"{'='*70}")
    
    return {
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'loss': test_loss,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_path>")
        print("Exemple: python evaluate.py models/brain_tumor_classifier_v1.keras")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Erreur: Modele introuvable: {model_path}")
        sys.exit(1)
    
    evaluate_model(model_path)