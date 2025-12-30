"""
Script de prediction sur une nouvelle image
"""
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (128, 128)

# Descriptions des classes
CLASS_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'Tumeur maligne du cerveau originaire des cellules gliales',
        'performance': '90.6% precision | 93.7% recall'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'Tumeur generalement benigne des meninges',
        'performance': '90.5% precision | 58.8% recall'
    },
    'notumor': {
        'name': 'No Tumor',
        'description': 'Aucune tumeur detectee',
        'performance': '87.6% precision | 97.3% recall'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'Tumeur de la glande pituitaire',
        'performance': '84.7% precision | 99.3% recall'
    }
}

def predict_image(model_path, image_path, show_plot=True):
    """
    Predit la classe d'une image IRM
    
    Args:
        model_path: Chemin vers le modele
        image_path: Chemin vers l'image
        show_plot: Afficher le graphique
    
    Returns:
        predicted_class, confidence, probabilities
    """
    print("="*70)
    print("PREDICTION SUR IMAGE IRM")
    print("="*70)
    
    # Charger le modele
    print(f"\n1. Chargement du modele: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Charger et pretraiter l'image
    print(f"2. Chargement de l'image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    img_original = img.copy()
    img_resized = cv2.resize(img, IMG_SIZE)
    img_preprocessed = img_resized / 255.0
    img_preprocessed = np.expand_dims(img_preprocessed, axis=-1)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    
    # Prediction
    print("3. Prediction en cours...")
    predictions = model.predict(img_preprocessed, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASSES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Affichage des resultats
    print(f"\n{'='*70}")
    print("RESULTATS")
    print(f"{'='*70}")
    print(f"\nClasse predite: {CLASS_INFO[predicted_class]['name'].upper()}")
    print(f"Confiance: {confidence:.2f}%")
    
    # Niveau de confiance
    if confidence > 90:
        print("Niveau de confiance: TRES ELEVE")
    elif confidence > 70:
        print("Niveau de confiance: ELEVE")
    elif confidence > 50:
        print("Niveau de confiance: MOYEN")
    else:
        print("Niveau de confiance: FAIBLE")
    
    print(f"\nDescription: {CLASS_INFO[predicted_class]['description']}")
    print(f"Performance du modele: {CLASS_INFO[predicted_class]['performance']}")
    
    print(f"\nProbabilites detaillees:")
    for i, class_name in enumerate(CLASSES):
        prob = predictions[0][i] * 100
        bar = "=" * int(prob / 2)
        print(f"   {CLASS_INFO[class_name]['name']:15s}: {prob:6.2f}% [{bar}]")
    
    # Visualisation
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Image
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title('Image IRM', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # Barres de probabilite
        colors = ['#FF6B6B' if i == predicted_class_idx else '#E8E8E8' 
                  for i in range(4)]
        bars = axes[1].bar([CLASS_INFO[c]['name'] for c in CLASSES], 
                          predictions[0] * 100, color=colors)
        axes[1].set_ylabel('Probabilite (%)', fontsize=12)
        axes[1].set_title(f'Predictions - {CLASS_INFO[predicted_class]["name"]} ({confidence:.1f}%)', 
                         fontsize=14, weight='bold')
        axes[1].set_ylim([0, 100])
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        os.makedirs('reports/figures', exist_ok=True)
        output_path = 'reports/figures/prediction_result.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualisation sauvegardee: {output_path}")
        plt.show()
    
    print(f"\n{'='*70}")
    
    return predicted_class, confidence, predictions[0]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <image_path>")
        print("\nExemple:")
        print("  python predict.py models/brain_tumor_classifier_v1.keras Testing/glioma/image.jpg")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"Erreur: Modele introuvable: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Erreur: Image introuvable: {image_path}")
        sys.exit(1)
    
    predict_image(model_path, image_path)