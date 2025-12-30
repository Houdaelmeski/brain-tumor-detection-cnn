"""
Script d'entrainement du modele
"""
import os
import sys
import argparse
import pickle
from datetime import datetime

import tensorflow as tf
from data import get_data_generators, get_class_weights
from models import create_cnn_model, compile_model, get_callbacks

def train_model(data_path='.', epochs=25, batch_size=32, learning_rate=0.001):
    """
    Entraine le modele CNN
    
    Args:
        data_path: Chemin vers les donnees
        epochs: Nombre d'epochs
        batch_size: Taille des batchs
        learning_rate: Taux d'apprentissage
    """
    print("="*70)
    print("ENTRAINEMENT DU MODELE CNN - DETECTION TUMEURS CEREBRALES")
    print("="*70)
    
    # Charger les donnees
    print("\n1. Chargement des donnees...")
    train_gen, val_gen, test_gen = get_data_generators(data_path, batch_size)
    class_weights = get_class_weights(train_gen)
    
    print(f"   Train: {train_gen.samples} images")
    print(f"   Validation: {val_gen.samples} images")
    print(f"   Test: {test_gen.samples} images")
    print(f"   Classes: {list(train_gen.class_indices.keys())}")
    
    # Creer le modele
    print("\n2. Creation du modele...")
    model = create_cnn_model()
    model = compile_model(model, learning_rate)
    
    print(f"\nArchitecture du modele:")
    model.summary()
    print(f"\nParametres: {model.count_params():,}")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/model_{timestamp}.keras'
    os.makedirs('models', exist_ok=True)
    callbacks = get_callbacks(model_path)
    
    # Entrainement
    print(f"\n3. Entrainement ({epochs} epochs)...")
    print(f"   Model sera sauvegarde dans: {model_path}")
    print(f"   Temps estime: 30-40 minutes sur CPU\n")
    
    import time
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    
    # Sauvegarder l'historique
    history_path = f'models/history_{timestamp}.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"\n4. Entrainement termine en {elapsed_time/60:.1f} minutes!")
    print(f"   Modele: {model_path}")
    print(f"   Historique: {history_path}")
    
    # Evaluation finale
    print("\n5. Evaluation sur test set...")
    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
    
    print(f"\nResultats finaux:")
    print(f"   Accuracy:  {test_acc*100:.2f}%")
    print(f"   Precision: {test_prec*100:.2f}%")
    print(f"   Recall:    {test_rec*100:.2f}%")
    print(f"   Loss:      {test_loss:.4f}")
    
    print("\n" + "="*70)
    print("ENTRAINEMENT TERMINE AVEC SUCCES!")
    print("="*70)
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrainer le modele CNN')
    parser.add_argument('--data-path', type=str, default='.',
                        help='Chemin vers les donnees (defaut: racine)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Nombre d epochs (defaut: 25)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Taille des batchs (defaut: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Taux d apprentissage (defaut: 0.001)')
    
    args = parser.parse_args()
    
    train_model(args.data_path, args.epochs, args.batch_size, args.learning_rate)