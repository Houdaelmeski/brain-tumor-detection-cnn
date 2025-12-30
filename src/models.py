"""
Architecture du modele CNN
"""
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models


def create_cnn_model(input_shape=(128, 128, 1), num_classes=4):
    """
    Cree le modele CNN pour la classification de tumeurs cerebrales
    
    Architecture:
    - 4 blocs convolutifs (32, 64, 128, 256 filtres)
    - Batch Normalization et Dropout
    - Global Average Pooling
    - 2 couches Dense (512, 256)
    - Sortie: 4 classes (softmax)
    
    Args:
        input_shape: Taille de l'image d'entree (128, 128, 1)
        num_classes: Nombre de classes de sortie (4)
    
    Returns:
        model: Modele Keras non compile
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloc 1 - Extraction de caracteristiques basiques
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Bloc 2 - Extraction de caracteristiques intermediaires
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),
        
        # Bloc 3 - Extraction de caracteristiques avancees
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        
        # Bloc 4 - Extraction de caracteristiques complexes
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        
        # Classification
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile le modele avec l'optimiseur et les metriques
    
    Args:
        model: Modele Keras
        learning_rate: Taux d'apprentissage (default: 0.001)
    
    Returns:
        model: Modele compile
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def get_callbacks(model_path='models/best_model.keras'):
    """
    Cree les callbacks pour l'entrainement
    
    Args:
        model_path: Chemin pour sauvegarder le meilleur modele
    
    Returns:
        list: Liste de callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

if __name__ == "__main__":
    # Test du module
    print("Test de creation du modele...")
    model = create_cnn_model()
    model = compile_model(model)
    model.summary()
    print(f"\nNombre de parametres: {model.count_params():,}")