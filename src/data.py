"""
Module de chargement et preparation des donnees
"""
import os
import numpy as np
from tensorflow import keras

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_data_generators(data_path='.', batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Cree les generateurs de donnees pour train/val/test
    
    Args:
        data_path: Chemin vers le dossier racine
        batch_size: Taille des batchs
        img_size: Taille des images (largeur, hauteur)
    
    Returns:
        train_generator, validation_generator, test_generator
    """
    train_path = os.path.join(data_path, 'Training')
    test_path = os.path.join(data_path, 'Testing')
    
    # Augmentation pour l'entrainement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
        fill_mode='nearest'
    )
    
    # Pas d'augmentation pour le test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generateur d'entrainement
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode='grayscale'
    )
    
    # Generateur de validation
    validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='grayscale'
    )
    
    # Generateur de test
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale'
    )
    
    return train_generator, validation_generator, test_generator

def get_class_weights(train_generator):
    """
    Calcule les poids des classes pour equilibrer l'entrainement
    
    Args:
        train_generator: Generateur d'entrainement
    
    Returns:
        dict: Poids des classes
    """
    from sklearn.utils import class_weight
    
    train_labels = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    return dict(enumerate(class_weights))

if __name__ == "__main__":
    # Test du module
    print("Test du chargement des donnees...")
    train_gen, val_gen, test_gen = get_data_generators()
    print(f"Train: {train_gen.samples} images")
    print(f"Validation: {val_gen.samples} images")
    print(f"Test: {test_gen.samples} images")
    print(f"Classes: {train_gen.class_indices}")