"""
Application Web Flask pour la detection de tumeurs cerebrales
"""
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Creer le dossier uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modele
MODEL_PATH = '../models/brain_tumor_classifier_v1.keras'
print(f"Chargement du modele: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modele charge avec succes!")

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (128, 128)

# Informations sur les classes
CLASS_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'Tumeur maligne du cerveau originaire des cellules gliales',
        'performance': '90.6% precision | 93.7% recall',
        'color': '#FF6B6B'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'Tumeur generalement benigne des meninges',
        'performance': '90.5% precision | 58.8% recall',
        'color': '#4ECDC4'
    },
    'notumor': {
        'name': 'No Tumor',
        'description': 'Aucune tumeur detectee',
        'performance': '87.6% precision | 97.3% recall',
        'color': '#95E1D3'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'Tumeur de la glande pituitaire',
        'performance': '84.7% precision | 99.3% recall',
        'color': '#F38181'
    }
}

def preprocess_image(image_path):
    """Pretraiter l'image pour le modele"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def image_to_base64(image_path):
    """Convertir l'image en base64"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Pretraiter et predire
        img = preprocess_image(filepath)
        
        if img is None:
            os.remove(filepath)
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Prediction
        predictions = model.predict(img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Probabilites
        probabilities = {
            CLASSES[i]: float(predictions[0][i] * 100)
            for i in range(len(CLASSES))
        }
        
        # Image en base64
        image_base64 = image_to_base64(filepath)
        
        # Nettoyer
        os.remove(filepath)
        
        # Resultat
        result = {
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'probabilities': probabilities,
            'class_info': CLASS_INFO[predicted_class],
            'image': image_base64
        }
        
        return jsonify(result)

@app.route('/model_info')
def model_info():
    """Informations sur le modele"""
    info = {
        'accuracy': 87.95,
        'precision': 89.14,
        'recall': 87.03,
        'classes': CLASSES,
        'class_details': CLASS_INFO,
        'architecture': {
            'type': 'CNN 4 blocs',
            'parameters': 850532,
            'input_size': '128x128 grayscale'
        }
    }
    return jsonify(info)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("APPLICATION WEB - DETECTION DE TUMEURS CEREBRALES")
    print("="*70)
    print("\nModele charge: brain_tumor_classifier_v1.keras")
    print("Accuracy: 87.95%")
    print("\nOuvrez votre navigateur sur: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)