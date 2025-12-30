// ===============================
// Variables globales
// ===============================
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const resultSection = document.getElementById('resultSection');
const uploadedImage = document.getElementById('uploadedImage');
const tumorType = document.getElementById('tumorType');
const confidence = document.getElementById('confidence');
const description = document.getElementById('description');
const performance = document.getElementById('performance');
const probabilityBars = document.getElementById('probabilityBars');

// ===============================
// Couleurs des classes
// ===============================
const classColors = {
    glioma: '#FF6B6B',
    meningioma: '#4ECDC4',
    notumor: '#95E1D3',
    pituitary: '#F38181'
};

// ===============================
// Sélection du fichier
// ===============================
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        fileName.textContent = fileInput.files[0].name;
        uploadBtn.disabled = false;
    } else {
        fileName.textContent = 'No file chosen';
        uploadBtn.disabled = true;
    }
});

// ===============================
// Upload & Predict
// ===============================
uploadBtn.addEventListener('click', async () => {
    if (fileInput.files.length === 0) {
        alert('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    uploadBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error(error);
        alert('Prediction error. Please try again.');
    } finally {
        uploadBtn.disabled = false;
    }
});

// ===============================
// Affichage des résultats
// ===============================
function displayResults(data) {
    // Image
    uploadedImage.src = 'data:image/jpeg;base64,' + data.image;

    // Type de tumeur
    tumorType.textContent = data.class_info.name;
    tumorType.style.color = classColors[data.prediction];

    // Confiance
    confidence.textContent = data.confidence.toFixed(2) + '%';

    // Description
    description.innerHTML = `
        <strong>${data.class_info.name}</strong><br>
        ${data.class_info.description}<br>
        <em>Confidence Level: ${data.confidence > 90 ? 'Very High Confidence' : 'High Confidence'}</em>
    `;

    // Performance du modèle (PAR CLASSE UNIQUEMENT)
    performance.innerHTML = `
        Model Performance for ${data.class_info.name}:<br>
        ${data.class_info.performance}
    `;

    // Probabilités
    displayProbabilityBars(data.probabilities);

    // Afficher la section résultats
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// ===============================
// Probability Distribution (AFFICHAGE EXACT)
// ===============================
function displayProbabilityBars(probabilities) {
    probabilityBars.innerHTML = '';

    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([className, prob]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'prob-bar-container';

        const label = document.createElement('div');
        label.className = 'prob-label';
        label.textContent = className.charAt(0).toUpperCase() + className.slice(1);

        const barWrapper = document.createElement('div');
        barWrapper.className = 'prob-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.style.backgroundColor = classColors[className];
        bar.style.width = prob + '%';
        bar.textContent = prob.toFixed(1) + '%';

        barWrapper.appendChild(bar);
        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper);
        probabilityBars.appendChild(barContainer);
    });
}
