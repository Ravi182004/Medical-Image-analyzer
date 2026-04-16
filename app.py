import os
import numpy as np
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
HISTORY_FILE = 'history.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# CLASS LABELS
# =========================
CLASS_MAPS = {
    'mri': ['Normal', 'Brain Tumor'],
    'ct': ['Benign', 'Malignant', 'Normal'],
    'xray': ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
}

# =========================
# LOAD MODEL
# =========================
def load_model_for(scan_type):
    try:
        if scan_type == 'mri':
            return load_model('models/brain_tumor_final.keras')
        elif scan_type == 'ct':
            return load_model('models/ctscan_mobilenet.keras')
        elif scan_type == 'xray':
            return load_model('models/chest_xray_best.keras')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

# =========================
# HISTORY SAVE
# =========================
def save_to_history(data):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        data['id'] = int(datetime.now().timestamp() * 1000)
        history.append(data)

        if len(history) > 50:
            history = history[-50:]

        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        print(f"History error: {e}")

def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

# =========================
# HOME
# =========================
@app.route('/')
def index():
    return render_template('index.html')

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    scan_type = request.form['scan_type']

    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    model = load_model_for(scan_type)

    prediction = "Unknown"
    confidence = None

    if model:
        try:
            x = preprocess_image(path)
            preds = model.predict(x)

            if scan_type == 'mri' and preds.shape[-1] == 1:
                prob = float(preds[0][0])
                idx = 1 if prob >= 0.5 else 0
                prediction = CLASS_MAPS['mri'][idx]
                confidence = prob if idx == 1 else 1 - prob

            else:
                idx = int(np.argmax(preds[0]))
                prediction = CLASS_MAPS[scan_type][idx]
                confidence = float(np.max(preds[0]))

            prediction = prediction.title()

        except Exception as e:
            print(e)
            prediction = "Error"

    # Save history
    save_to_history({
        "image_filename": "uploads/" + filename,
        "scan_type": scan_type,
        "label": prediction,
        "confidence": round(confidence * 100, 2) if confidence else "N/A",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2) if confidence else None,
        filename=filename
    )


@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/chatbot')
def chatbot():
    qa = {
        "What are symptoms of COVID-19?": "Fever, dry cough, and breathing difficulty.",
        "What is diabetes?": "A condition affecting blood sugar levels.",
        "What is blood pressure?": "Force of blood against artery walls.",
        "What is fever?": "Temporary rise in body temperature.",
        "What causes headache?": "Stress, dehydration, or illness.",
        "What is cold?": "Viral infection causing sneezing and runny nose.",
        "What are allergies?": "Immune response to foreign substances.",
        "Why is exercise important?": "Improves overall health and fitness.",
        "What is a balanced diet?": "Includes all nutrients in proper proportion.",
        "What is malaria?": "Disease caused by mosquito bites."
    }

    return render_template("chatbot.html", qa=qa)

@app.route('/history')
def history():
    history_data = get_history()
    return render_template('history.html', history=history_data)

# =========================
# DOCTOR
# =========================
DOCTORS_DB = {
    "Brain Tumor": [
        {"name": "Dr. Arjun Sharma", "specialist": "Neurosurgeon", "location": "Delhi", "hospital": "AIIMS Delhi"},
        {"name": "Dr. Priya Nair", "specialist": "Neuro-Oncologist", "location": "Mumbai", "hospital": "Tata Memorial Hospital"},
        {"name": "Dr. Ramesh Iyer", "specialist": "Neurosurgeon", "location": "Bangalore", "hospital": "Manipal Hospital"},
    ],
    "Malignant": [
        {"name": "Dr. Sunita Verma", "specialist": "Oncologist", "location": "Chennai", "hospital": "Apollo Hospital"},
        {"name": "Dr. Anil Kapoor", "specialist": "Thoracic Surgeon", "location": "Hyderabad", "hospital": "Yashoda Hospital"},
        {"name": "Dr. Meena Pillai", "specialist": "Pulmonologist", "location": "Pune", "hospital": "Ruby Hall Clinic"},
    ],
    "Benign": [
        {"name": "Dr. Rajiv Gupta", "specialist": "Pulmonologist", "location": "Delhi", "hospital": "Fortis Hospital"},
        {"name": "Dr. Kavitha Rao", "specialist": "Radiologist", "location": "Bangalore", "hospital": "Narayana Health"},
        {"name": "Dr. Suresh Menon", "specialist": "General Surgeon", "location": "Kochi", "hospital": "Amrita Hospital"},
    ],
    "Covid-19": [
        {"name": "Dr. Deepak Singh", "specialist": "Pulmonologist", "location": "Delhi", "hospital": "Max Hospital"},
        {"name": "Dr. Anita Desai", "specialist": "Infectious Disease Specialist", "location": "Mumbai", "hospital": "Hinduja Hospital"},
        {"name": "Dr. Vikram Bose", "specialist": "Critical Care Specialist", "location": "Kolkata", "hospital": "AMRI Hospital"},
    ],
    "Pneumonia": [
        {"name": "Dr. Neha Joshi", "specialist": "Pulmonologist", "location": "Pune", "hospital": "KEM Hospital"},
        {"name": "Dr. Sameer Khan", "specialist": "Chest Physician", "location": "Lucknow", "hospital": "SGPGI"},
        {"name": "Dr. Lata Krishnan", "specialist": "Respiratory Medicine", "location": "Chennai", "hospital": "Sri Ramachandra Hospital"},
    ],
    "Tuberculosis": [
        {"name": "Dr. Mohan Das", "specialist": "Pulmonologist", "location": "Delhi", "hospital": "LRS Institute"},
        {"name": "Dr. Farida Sheikh", "specialist": "Chest Specialist", "location": "Mumbai", "hospital": "Sewri TB Hospital"},
        {"name": "Dr. Ravi Teja", "specialist": "Infectious Disease", "location": "Hyderabad", "hospital": "Gandhi Hospital"},
    ],
}

@app.route("/doctor")
def doctor():
    disease = request.args.get("disease")
    doctors = DOCTORS_DB.get(disease, [
        {"name": "Dr. General Physician", "specialist": "General Medicine", "location": "Your City", "hospital": "Nearest Hospital"},
    ])
    return render_template("doctor.html", disease=disease, doctors=doctors)

# =========================
# DIET
# =========================
DIET_DB = {
    "Brain Tumor": {
        "veg":    "Spinach, Broccoli, Walnuts, Flaxseeds, Turmeric milk, Blueberries, Green tea, Whole grains, Lentils.",
        "nonveg": "Salmon, Sardines, Eggs, Chicken (grilled), Fish oil, Leafy greens, Walnuts, Berries.",
        "vegan":  "Flaxseeds, Chia seeds, Walnuts, Tofu, Soy milk, Blueberries, Broccoli, Turmeric, Green tea.",
    },
    "Malignant": {
        "veg":    "Cruciferous vegetables, Garlic, Onion, Tomatoes, Green tea, Berries, Turmeric, Whole grains.",
        "nonveg": "Grilled fish, Skinless chicken, Eggs, Leafy greens, Berries, Garlic, Olive oil.",
        "vegan":  "Legumes, Lentils, Tofu, Berries, Broccoli, Cauliflower, Flaxseeds, Nuts, Green tea.",
    },
    "Benign": {
        "veg":    "Balanced diet with fruits, vegetables, whole grains, low-fat dairy, nuts, and seeds.",
        "nonveg": "Lean meats, Fish, Eggs, Fruits, Vegetables, Whole grains, Low-fat dairy.",
        "vegan":  "Tofu, Legumes, Whole grains, Nuts, Seeds, Fresh fruits and vegetables.",
    },
    "Covid-19": {
        "veg":    "Citrus fruits, Ginger tea, Turmeric milk, Garlic, Spinach, Bell peppers, Yogurt, Almonds.",
        "nonveg": "Chicken soup, Eggs, Fish, Citrus fruits, Ginger, Garlic, Honey, Warm broths.",
        "vegan":  "Lemon water, Ginger tea, Turmeric, Garlic, Berries, Legumes, Soy milk, Pumpkin seeds.",
    },
    "Pneumonia": {
        "veg":    "Warm soups, Herbal teas, Honey, Garlic, Ginger, Citrus fruits, Yogurt, Leafy greens.",
        "nonveg": "Chicken broth, Eggs, Fish, Warm soups, Honey, Garlic, Ginger, Orange juice.",
        "vegan":  "Vegetable broth, Ginger tea, Lemon water, Garlic, Berries, Tofu, Pumpkin seeds.",
    },
    "Tuberculosis": {
        "veg":    "High-calorie foods: Milk, Paneer, Nuts, Banana, Rice, Dal, Potatoes, Eggs (if ovo-veg), Cheese.",
        "nonveg": "Eggs, Chicken, Fish, Milk, Red meat (moderate), Nuts, Bananas, Rice, Pulses.",
        "vegan":  "Soy milk, Tofu, Lentils, Chickpeas, Nuts, Avocado, Brown rice, Fortified cereals.",
    },
}

@app.route("/diet", methods=["GET", "POST"])
def diet():
    disease = request.args.get("disease")
    diet_result = None
    if request.method == "POST":
        preference = request.form.get("preference")
        disease_diets = DIET_DB.get(disease, {})
        diet_result = disease_diets.get(preference, "Please maintain a balanced and nutritious diet. Consult your doctor for personalised guidance.")
    return render_template("diet.html", disease=disease, diet=diet_result)

# =========================
# ABOUT
# =========================
@app.route("/about")
def about():
    return render_template("about.html")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)