from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import random  
from werkzeug.utils import secure_filename

#Clear Memory
tf.keras.backend.clear_session()

app = Flask(__name__)
CORS(app)

# Settings
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'brain_tumor_vgg16.h5' 
CATEGORIES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
print(f"Loading Model: {MODEL_PATH}...")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"🔴 Error loading model: {e}")
else:
    print(f"🔴 ERROR: '{MODEL_PATH}' not found!")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img, (128, 128))
            input_data = np.array(new_array).reshape(-1, 128, 128, 3) / 255.0

            prediction = model.predict(input_data)
            class_index = np.argmax(prediction) 
            result_name = CATEGORIES[class_index]
            confidence = float(prediction[0][class_index]) * 100
            status = "safe" if result_name == "No Tumor" else "danger"

            return jsonify({
                'result': result_name,
                'confidence': f"{confidence:.2f}%",
                'status': status
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Chatbot Route
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        context_result = data.get('context', 'No context')

        #Greetings
        if "hello" in user_message or "hi" in user_message:
            reply = "Hello! I am Dr. AI. I can assist you with your MRI scan report."
        
        elif "thank" in user_message:
            reply = "You're welcome! Stay healthy. 🏥"

        #Risk
        elif "danger" in user_message or "serious" in user_message or "worry" in user_message:
            if "No Tumor" in context_result:
                reply = "Good news! The scan shows 'No Tumor', so there is no need to worry. Maintain a healthy lifestyle."
            else:
                reply = f"Since the scan detected a {context_result.split(',')[0]}, it requires medical attention. Please consult a neurologist immediately."

        #Treatment
        elif "treatment" in user_message or "cure" in user_message or "medicine" in user_message:
            if "No Tumor" in context_result:
                reply = "No treatment is needed as the scan is normal. Regular checkups are recommended."
            else:
                reply = "Treatment depends on the tumor size and location. Common options include surgery, radiation therapy, and medication."

        # Diet
        elif "food" in user_message or "eat" in user_message or "diet" in user_message:
            reply = "A balanced diet rich in antioxidants (fruits, vegetables, and nuts) supports brain health. Avoid processed foods and excessive sugar."

        # Explain 
        elif "explain" in user_message or "what is" in user_message:
            if "Glioma" in context_result:
                reply = "Glioma is a type of tumor that occurs in the brain and spinal cord. It affects the glial cells that support nerve cells."
            elif "Meningioma" in context_result:
                reply = "Meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull."
            elif "Pituitary" in context_result:
                reply = "Pituitary tumors are abnormal growths that develop in your pituitary gland. They can affect hormone levels."
            else:
                reply = "The scan result is normal. No abnormalities were detected in the brain structure."

        # Random smart answers
        else:
            smart_fillers = [
                "That's a good question. Based on the scan, I recommend discussing this further with a specialist.",
                "The AI analysis is 98% accurate, but clinical correlation by a doctor is always necessary.",
                "Could you please clarify? I am focused on analyzing the MRI results provided.",
                "Please consult a doctor for a detailed diagnosis tailored to your medical history."
            ]
            reply = random.choice(smart_fillers)

        return jsonify({'reply': reply})

    except Exception as e:
        print("Chat Error:", e)
        return jsonify({'reply': "I am strictly a medical AI. Please ask relevant questions."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)