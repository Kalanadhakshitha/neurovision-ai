# 🧠 NeuroVision AI - Brain Tumor Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18.0-61DAFB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-black)

**NeuroVision AI** is an advanced web application designed to assist in the early detection of brain tumors using MRI scans. Powered by a Deep Learning model (**VGG16**), it classifies MRI images into four categories with high accuracy. The system also features an AI Chatbot (**Dr. AI**) to provide context-aware medical assistance.

---

## 🚀 Live Demo

Check out the live application here:  
👉 **[https://neurovision-ai.vercel.app](https://neurovision-ai.vercel.app)** _(Backend deployed on PythonAnywhere | Frontend on Vercel)_

---

## ✨ Key Features

- **🖼️ MRI Image Analysis:** Upload an MRI scan to detect tumors instantly.
- **🔍 4-Way Classification:** Detects **Glioma, Meningioma, Pituitary tumors**, or **No Tumor**.
- **🤖 Dr. AI Chatbot:** An integrated medical assistant that answers questions based on the diagnosis.
- **⚡ Real-time Results:** Fast and accurate predictions.
- **📱 Responsive Design:** Works smoothly on both desktop and mobile devices.

---

## 🛠️ Tech Stack

- **Frontend:** React.js, CSS3
- **Backend:** Flask (Python)
- **AI/ML Model:** TensorFlow, Keras, VGG16 (Transfer Learning)
- **Deployment:** Vercel (Frontend), PythonAnywhere (Backend)

---

## 📸 Screenshots

<img width="1537" height="858" alt="ss3" src="https://github.com/user-attachments/assets/e86bae8b-b1f8-43dd-b5f6-303f3b83719c" />
<img width="648" height="840" alt="ss2" src="https://github.com/user-attachments/assets/437f5794-286e-4472-bad5-113013197744" />
<img width="1787" height="852" alt="ss1" src="https://github.com/user-attachments/assets/1c2b404c-651b-4a21-82c8-6dc4893e2703" />


## ⚙️ How to Run Locally

Follow these steps to set up the project on your local machine.

1. Clone the Repository

git clone [https://github.com/Kalanadhakshitha/neurovision-ai.git](https://github.com/Kalanadhakshitha/neurovision-ai.git)
cd neurovision-ai

2. Backend Setup (Flask)
   Navigate to the root directory (where app.py is located):

# Install dependencies

pip install -r requirements.txt

# Run the Flask Server

python app.py

#The backend will start at http://127.0.0.1:5000

3. Frontend Setup (React)
   Open a new terminal and navigate to the client folder:

cd client

# Install Node modules

npm install

# Start the React App

npm start

The app will open at http://localhost:3000

🧠 Model Information
The AI model utilizes VGG16 architecture trained on a dataset of MRI scans. It achieves high accuracy by leveraging Transfer Learning techniques to identify complex patterns in brain imaging.

Disclaimer: This tool is for educational and assistive purposes only and should not replace professional medical diagnosis.
