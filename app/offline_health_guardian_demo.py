# offline_health_guardian_full_fixed.py
import streamlit as st
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
import cv2
import os
import tempfile
import time
import librosa
import json
from datetime import datetime

st.title("🩺 AI Health Guardian - Demo MVP (Fixed)")
st.markdown("⚠️ **Demo only: predictions are NOT medically accurate**")

# -----------------------------
# Demo model placeholders
# -----------------------------
ANEMIA_MODEL_PATH = "E:/AI_Project/models/anemia_model_finetuned_final.h5"
COUGH_MODEL_PATH = "E:/AI_Project/models/cough_risk_binary_cnn.h5"
RPPG_MODEL_PATH = "E:/AI_Project/models/rppg_model_finetuned.h5"

# Load models safely (demo)
try: anemia_model = tf.keras.models.load_model(ANEMIA_MODEL_PATH)
except: anemia_model = None
try: cough_model = tf.keras.models.load_model(COUGH_MODEL_PATH)
except: cough_model = None
try: rppg_model = tf.keras.models.load_model(RPPG_MODEL_PATH)
except: rppg_model = None

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_image(file, size=(224,224)):
    img = Image.open(file).convert("RGB").resize(size)
    return np.expand_dims(np.array(img)/255.0, axis=0)

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return torch.tensor(mfccs).unsqueeze(0).unsqueeze(0).float()

def live_camera_demo(frames=50):
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    for i in range(frames):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        time.sleep(0.05)
    cap.release()

# -----------------------------
# Save last scan
# -----------------------------
def save_last_scan(module_name, result_dict):
    data = {
        "module": module_name,
        "result": result_dict,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("last_scan.json", "w") as f:
        json.dump(data, f, indent=4)
# -----------------------------
# Sidebar navigation
# -----------------------------
app_mode = st.sidebar.radio("Choose Module", [
    "Home", "Anemia Scan", "Cough/Breath Scan", "HR/BP/Stress/Emotion",
    "Diabetes Quiz", "Symptoms Advice", "Vitamin/Mineral Check",
    "BMI Calculator", "ECG (No Demo)", "NLP Advice (No Demo)"
])

# -----------------------------
# Modules
# -----------------------------
if app_mode == "Home":
    st.write("Welcome! Use the sidebar to navigate.")

# 1️⃣ Anemia Scan
elif app_mode == "Anemia Scan":
    st.header("👁️ Anemia Detection Demo")
    st.info("Use camera or upload image. Predictions are demo only.")
    
    uploaded_file = st.file_uploader("Upload eye image (jpg/png)", type=["jpg","png"])
    if uploaded_file:
        # Demo prediction: realistic placeholders
        pred_prob = np.random.uniform(0.05, 0.95)
        st.write(f"Predicted anemia probability: {pred_prob*100:.1f}%")
        if pred_prob < 0.5:
            st.success("✅ Likely Healthy")
            st.info("Advice: Maintain balanced diet with iron-rich foods like leafy greens, beans, and lean meat.")
        else:
            st.warning("⚠️ Possible Anemia")
            st.info("Advice: Consult a doctor for blood tests, consider iron supplements if needed.")
    
    if st.button("Start Live Camera Scan"):
        live_camera_demo()
        pred_prob = np.random.uniform(0.05, 0.95)
        st.write(f"Predicted anemia probability: {pred_prob*100:.1f}%")
        if pred_prob < 0.5:
            st.success("✅ Likely Healthy")
            st.info("Advice: Maintain balanced diet with iron-rich foods like leafy greens, beans, and lean meat.")
            status = "Likely Healthy"
            advice = ["Maintain balanced diet with iron-rich foods like leafy greens, beans, and lean meat."]
        else:
            st.warning("⚠️ Possible Anemia")
            st.info("Advice: Consult a doctor for blood tests, consider iron supplements if needed.")
            status = "Possible Anemia"
            advice = ["Consult a doctor for blood tests, consider iron supplements if needed."]

        save_last_scan("Anemia Scan", {"pred_prob": pred_prob, "status": status, "advice": advice})

# 2️⃣ Cough/Breath Scan
elif app_mode == "Cough/Breath Scan":
    st.header("🤧 Cough & Breath Demo")
    st.info("Use microphone or upload audio. Demo only.")
    
    uploaded_audio = st.file_uploader("Upload audio (wav)", type=["wav"])
    if uploaded_audio:
        pred_prob = np.random.uniform(0.05, 0.95)
        st.write(f"Predicted respiratory risk: {pred_prob*100:.1f}%")
        if pred_prob < 0.5:
            st.success("✅ Normal")
            st.info("Advice: Breathing appears normal. Maintain hydration and avoid smoke/pollution.")
        else:
            st.warning("⚠️ Possible Respiratory Issue")
            st.info("Advice: Monitor symptoms, consult a doctor if cough persists, consider lung check-up.")
    
    if st.button("Start Live Audio Scan (Demo)"):
        st.warning("Recording 5 seconds...")
        time.sleep(5)
        pred_prob = np.random.uniform(0.05, 0.95)
        st.write(f"Predicted respiratory risk: {pred_prob*100:.1f}%")
        if pred_prob < 0.5:
            st.success("✅ Normal")
            st.info("Advice: Breathing appears normal. Maintain hydration and avoid smoke/pollution.")
            status = "Normal"
            advice = ["Breathing appears normal. Maintain hydration and avoid smoke/pollution."]
        else:
            st.warning("⚠️ Possible Respiratory Issue")
            st.info("Advice: Monitor symptoms, consult a doctor if cough persists, consider lung check-up.")
            status = "Possible Respiratory Issue"
            advice = ["Monitor symptoms, consult a doctor if cough persists, consider lung check-up."]

        save_last_scan("Cough/Breath Scan", {"pred_prob": pred_prob, "status": status, "advice": advice})
# 3️⃣ HR/BP/Stress/Emotion (rPPG demo)
elif app_mode == "HR/BP/Stress/Emotion":
    st.header("❤️ HR/BP/Stress/Emotion Live Demo")
    st.info("Camera scanning demo. Predictions are demo placeholders only.")
    
    if st.button("Start Live Scan"):
        live_camera_demo(frames=50)  # Live camera feed

        # ---- Demo predictions with medically plausible ranges ----
        hr = np.random.randint(60, 100)  # Heart rate in bpm
        bp_sys = np.random.randint(110, 130)  # Systolic BP
        bp_dia = np.random.randint(70, 85)    # Diastolic BP
        stress_level = np.random.choice(["Low", "Moderate", "High"])
        emotion = np.random.choice(["Happy", "Sad", "Neutral", "Angry", "Surprised"])
        advice = []

        # ---- Display results ----
        st.subheader("📊 Predicted Health Metrics (Demo)")
        st.write(f"**Heart Rate (HR):** {hr} bpm")
        st.write(f"**Blood Pressure (BP):** {bp_sys}/{bp_dia} mmHg")
        st.write(f"**Stress Level:** {stress_level}")
        st.write(f"**Detected Emotion:** {emotion}")

        # ---- Provide medically plausible advice based on values ----
        if hr < 60:
            st.warning("⚠️ Heart rate below normal: may indicate bradycardia. Monitor and consult doctor if persistent.")
            advice.append("Heart rate below normal")
        elif hr > 100:
            st.warning("⚠️ Heart rate above normal: may indicate tachycardia. Monitor and consult doctor if persistent.")
            advice.append("Heart rate above normal")
        else:
            st.success("✅ Heart rate within normal range.")

        if bp_sys > 120 or bp_dia > 80:
            st.warning("⚠️ Blood pressure slightly elevated. Maintain healthy diet, exercise, and monitor regularly.")
            advice.append("Slightly elevated BP")
        else:
            st.success("✅ Blood pressure within normal range.")

        if stress_level == "High":
            st.warning("⚠️ High stress detected. Consider relaxation techniques and rest.")
            advice.append("High stress detected")
        elif stress_level == "Moderate":
            st.info("ℹ️ Moderate stress: maintain balanced lifestyle.")
            advice.append("Moderate stress")
        else:
            st.success("✅ Low stress: good condition.")

        st.info("ℹ️ Emotion detection is demo only; use for awareness, not diagnosis.")

        save_last_scan("HR/BP/Stress/Emotion", {
            "hr": hr,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "stress_level": stress_level,
            "emotion": emotion,
            "advice": advice
        })
        save_last_scan("HR/BP/Stress/Emotion", {
            "hr": hr,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "stress_level": stress_level,
            "emotion": emotion,
            "advice": advice
        })

# 4️⃣ Diabetes Quiz
elif app_mode == "Diabetes Quiz":
    st.header("📝 Diabetes Risk Quiz Demo")
    age = st.number_input("Age", 1, 120)
    weight = st.number_input("Weight (kg)", 1, 300)
    height = st.number_input("Height (cm)", 50, 250)
    family_history = st.radio("Family history of diabetes?", ["Yes","No"])
    glucose = st.number_input("Fasting glucose (mg/dL)", 50, 500)
    
    risk_score = 0
    risk_score += 1 if age>45 else 0
    risk_score += 1 if weight/((height/100)**2)>25 else 0
    risk_score += 2 if family_history=="Yes" else 0
    risk_score += 2 if glucose>126 else 0
    if st.button("Check Diabetes Risk"):
        risk_percent = (risk_score/6)*100
        st.write(f"Diabetes Risk Score: {risk_score}/6 ({risk_percent:.1f}%)")
        if risk_score<3:
            st.success("✅ Low risk")
            st.info("Advice: Maintain healthy diet and regular exercise.")
            status = "Low risk"
            advice = ["Maintain healthy diet and regular exercise."]
        else:
            st.warning("⚠️ High risk")
            st.info("Advice: Consult doctor, monitor blood sugar regularly, maintain active lifestyle.")
            status = "High risk"
            advice = ["Consult doctor, monitor blood sugar regularly, maintain active lifestyle."]
        save_last_scan("Diabetes Quiz", {"risk_score": risk_score, "risk_percent": risk_percent, "status": status, "advice": advice})
        risk_percent = (risk_score/6)*100
        st.write(f"Diabetes Risk Score: {risk_score}/6 ({risk_percent:.1f}%)")
        if risk_score<3:
            st.success("✅ Low risk")
            st.info("Advice: Maintain healthy diet and regular exercise.")
        else:
            st.warning("⚠️ High risk")
            st.info("Advice: Consult doctor, monitor blood sugar regularly, maintain active lifestyle.")
        save_last_scan("Diabetes Quiz", {"risk_score": risk_score, "risk_percent": risk_percent, "status": status, "advice": advice})


# 5️⃣ Symptoms Advice
elif app_mode == "Symptoms Advice":
    st.header("💡 Symptoms & Advice Demo")
    fatigue = st.checkbox("Fatigue")
    dizziness = st.checkbox("Dizziness")
    breathless = st.checkbox("Breathlessness")
    irregular_heartbeat = st.checkbox("Irregular heartbeat")
    thirst = st.checkbox("Excessive thirst")
    
    risk = np.random.uniform(0,1)
    st.write(f"Demo risk score: {risk*100:.1f}%")
    
    # advice depends on symptoms
    advice = []
    if fatigue: advice.append("Ensure proper rest and balanced diet.")
    if dizziness: advice.append("Monitor blood pressure, consult doctor if frequent.")
    if breathless: advice.append("Check respiratory health, avoid heavy exertion.")
    if irregular_heartbeat: advice.append("Monitor heart, see cardiologist if persists.")
    if thirst: advice.append("Check blood sugar, stay hydrated.")
    
    if advice:
        for a in advice:
            st.warning(a)
    else:
        st.success("✅ No major symptoms detected. Maintain healthy lifestyle.")
    
    save_last_scan("Symptoms Advice", {"risk_score": risk, "advice": advice})

# 6️⃣ Vitamin/Mineral Check
elif app_mode == "Vitamin/Mineral Check":
    st.header("🍎 Vitamin & Mineral Demo")
    st.info("Future: scanning face, lips, hair and diet analysis coming soon.")
    iron = st.number_input("Iron intake (mg/day)", 0, 100)
    calcium = st.number_input("Calcium intake (mg/day)", 0, 2000)
    vitamin_d = st.number_input("Vitamin D intake (IU/day)", 0, 5000)
    if st.button("Check Deficiency"):
        iron_prob = np.clip((18-iron)/18,0,1)
        calcium_prob = np.clip((1000-calcium)/1000,0,1)
        vitamin_d_prob = np.clip((600-vitamin_d)/600,0,1)
        
        st.write(f"Iron deficiency probability: {iron_prob*100:.1f}%")
        st.write(f"Calcium deficiency probability: {calcium_prob*100:.1f}%")
        st.write(f"Vitamin D deficiency probability: {vitamin_d_prob*100:.1f}%")
        
        advice = []
        if iron<18:
            st.warning("⚠️ Iron deficiency likely. Eat leafy greens, beans, or consider supplements.")
            advice.append("Iron deficiency")
        else:
            st.success("✅ Iron sufficient")
        
        if calcium<1000:
            st.warning("⚠️ Calcium deficiency likely. Consider dairy or fortified foods.")
            advice.append("Calcium deficiency")
        else:
            st.success("✅ Calcium sufficient")
        
        if vitamin_d<600:
            st.warning("⚠️ Vitamin D deficiency likely. Ensure sunlight exposure and supplementation.")
            advice.append("Vitamin D deficiency")
        else:
            st.success("✅ Vitamin D sufficient")
        
        save_last_scan("Vitamin/Mineral Check", {
            "iron_prob": iron_prob,
            "calcium_prob": calcium_prob,
            "vitamin_d_prob": vitamin_d_prob,
            "advice": advice
        })


# 7️⃣ BMI Calculator
elif app_mode == "BMI Calculator":
    st.header("⚖️ BMI Demo")
    weight = st.number_input("Weight (kg)", 1, 300, key="bmi_weight")
    height = st.number_input("Height (cm)", 50, 250, key="bmi_height")
    
    if st.button("Calculate BMI"):
        bmi = weight / ((height/100)**2)
        st.write(f"Your BMI: {bmi:.1f}")
        # Provide medically plausible categories
        if bmi < 18.5:
            status = "Underweight"
            advice = "Consider nutrient-rich diet and consult nutritionist."
            st.warning(f"⚠️ {status}. {advice}")
        elif bmi < 25:
            status = "Normal weight"
            advice = "Maintain healthy lifestyle."
            st.success(f"✅ {status}. {advice}")
        elif bmi < 30:
            status = "Overweight"
            advice = "Monitor diet, exercise regularly."
            st.warning(f"⚠️ {status}. {advice}")
        else:
            status = "Obese"
            advice = "Strongly advised to consult healthcare provider and manage weight."
            st.warning(f"⚠️ {status}. {advice}")
        
        save_last_scan("BMI Calculator", {"bmi": bmi, "status": status, "advice": advice})


# 8️⃣ ECG
elif app_mode == "ECG (No Demo)":
    st.warning("❌ No demo available. ECG module coming in future versions.")

# 9️⃣ NLP Advice
elif app_mode == "NLP Advice (No Demo)":
    st.warning("❌ No demo available. NLP advice module coming in future versions.")
