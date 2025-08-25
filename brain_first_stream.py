import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import os
from collections import Counter

# Disable GPU to avoid CUDA/XLA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class_names = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']

# Load models
yolo_model = YOLO('/home/ciphermind/programing/python/anaconda/jupyter/runs/detect/train5/weights/best.pt')
class_model = tf.keras.models.load_model('/home/ciphermind/programing/python/anaconda/jupyter/my_full_pre_train_model/brain_tumor_modelv3.keras')


def preprocess_for_classifier(img, target_size=(224, 224)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def yolo_confidence_scores(yolo_preds):
    """Return weighted confidence scores per class from YOLO detections."""
    scores = np.zeros(len(class_names))
    for box in yolo_preds:
        class_id = int(box.cls)
        confidence = float(box.conf)
        scores[class_id] += confidence
    if np.sum(scores) == 0:
        return np.ones(len(class_names)) / len(class_names)  # fallback uniform if nothing detected
    return scores / np.sum(scores)


def ensemble_yolo_cnn(yolo_preds, cnn_pred, weights=[0.7, 0.3]):
    yolo_scores = yolo_confidence_scores(yolo_preds)
    cnn_scores = cnn_pred[0]

    combined_scores = yolo_scores * weights[0] + cnn_scores * weights[1]
    final_class_idx = np.argmax(combined_scores)
    final_confidence = combined_scores[final_class_idx]
    return class_names[final_class_idx], final_confidence


def detect_and_classify(uploaded_file):
    pil_img = Image.open(uploaded_file).convert('RGB')
    img = np.array(pil_img)
    img_draw = img.copy()
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    # YOLO detections
    yolo_results = yolo_model(pil_img)
    yolo_detections = []

    if len(yolo_results) > 0:
        for r in yolo_results:
            for box in r.boxes:
                yolo_detections.append(box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (144, 233, 34), 2)

    # CNN prediction
    cnn_input = preprocess_for_classifier(img)
    cnn_prediction = class_model.predict(cnn_input, verbose=0)

    # Ensemble
    final_class, final_confidence = ensemble_yolo_cnn(yolo_detections, cnn_prediction)

    # Draw final label
    label = f"{final_class} {final_confidence:.2f}"
    if yolo_detections:
        x1, y1, _, _ = map(int, yolo_detections[0].xyxy[0])
        cv2.putText(img_draw, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 233, 34), 2)
    else:
        cv2.putText(img_draw, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    return img_rgb, final_class, final_confidence


# Streamlit UI
st.title("Brain Tumor Detection & Classification (YOLO + CNN Ensemble)")

uploaded_files = st.file_uploader("Upload MRI images",
                                  type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        result_img, pred_class, confidence = detect_and_classify(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(result_img, caption=f"Result - {uploaded_file.name}")
        with col2:
            st.subheader("Prediction Details")
            st.write(f"**Class:** {pred_class}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.write("**Models Used:** YOLO + CNN Ensemble (0.7 YOLO, 0.3 CNN)")

        st.write("---")

