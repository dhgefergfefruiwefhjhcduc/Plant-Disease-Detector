import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow import keras
import json
from datetime import datetime

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_detection_model.h5')
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)
    return model, class_labels

model, class_labels = load_model()

def main():
    st.title("🌿 Plant Disease Detection")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Save the file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display image
        img = Image.open("temp_image.jpg")
        st.image(img, use_container_width=True)
        
        if st.button("Analyze Image"):
            # Process image
            img_processed = keras.preprocessing.image.load_img("temp_image.jpg", target_size=(150, 150))
            img_array = keras.preprocessing.image.img_to_array(img_processed)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_class = class_labels[predicted_class_idx]
            
            st.success(f"**Plant type:** {predicted_class.split('___')[0]}")
            st.success(f"**Disease name:** {predicted_class.split('___')[1].replace('_', ' ')}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
if __name__ == "__main__":
    main()