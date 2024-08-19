import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from groq import Groq
import os

# Load your model
model = tf.keras.models.load_model("model.keras")
class_names = ['Corn___Common_Rust',
               'Corn___Gray_Leaf_Spot',
               'Corn___Healthy',
               'Corn___Northern_Leaf_Blight',
               'Pepper__bell___Bacterial_spot',
               'Pepper__bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Rice___Brown_Spot',
               'Rice___Healthy',
               'Rice___Leaf_Blast',
               'Rice___Neck_Blast',
               'Sugarcane__Healthy',
               'Sugarcane__Mosaic',
               'Sugarcane__RedRot',
               'Sugarcane__Rust',
               'Sugarcane__Yellow',
               'Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy',
               'Wheat___Brown_Rust',
               'Wheat___Healthy',
               'Wheat___Yellow_Rust']

# Groq API setup
api_key = "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"
client = Groq(api_key=api_key)

# Function to get summary from Groq API
def get_summary_from_groq(predicted_class):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Provide a detailed solution about the plant disease: {predicted_class}.",
                }
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content
    except Exception as e:
        return "No summary available for this disease."

# Function to predict with OOD detection
def predict_with_ood_detection(img, model, threshold=0.5):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)
    
    if max_prob < threshold:
        return "OOD", max_prob, "The image is considered out-of-distribution."
    else:
        predicted_class = class_names[np.argmax(predictions[0])]
        summary = get_summary_from_groq(predicted_class)
        return predicted_class, max_prob, summary

# Streamlit app
st.title("Crop Disease Classification with OOD Detection")
st.write("Upload an image of the crop.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict the class, confidence, and summary with OOD detection
    predicted_class, confidence, summary = predict_with_ood_detection(image, model)

    # Display the results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
    st.write("**Summary:**")
    st.write(summary)

