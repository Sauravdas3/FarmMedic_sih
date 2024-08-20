# FarmMedic

FarmMedic is an AI-powered application developed in response to the SIH 2024 Problem Statement 1638, which focuses on providing a quick and effective solution for diagnosing and managing plant diseases and pest infestations in farming. By leveraging machine learning models, FarmMedic offers farmers real-time assistance in identifying issues with their crops through image-based analysis and recommending corrective actions.

## Problem Statement 1638 - SIH 2024

The problem statement, as issued by the Ministry of Agriculture, tasked participants with creating an innovative and cost-effective solution that enables farmers to:

1. Detect plant diseases and pest infestations through image recognition.
2. Recommend appropriate remedies and treatments.
3. Provide real-time assistance via a mobile or web application, even in remote areas with poor internet connectivity.

## Key Features

- **Image Recognition**: Using machine learning and image processing, FarmMedic accurately identifies plant diseases and pests from photos uploaded by farmers.
- **Real-Time Recommendations**: Based on the analysis, the app provides immediate suggestions on how to treat the identified disease or pest infestation.
- **Offline Mode**: The app is designed to work efficiently even in remote areas with limited or no internet connectivity by utilizing local storage and offline data.
- **User-Friendly Interface**: The interface is designed to be intuitive, with support for regional languages and easy-to-understand instructions.

## Technologies Used

### Backend

- **Streamlit**: Used for creating an interactive and easy-to-use web interface for farmers to upload images and receive diagnoses. Streamlit also handles the backend processes like calling the machine learning models for predictions.
- **Flask**: If necessary, Flask can be used for serving additional API endpoints, but Streamlit handles most of the backend functionality.
- **TensorFlow/Keras**: The ML models are built and trained using TensorFlow/Keras for high accuracy in disease and pest detection.

### Machine Learning

- **Image Classification Models**: Trained using a large dataset of plant images annotated with disease and pest labels. These models use convolutional neural networks (CNNs) to recognize patterns in the images and predict the presence of specific diseases or pests.

### Deployment

- **Streamlit Cloud**: The FarmMedic app is deployed using Streamlit Cloud, which provides a quick and simple way to host the application online.


