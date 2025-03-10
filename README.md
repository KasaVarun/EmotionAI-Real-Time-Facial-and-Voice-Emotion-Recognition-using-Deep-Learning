# EmotionAI-Real-Time-Facial-and-Voice-Emotion-Recognition-using-Deep-Learning
EmotionAI is a comprehensive deep learning project designed to recognize human emotions from both facial expressions and voice. The project leverages state-of-the-art machine learning techniques to classify emotions in real-time, making it suitable for applications in mental health monitoring, human-computer interaction, and entertainment.
# Emotion Recognition Project

This project is designed to recognize emotions from facial expressions and voice using deep learning models. The project includes multiple components for training and deploying models for emotion recognition from images, videos, and audio files.

## Project Overview

The project consists of the following main components:

1. **Facial Emotion Recognition**: Recognizes emotions from facial expressions in images and live video streams.
2. **Voice Emotion Recognition**: Recognizes emotions from audio files using voice features.

The project uses TensorFlow and Keras for building and training deep learning models, OpenCV for image processing, and Librosa for audio feature extraction.

## Repository Structure

- **app.py**: The main Streamlit application for emotion recognition from images, videos, and audio.
- **emotion_model.py**: Script for building, training, and evaluating the facial emotion recognition model using a VGG16-based architecture.
- **facetrain.py**: Script for training a custom CNN model for facial emotion recognition.
- **train_facial_emotion.py**: Script for loading and preprocessing the facial emotion dataset.
- **voice_emotion_model.py**: Script for building, training, and evaluating the voice emotion recognition model.

## Algorithms and Models

### Facial Emotion Recognition

- **Model Architecture**:
  - **VGG16-based Model**: A pre-trained VGG16 model is used as the base, with additional dense layers for emotion classification.
  - **Custom CNN Model**: A custom CNN model with multiple convolutional layers, batch normalization, dropout, and dense layers is also used.

- **Training**:
  - The model is trained on a dataset of facial images labeled with emotions (angry, disgust, fear, happy, neutral, sad, surprise).
  - Data augmentation techniques (rotation, shifting, flipping, etc.) are applied to improve generalization.

- **Inference**:
  - The trained model is used to predict emotions from images and live video streams using OpenCV for face detection.

### Voice Emotion Recognition

- **Feature Extraction**:
  - **MFCC (Mel-frequency cepstral coefficients)**: 40 MFCC features are extracted from audio files to represent voice characteristics.

- **Model Architecture**:
  - A neural network with multiple dense layers and dropout for regularization is used for emotion classification.

- **Training**:
  - The model is trained on a dataset of audio files labeled with emotions.
  - The dataset is split into training and testing sets, and features are standardized.

- **Inference**:
  - The trained model is used to predict emotions from uploaded audio files.

## Workflow

1. **Data Preparation**:
   - Facial emotion dataset is loaded and preprocessed (resized, normalized, and augmented).
   - Voice emotion dataset is loaded, and MFCC features are extracted.

2. **Model Training**:
   - Facial emotion models (VGG16-based and custom CNN) are trained on the preprocessed dataset.
   - Voice emotion model is trained on the extracted MFCC features.

3. **Model Evaluation**:
   - Models are evaluated on validation/test sets to measure accuracy and loss.

4. **Deployment**:
   - The trained models are integrated into a Streamlit application for real-time emotion recognition from images, videos, and audio.

5. **Inference**:
   - Users can upload images or audio files, or use a webcam for live emotion detection.

## Usage

1. **Training Models**:
   - Run `emotion_model.py` or `facetrain.py` to train the facial emotion recognition models.
   - Run `voice_emotion_model.py` to train the voice emotion recognition model.

2. **Running the Application**:
   - Run `app.py` to start the Streamlit application.
   - Upload images or audio files, or use the webcam for live emotion detection.

## Dependencies

- TensorFlow
- Keras
- OpenCV
- Librosa
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn

## Conclusion

This project demonstrates the use of deep learning for emotion recognition from both facial expressions and voice. The models are trained on labeled datasets and deployed in a user-friendly Streamlit application for real-time emotion detection. The project can be extended with more advanced models and larger datasets for improved accuracy.
