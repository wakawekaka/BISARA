# Bisara: Sign Language to Speech Translator

Bisara is a real-time sign language recognition application that translates Indonesian Sign Language (Bahasa Isyarat Indonesia) into speech. It uses a deep learning model to recognize hand gestures from a live webcam feed and synthesizes the corresponding words as speech.

This project is live on GitHub Pages\! 🚀

## Features

  * **Real-time Hand Gesture Recognition**: Utilizes the MediaPipe library to detect and track hand landmarks from a webcam feed.
  * **Deep Learning Model**: An LSTM model built with TensorFlow and Keras classifies sequences of hand movements into corresponding words.
  * **Web-Based Interface**: The user interface is built with HTML, CSS, and JavaScript, allowing for easy access through any modern web browser.
  * **Text and Speech Output**: Detected signs are displayed as text and can be translated into audible speech using the browser's built-in speech synthesis capabilities.

## Technologies Used

  * **Front-End**: HTML, CSS, JavaScript
  * **Machine Learning**:
      * **Python**
      * **TensorFlow / Keras** for building and training the LSTM model.
      * **TensorFlow.js** for running the model directly in the browser.
      * **MediaPipe** for hand landmark detection.
      * **Jupyter Notebook** for model development and experimentation.
      * **NumPy** and **OpenCV** for data preprocessing.

-----

## How It Works

1.  **Data Collection**: The model was trained on video sequences of different sign language gestures.
2.  **Landmark Extraction**: For each video frame, MediaPipe extracts 21 key landmarks for each hand.
3.  **Sequence Processing**: A sequence of 30 frames is fed into the LSTM model. Data augmentation (flipping hand coordinates) is used to create a more robust model.
4.  **Model Training**: The LSTM model is trained to classify these sequences into one of the predefined sign categories.
5.  **Real-time Prediction**: In the browser, `script.js` uses the webcam feed, sends it to MediaPipe for landmark extraction, and then passes the data to the TensorFlow.js model for real-time prediction.
6.  **Speech Synthesis**: The predicted word is appended to a textbox. The text can then be converted to speech using the Web Speech API.