# ASL Interpretation using Machine Learning

This project demonstrates the implementation of an Artificial Intelligence system for interpreting American Sign Language (ASL) gestures in real-time through a webcam. It leverages machine learning techniques and utilizes both video and image data to train a model capable of recognizing and interpreting ASL gestures.

## Objective

The primary objective of this project is to create a system that can accurately interpret ASL gestures captured through a webcam, enabling communication in ASL for individuals who are deaf or hard of hearing. 

## Technologies Used

- Python
- TensorFlow
- OpenCV
- Streamlit

## Project Overview

### Data Collection and Preprocessing

- **`data/`**: Contains self - maade  datasets comprising video recordings and image frames of various ASL gestures. These datasets are utilized for training the machine learning model. The data undergoes preprocessing to extract essential features and labels required for training. The model can predict five asl words [hello,thanks,iloveyou,yes, peace] into text.

### Model Training

- **`models/`**: Stores the trained machine learning models. Typically, convolutional neural networks (CNNs) or similar architectures are used for training with the preprocessed video and image data.

### Environment

- **`VsCode`**: We used visual studio code to implement the project because of the internet instability factor of jupyter notebook and colab

### Real-time Interpretation Interface

- **`deployment.py`**: This streamlit application serves as the interface for real-time ASL interpretation through a connected webcam.

## Setup and Usage
 To use the web app, run  deployment .py and enter 'streamlit run deployment.py' in your command prompt or terminal. OR enter and search this web address 'http://localhost:8501'

## Evaluation
 Metrics like categorical accuracy  was employed to measure the accuracy of out model. With an accuracy of 100%



