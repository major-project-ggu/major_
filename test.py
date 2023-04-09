import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image


# Load the trained model
model = load_model('C:\\Users\\satad\\Downloads\\breast_cancer_model.h5')

st.title('Breast Cancer Diagnosis')
name = st.text_input("Name")
col1,col2 = st.columns(2)
age = col1.text_input("Age")
gender = col2.selectbox("Gender",("Male","Female"))
phone = col1.text_input("Phone No.")
date = col2.date_input("Date")

def display_images(img2,segmented_image,contour_img):
    col1.image([img2], caption=['Input Image'])
    if(col1.button("submit")):
        col2.image([segmented_image], caption=['Segmented Image'])
        # Resize and preprocess the image for diagnosis prediction
        img = cv2.resize(contour_img, (64, 64))
        img = np.reshape(img, (1, 64, 64, 1))

        # Predict the class of the image
        y_pred = model.predict(img)
        y_pred = np.argmax(y_pred, axis=1)

        col2.write("Report Section:")
        col2.write(name, caption = "Name : ")
        col2.write(age, caption = "Age : ")
        col2.write(gender, caption = "Gender : ")
        col2.write(phone, caption = "Phone : ")
        col2.write(date, caption = "Date : ")
        # Print the predicted class
        # col2.write("Outcome: ")
        if np.all(y_pred == 0):
            col2.write('Remark: The image is normal.')
        elif np.any(y_pred == 2):
            col2.write('Remark: The image is malignant.')
        else:
            col2.write('Remark: The image is benign.')


def segment_and_diagnose(image_path):
    input_image = cv2.imread(image_path)
    img2 = cv2.resize(input_image, (256, 256))

    # Convert the input image to grayscale
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur with a kernel size of (3, 3) to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Apply adaptive thresholding using the "adaptive Gaussian" method
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 2)

    # Apply erosion followed by dilation to remove small noise and connect broken contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a black image
    contour_img = np.zeros(img2.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)

    # Apply segmentation to the input image
    segmented_image = cv2.bitwise_and(img2, img2, mask=contour_img)
    # Display the input image and segmented image
    display_images(img2,segmented_image,contour_img)
# Streamlit web app
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'pgm'])
if uploaded_file is not None:
    # Save the uploaded image
    image = Image.open(uploaded_file)
    image_path = 'uploaded_image.jpg'
    image.save(image_path)
    # Segment and diagnose the uploaded image
    segment_and_diagnose(image_path)