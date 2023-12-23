# app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Load pre-trained model
model = tf.keras.applications.VGG16(weights='imagenet')

# Function to predict the food item
def predict_food(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.vgg16.decode_predictions(predictions, top=1)[0]
    
    return decoded_predictions[0][1]

# Streamlit app
st.title("Vietnamese Food Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image(Image.open(uploaded_file), caption="Uploaded Image.", use_column_width=True)

    st.write("")
    st.write("Classifying...")

    food_name = predict_food(image_path)
    st.success(f"The food in the image is: {food_name}")

# Run the app
if __name__ == "__main__":
    st.run_app()
