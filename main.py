import streamlit as st
from PIL import Image
import tensorflow as tf

st.title('Vietnamese Food Recognition')

# Load the model from a .h5 file
model = tf.keras.models.load_model('food_recognition_model.h5')

st.header('Upload an image of Vietnamese food')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        # Preprocess the image for prediction
        image = image.resize((224, 224))  # Adjust to the input size of your model
        image_array = np.array(image)
        image_array = image_array / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Display the predicted class (replace with your class labels)
        class_labels = ["Pho", "Banh Mi", "Bun Cha", "Com Tam", "..."]
        st.header('Result')
        st.text(f'Predicted Food: {class_labels[predicted_class]}')
