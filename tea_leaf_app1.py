import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Define class names
class_names = [
    "Anthraconse",
    "Algal Leaf",
    "Bird_Eye Spot",
    "Brown Blight",
    "Red_Leaf Spot",
    "White Spot",
    "Gray Light",
    "Healthy"
]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img = tf.convert_to_tensor(img)
    img = tf.cast(img, dtype=tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

# Prediction function
def make_prediction(model, image):
    predictions = model.predict(image)

    if predictions.shape[1] != len(class_names):
        raise ValueError(
            f"Model returned {predictions.shape[1]} outputs, but {len(class_names)} class names are defined."
        )

    sorted_indices = tf.argsort(predictions[0])[::-1]
    sorted_indices = [int(i) for i in sorted_indices.numpy() if i < len(class_names)]

    if len(sorted_indices) < 2:
        raise ValueError("Not enough valid class indices returned by the model.")

    predicted_class = class_names[sorted_indices[0]]
    second_highest_class = class_names[sorted_indices[1]]

    return predicted_class, second_highest_class, predictions[0], sorted_indices

# Apply background color and styling
def apply_custom_styles():
    st.markdown(
        f"""
        <style>
        html, body {{
            background-color: #f0f4f7;
            font-family: 'Segoe UI', sans-serif;
        }}

        h1, h2, .stMarkdown, .st-bq, .st-c5 {{
            color: #2e7d32;
        }}

        .stFileUploader label {{
            font-weight: bold;
            color: #1b5e20;
        }}

        .stButton>button {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
        }}

        .stSpinner {{
            color: #00695c;
        }}

        .css-1aumxhk {{
            background-color: rgba(255, 255, 255, 0.7) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main app
def main():
    # Print working directory for debugging
    print("Current dir:", os.getcwd())
    print("Files in dir:", os.listdir())

    # Apply custom CSS
    apply_custom_styles()

    # Wrap all UI elements inside the main div class
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.title("🍃 Tea Leaf Disease Prediction System")
    st.write("Upload an image to detect the disease. The system supports the following types:")
    st.markdown("""
    - Algal Leaf  
    - Anthraconse  
    - Bird Eye Spot  
    - Brown Blight  
    - Red Leaf Spot  
    - White Spot  
    - Gray Light
    - Healthy  
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Analyzing image..."):
                image = preprocess_image(uploaded_file)

                model_path = r"E:/My/Projects/web_site/model/my_model.h5"
                if not os.path.exists(model_path):
                    st.error(f"Model file '{model_path}' not found!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                model = tf.keras.models.load_model(model_path)
                predicted_class, second_class, probabilities, sorted_indices = make_prediction(model, image)

                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                st.subheader(
                    f"🩺 Predicted Disease: {predicted_class} ({probabilities[sorted_indices[0]] * 100:.2f}%)"
                )
                st.write(
                    f"🧐 Second most likely: {second_class} ({probabilities[sorted_indices[1]] * 100:.2f}%)"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Close the main div wrapper
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
