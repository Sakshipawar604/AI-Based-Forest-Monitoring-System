import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


# Load pre-trained model
MODEL_PATH = 'deforestation_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train the model first.")
model = load_model(MODEL_PATH)



def predict_deforestation(image_path):
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    mask = model.predict(image_array)[0].squeeze()
    deforestation_percentage = np.mean(mask) * 100.0
    
    # Show results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(load_img(image_path))
    axes[0].set_title('Original Image')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Deforestation: {deforestation_percentage:.2f}%')
    st.pyplot(fig)
    return deforestation_percentage

def detect_tree_areas(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, "Error: Image not found!"

    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_green_area = sum(cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 500)

    total_pixels = image.shape[0] * image.shape[1]
    green_percent = (total_green_area / total_pixels) * 100

    area_type = "Urban or Barren land" if green_percent < 5 else \
                "Sparse greenery - likely residential or agricultural boundary" if green_percent < 20 else \
                "Semi-urban or mixed vegetation" if green_percent < 50 else "Dense forest or garden area"
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(original, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    
    info_lines = [
        f"[INFO] Image Resolution: {image.shape[1]}x{image.shape[0]}",
        f"[INFO] Detected {len(contours)} significant green regions.",
        f"[INFO] Approx. Green Coverage Area: {green_percent:.2f}% of the image",
        f"[INFO] Area Type Guess: {area_type}",
        "[INFO] Possible Trees in this region: Neem, Banyan, Mango, Coconut, Gulmohar (Common in Indian urban & semi-urban)"
    ]
    
    return original, mask, info_lines

def count_trees(image_path):
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    
    model = load_model("tree_detector_vgg16_unet.h5")
    original_img = load_img(image_path)
    img_resized = load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_mask = model.predict(img_array)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = pred_mask.squeeze()
    
    pred_mask_resized = cv2.resize(pred_mask, (original_img.size[0], original_img.size[1]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tree_count = len(contours)
    
    output_img = np.array(original_img)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    return output_img, tree_count

def forest_canopy_analysis(image_path):
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    
    model = load_model('forest_canopy_unet.h5')
    
    img = load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = img_to_array(img) / 255.0
    input_img = np.expand_dims(img_array, axis=0)
    
    predicted_mask = model.predict(input_img)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    
    canopy_pixels = np.sum(predicted_mask)
    total_pixels = IMAGE_HEIGHT * IMAGE_WIDTH
    canopy_percentage = (canopy_pixels / total_pixels) * 100
    
    carbon_stock = canopy_percentage * 1.5
    
    predicted_mask_visual = (predicted_mask * 255).astype(np.uint8)
    predicted_mask_visual = predicted_mask_visual[:, :, 0]  # Take the first channel
    
    return img, predicted_mask_visual, canopy_percentage, carbon_stock

st.set_page_config(page_title="Satellite Images Forest ML Project", layout="wide")

st.sidebar.title("Menu")
option = st.sidebar.selectbox("Select an option", ["Home", "Tree Species", "Tree Count", "Forest Canopy and Estimated Carbon Stock","Deforestation Prediction"])

if option == "Forest Canopy and Estimated Carbon Stock":
    st.title("Forest Canopy and Estimated Carbon Stock")
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_path = "temp_forest.jpg"
        image.save(img_path)
        
        img, predicted_mask, canopy_percentage, carbon_stock = forest_canopy_analysis(img_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(predicted_mask, caption="Predicted Canopy Mask", use_column_width=True, clamp=True)
        
        st.write(f"Forest Canopy Percentage: {canopy_percentage:.2f}%")
        st.write(f"Estimated Carbon Stock: {carbon_stock:.2f} tons/ha equivalent")

elif option == "Home":
    st.title("Satellite Images Forest ML Project")
    st.subheader("Project Overview")
    st.write("This project aims to analyze satellite images and detect green areas (forests, vegetation) using machine learning techniques.")
    st.subheader("Objectives")
    st.write("- Automate green cover analysis from satellite images.")
    st.write("- Estimate tree species distribution based on green coverage.")
    st.write("- Provide insights for environmental monitoring and urban planning.")
    st.subheader("Conclusion")
    st.write("This project provides an automated way to analyze vegetation coverage, classify land types, and offer insights into tree distribution, aiding environmental and urban research.")

elif option == "Tree Species":
    st.title("Tree Species Detection")
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_path = "temp.jpg"
        image.save(img_path)
        
        processed_image, mask_image, info_lines = detect_tree_areas(img_path)
        
        if processed_image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.image(processed_image, caption="Detected Green Areas", use_column_width=True, channels="BGR")
                
            st.image(mask_image, caption="Green Mask", width=500, channels="GRAY")
            
            for line in info_lines:
                st.write(line)
        else:
            st.error("Error processing the image. Please try a different one.")
        
elif option == "Tree Count":
    st.title("Tree Count Detection")
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_path = "temp_tree.jpg"
        image.save(img_path)
        
        tree_image, tree_count = count_trees(img_path)
        
        st.image(tree_image, caption=f"Detected Trees: {tree_count}", width=500, channels="BGR")
        st.write(f"Detected Trees: {tree_count}")


elif option == "Deforestation Prediction":
    st.title("Deforestation Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image",width=500)
        
        if st.button("Predict Deforestation"):
            percentage = predict_deforestation(img_path)
            st.write(f"### Predicted Deforestation: {percentage:.2f}%")