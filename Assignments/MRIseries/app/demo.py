import pydicom
import os, re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
#import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image, ImageDraw
from glob import glob
import sys
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from alibi.explainers import AnchorImage

from scripts.demo_utils import check_prediction_tag, load_dicom_data, apply_window_level, normalize_array, get_single_image_inference, generate_colored_lime_mask
from scripts.demo_utils import extract_number_from_filename, lime_text, gradcam_text
from scripts.demo_utils import get_lime_mask, lime_predict_fn, normalize_to_255
from scripts.demo_utils import generate_gradcam, overlay_gradcam
from  scripts.process_tree import Processor 
from scripts.cnn.cnn_inference import *
from  scripts.config import *
from scripts.utils import *



#from azure.storage.blob import BlobServiceClient

# connection_string = "your_connection_string"
# container_name = "your_container_name"
# local_file_path = "path/to/your/local/file"

# blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# container_client = blob_service_client.get_container_client(container_name)

st.set_page_config(page_title="Abdominal MRI Series Classifier", layout="wide")

st.title("Abdominal MRI Series Classifier")
st.subheader("Duke AIPI590-XAI Final Project")
st.write("Chad Miller")

model = load_pixel_model('models/best_0606.pth', model_type='DenseNet')

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Match the model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# the place to find the image data
start_folder = "data/"
#start_folder = os.environ.get("SOURCE_DATA_PATH")

# the place to put processed image data
destination_folder = 'data/processed/'
destination_folder = st.sidebar.text_input("Enter destination folder path:", value="")
#destination_folder = os.environ.get("SOURCE_DATA_PATH")

selected_images = None
dicom_df = None
# check for dicom images within the subtree and build selectors for patient, exam, series
if os.path.exists(start_folder) and os.path.isdir(start_folder):
    print('dicom images found in ', start_folder)
    folder = st.sidebar.selectbox("Select a source folder:", os.listdir(start_folder), index=0)
    selected_folder = os.path.join(start_folder, folder)

    #dest_folder = st.sidebar.input("Select a destination folder")

    # if there are dicom images somewhere in the tree
    if os.path.exists(selected_folder) and os.path.isdir(selected_folder):
        dicom_df = load_dicom_data(selected_folder)
        #print(dicom_df)

        if not dicom_df.empty:
            # Select patient
            unique_patients = dicom_df["patient"].drop_duplicates().tolist()
            selected_patient = st.selectbox("Select a patient:", unique_patients, key='patient_selectbox')

            # Select exam for the selected patient
            unique_exams = dicom_df[dicom_df["patient"] == selected_patient]["exam"].drop_duplicates().tolist()
            selected_exam = st.selectbox("Select an exam:", unique_exams, key='exam_selectbox')

            # Select series for the selected exam
            unique_series = dicom_df[
                (dicom_df["patient"] == selected_patient) & (dicom_df["exam"] == selected_exam)
            ]["series"].drop_duplicates().tolist()
            selected_series = st.selectbox("Select a series:", unique_series, key='series_selectbox')

            if not dicom_df.empty:
                # Check if there are labels for the selected exam
                has_labels = dicom_df[dicom_df["exam"] == selected_exam]["label"].notnull().any()

                if has_labels:
                    # Select predicted class for the selected series
                    unique_labels = dicom_df[(dicom_df["patient"] == selected_patient) & (dicom_df["exam"] == selected_exam)]["label"].drop_duplicates().tolist()
                    selected_label = st.selectbox("Select images predicted to be of type:", unique_labels)
                else:
                    st.write("The selected exam has no labels available in the DICOM tags.")
                    selected_label = None

            source_selector = st.radio("Select source:", ["Series", "Predicted Type"])

            if source_selector == 'Series': 

                # Display images for the selected series
                selected_images = dicom_df[
                    (dicom_df["patient"] == selected_patient) &
                    (dicom_df["exam"] == selected_exam) &
                    (dicom_df["series"] == selected_series)]["file_path"].tolist()
            
            elif (source_selector == "Predicted Type") and has_labels:
                selected_images = dicom_df[
                    (dicom_df["patient"] == selected_patient) &
                    (dicom_df["exam"] == selected_exam) &
                    (dicom_df["label"] == selected_label)]["file_path"].tolist()

            st.subheader("Selected Study Images")
            cols = st.columns(4)

            # Sort images within each series by filename
            #selected_images.sort(key=lambda x: os.path.basename(x))
            if selected_images:
                
                # Move the window level and image scroll controls below the image
                # window_center = st.slider("Window Center", min_value=-1024, max_value=1024, value=0, step=1)
                # window_width = st.slider("Window Width", min_value=1, max_value=4096, value=4096, step=1)

                selected_images.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
                image_idx = st.select_slider("View an image", options=range(len(selected_images)), value=0)

                # read in the dicom data for the current images and see if there are labels in the DICOM metadata
                image_path = selected_images[image_idx]
                # Convert to string if necessary
                if not isinstance(image_path, str):
                    image_path = str(image_path)            

                # Check if it's a valid file path
                if os.path.isfile(image_path):
                    print(f"{image_path} is a valid file path.")
                else:
                    print(f"{image_path} is not a valid file path.")
            
                dcm_data = pydicom.dcmread(image_path)
                predicted_type  = check_prediction_tag(dcm_data)

                window_width = st.sidebar.slider("Window Width", min_value=1, max_value=4096, value=2500, step=1)
                window_center = st.sidebar.slider("Window Level", min_value=-1024, max_value=1024, value=0, step=1)
                
                with st.container():
            
                    image_file = selected_images[image_idx]
                    try:
                        dcm_data = pydicom.dcmread(image_file)
                        image = dcm_data.pixel_array
                        image = apply_window_level(image, window_center=window_center, window_width=window_width)
                        image = Image.fromarray(normalize_array(image))  # Scale the values to 0-255 and convert to uint8
                        #image = Image.fromarray(dcm_data.pixel_array)
                        image = image.convert("L")
                        if predicted_type:
                            draw = ImageDraw.Draw(image)
                            text = f"Predicted Type: {predicted_type}"
                            draw.text((10, 10), text, fill="white")  # You can adjust the position (10, 10) as needed
                        
                           
                        else:
                            draw = ImageDraw.Draw(image)
                            text = f'No prediction yet'
                            draw.text((10,10), text, fill='white')
                        st.image(image, caption=os.path.basename(image_file), use_container_width=True)
                    
                    except Exception as e:
                        pass
            
        
            else:
                st.write('No type of this predicted class in the exam.')


            process_images = st.sidebar.button("Process Images")
            if process_images:
                if not destination_folder:
                    destination_folder = start_folder
                processor = Processor(start_folder, destination_folder, model=model, overwrite=True, write_labels=True)

                new_processed_df = processor.pipeline_new_studies()
          
            get_inference = st.button("Get Predicted Class For This Image")
            if get_inference:
                # st.write(image_path)
                predicted_type, predicted_confidence = get_single_image_inference(image_path, model)
                st.write(f'Predicted type: {predicted_type}, confidence score: {predicted_confidence:.2f}')
                
                # dropdown to display the confusion matrix
                with st.expander("View Confusion Matrix"):
                    st.image("assets/FigPixel20230412.png", caption="Confusion Matrix")

            st.write(f'Explainable AI methods that may help to understand the model')
            with st.expander("What is LIME and what does it tell me?"):
                st.markdown(lime_text)

            with st.expander("What is Grad-CAM and what does it tell me?"):
                st.markdown(gradcam_text)

            # Initialize variables for LIME and Grad-CAM images
            lime_colored_mask = None
            gradcam_overlay = None

            # Grad-CAM explanation
            generate_gradcam_btn = st.button("Generate Grad-CAM Explanation (Fast)")
            if generate_gradcam_btn:
                if image_path:
                    try:
                        # Load the DICOM image
                        ds = pydicom.dcmread(image_path)
                        pixel_array = ds.pixel_array
                        normalized_image = normalize_to_255(pixel_array)  # Normalize pixel values to 0-255
                        pil_image = Image.fromarray(normalized_image).convert("RGB")  # Convert to PIL image

                        # Define the target layer
                        target_layer = model.densenet.features.denseblock4  # Layer for the Grad-CAM explanation

                        # Generate the Grad-CAM heatmap
                        cam = generate_gradcam(pil_image, model, target_layer, test_transform)

                        # Overlay Grad-CAM on the original image
                        gradcam_overlay = overlay_gradcam(cam, pil_image)

                        # Display the Grad-CAM heatmap
                        st.image(gradcam_overlay, caption="Grad-CAM Explanation (Red: High Importance, Blue: Low Importance)", use_container_width=True)

                    except Exception as e:
                        st.error(f"Error generating Grad-CAM explanation: {e}")
                else:
                    st.warning("Please select an image.")

            # LIME explanation
            get_lime_explanation = st.button("Generate LIME and Grad-CAM Explanation (Slow)")
                     
            if get_lime_explanation:
                st.write('Generating LIME explanation. This may take approximately 1-2 minutes...')
                if image_path:
                    try:
                        # Load the DICOM image
                        ds = pydicom.dcmread(image_path)
                        pixel_array = ds.pixel_array
                        normalized_image=normalize_to_255(pixel_array)
                        pil_image = Image.fromarray(normalized_image).convert("RGB")
                        
                        # Generate Grad-CAM
                        target_layer = model.densenet.features.denseblock4  # Adjust as needed for your model
                        cam = generate_gradcam(pil_image, model, target_layer, test_transform)
                        gradcam_overlay = overlay_gradcam(cam, pil_image)

                        # Generate the LIME explanation with green and yellow coloring
                        lime_colored_mask = generate_colored_lime_mask(normalized_image, model, lime_predict_fn, test_transform=test_transform)

                        # Display the LIME explanation and Grad-CAM side by side
                        st.subheader("LIME + Grad-CAM Explanations")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(gradcam_overlay, caption="Grad-CAM Explanation (Red: High Importance, Blue: Low Importance)", use_container_width=True)                   
                        with col2:
                            st.image(lime_colored_mask, caption="LIME Explanation (Green: Positive, Red: Negative)", use_container_width=True)
                
                    except Exception as e:
                        st.error(f"Error generating LIME explanation: {e}")
                else:
                    st.warning("Please select an image.")
            
            

            # # Generate Anchors explanation
            # generate_anchors = st.button("Generate Anchors Explanation")
            # if generate_anchors:
            #     if image_path:
            #         try:
            #             # Load the DICOM image
            #             ds = pydicom.dcmread(image_path)
            #             image = ds.pixel_array
            #             #image=normalize_to_255(image)

            #             # Generate the Anchors explanation
            #             explanation = generate_anchor_explanation(
            #                 image=image,
            #                 model=model,
            #                 device=next(model.parameters()).device,
            #                 explainer=anchor_explainer,
            #                 abd_label_dict=abd_label_dict,
            #                 test_transform=test_transform
            #             )

            #             # Visualize the explanation
            #             fig = visualize_anchor_explanation(image, explanation, title=f"Anchor Explanation for {os.path.basename(image_path)}")
            #             st.pyplot(fig)

            #         except Exception as e:
            #             st.error(f"Error generating Anchors explanation: {e}")
            #     else:
            #         st.warning("Please select an image.")

        else:
            st.warning("No DICOM files found in the folder.")
else:
    st.error("Invalid start folder path.")


