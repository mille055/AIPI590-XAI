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
from scripts.process_tree import Processor 

from scripts.config import *
from scripts.utils import *
from scripts.cnn.cnn_inference import *

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries



# Function to check if the image has been processed and return the value in the DICOM tag (0010, 1010)
def check_prediction_tag(dcm_data):
    prediction = None
    if (0x0011, 0x1010) in dcm_data:
        prediction =  abd_label_dict[str(dcm_data[0x0011, 0x1010].value)]['short']  # this gets the numeric label written into the DICOM and converts to text description
        # if there are submodel predictions
        
        
        if (0x0011, 0x1012) in dcm_data:
            substring = dcm_data[0x0011, 0x1012].value
            sublist = substring.split(',')
            try:
                prediction_cnn = abd_label_dict[sublist[1]]['short']
                
            except Exception as e:
                pass
        return prediction
    else:
        return None
    

@st.cache_resource
def load_dicom_data(folder):
    # function for getting the dicom images within the subfolders of the selected root folder
    # assumes the common file structure of folder/patient/exam/series/images
    data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".dcm"):
                try:
                    dcm_file_path = os.path.join(root, file)
                    #print(dcm_file_path)
                    dcm_data = pydicom.dcmread(dcm_file_path)
                    
                    
                    
                    label = check_prediction_tag(dcm_data)
                    
                    data.append(
                        {
                            "patient": dcm_data.PatientName,
                            "exam": dcm_data.StudyDescription,
                            "series": dcm_data.SeriesDescription,
                            "file_path": dcm_file_path,
                            "label": label
                        }
                    )
                except Exception as e:
                    with st.exception("Exception"):
                        st.error(f"Error reading DICOM file {file}: {e}")

    return pd.DataFrame(data)


# for adjusting the W/L of the displayed image
def apply_window_level(image, window_center, window_width):
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    image = np.clip(image, min_value, max_value)
    return image

def normalize_array(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max != arr_min:
        return (arr - arr_min) * 255 / (arr_max - arr_min)
    else:
        return 0
    
def get_single_image_inference(image_path, model):
    '''
    Gets a set of inference predicted class and confidence score for the overall fusion model and for the submodels
    Inputs: 
        image_path(str): path to the image
        model(class):  trained model
    Outputs: 
        predictions (str) and confidence (float) for the various classifiers
   '''
    
   
    img_df = pd.DataFrame.from_dicoms([image_path])
   
    predicted_series_class, predicted_series_confidence = pixel_inference(model, img_df.fname)
    predicted_class_single = predicted_series_class[0]
    predicted_confidence_single=np.max(predicted_series_confidence)
    predicted_class = abd_label_dict[str(predicted_class_single)]['short'] #abd_label_dict[str(predicted_series_class)]['short']
    predicted_confidence = np.round(predicted_confidence_single, 2)
    

    return predicted_class, predicted_confidence


def extract_number_from_filename(filename):
    # Extract numbers from the filename using a regular expression
    numbers = re.findall(r'\d+', os.path.basename(filename))
    if numbers:
        # Return the last number in the list if there are any numbers found
        return int(numbers[-1])
    else:
        # Return -1 if no numbers are found in the filename
        return -1

def lime_predict_fn(images, model):
    """
    Wrapper for LIME to preprocess images and get predictions.

    Args:
        images (ndarray): Batch of images as numpy arrays (H, W, C).
        model (torch.nn.Module): The PyTorch model.

    Returns:
        ndarray: Softmax probabilities for each class.
    """
    device = next(model.parameters()).device  # Ensure device compatibility
    images = torch.tensor(images.transpose(0, 3, 1, 2)).float().to(device)  # Convert to (batch, channels, H, W)

    with torch.no_grad():
        outputs = model(images)

    return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

def get_lime_mask(image, model, lime_predict_fn, test_transform, progress_callback=None):
    """
    Generate a LIME mask for an image.

    Args:
        image (ndarray): The input image.
        model (torch.nn.Module): The trained model.
        lime_predict_fn (Callable): LIME-compatible prediction function.
        test_transform (transforms.Compose): The test transform pipeline.

    Returns:
        ndarray: LIME mask.
    """
    # Initialize the LIME explainer
    explainer = LimeImageExplainer()

    # Define a wrapper for progress updates
    def progress_wrapper(current, total):
        if progress_callback:
            progress_callback(current, total)

    # Preprocess the image for the model
    image_pil = Image.fromarray(image).convert('RGB')  # Ensure RGB
    processed_image = test_transform(image_pil).unsqueeze(0).to(next(model.parameters()).device)

    # Convert processed image back to numpy for LIME
    image_np = processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Generate the explanation
    explanation = explainer.explain_instance(image_np, lambda imgs: lime_predict_fn(imgs, model),
                                             top_labels=1, hide_color=0, num_samples=1000)

    # Get the mask for the top predicted class
    _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

    return mask


def generate_lime_explanation(image, model, predict_fn, num_samples=1000, progress_callback=None):
    """
    Generate a LIME explanation for the given image and model.

    Args:
        image (ndarray): The input image to explain.
        model (Callable): The prediction model.
        predict_fn (Callable): A prediction function that matches LIME's expected input.
        num_samples (int): Number of perturbed samples to generate for LIME.

    Returns:
        ndarray: Image with LIME overlay.
    """
    explainer = LimeImageExplainer()

    # Define a wrapper for progress updates
    def progress_wrapper(current, total):
        if progress_callback:
            progress_callback(current, total)

    explanation = explainer.explain_instance(
        image,  # The image to explain
        predict_fn,  # The model's prediction function
        top_labels=1,  # Number of top labels to explain
        hide_color=0,
        num_samples=num_samples,  # Number of samples to generate
        progress_bar=progress_wrapper
    )

    # Get the explanation for the top predicted label
    top_label = explanation.top_labels[0]
    lime_overlay, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    # Overlay LIME explanation on the image
    lime_overlay = mark_boundaries(lime_overlay, mask)
    return lime_overlay

def generate_colored_lime_mask(image, model, lime_predict_fn, num_samples=1000):
    """
    Generate a LIME mask with green and yellow coloring.

    Args:
        image (ndarray): The input image.
        model (torch.nn.Module): The trained model.
        lime_predict_fn (Callable): LIME-compatible prediction function.
        num_samples (int): Number of perturbed samples.

    Returns:
        ndarray: Image with green and yellow LIME mask.
    """
    explainer = LimeImageExplainer()

    # Generate the explanation
    explanation = explainer.explain_instance(
        image,
        lambda imgs: lime_predict_fn(imgs, model),
        top_labels=1,
        hide_color=0,
        num_samples=num_samples
    )

    # Get the mask with both positive (green) and negative (yellow) contributions
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],  # Focus on the top predicted class
        positive_only=False,       # Include both positive and negative contributions
        num_features=10,           # Number of superpixels to highlight
        hide_rest=False            # Keep the unimportant areas visible in grayscale
    )

    # Overlay the LIME mask on the original image
    lime_overlay = mark_boundaries(temp, mask)

    return lime_overlay


def normalize_to_255(image):
    """
    Normalize image pixel values to the range [0, 255].
    
    Args:
        image (ndarray): Input image array.
    
    Returns:
        ndarray: Normalized image.
    """
    image = image.astype(np.float32)  # Ensure the data type supports decimals
    image -= image.min()  # Shift to make minimum 0
    image /= image.max()  # Scale to make maximum 1
    image *= 255  # Scale to range [0, 255]
    return image.astype(np.uint8)  # Convert to 8-bit integers