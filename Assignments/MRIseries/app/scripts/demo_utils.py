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
from skimage.transform import resize
from alibi.explainers import AnchorImage
from alibi.explainers import CounterFactual


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

def generate_colored_lime_mask(image, model, lime_predict_fn, test_transform, num_samples=1000):
    """
    Generate a LIME mask with green and yellow coloring.

    Args:
        image (ndarray): The input image.
        model (torch.nn.Module): The trained model.
        lime_predict_fn (Callable): LIME-compatible prediction function.
        num_samples (int): Number of perturbed samples.
        test_transform (transforms.Compose): The test transform pipeline.

    Returns:
        ndarray: Image with green and yellow LIME mask.
    """
    explainer = LimeImageExplainer()

    # Preprocess the image for model inference
    image_pil = Image.fromarray(image).convert('RGB')  # Ensure RGB
    processed_image = test_transform(image_pil).unsqueeze(0).to(next(model.parameters()).device)
    image_np = processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    print(f"image_np shape: {image_np.shape}, min: {image_np.min()}, max: {image_np.max()}")
    print(f"image_np dtype: {image_np.dtype}")

    # Generate the explanation
    explanation = explainer.explain_instance(
        image_np,
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


lime_text = """
                    ### Understanding LIME (Local Interpretable Model-Agnostic Explanations)
                    **LIME** is a technique used to interpret the predictions of machine learning models. It works by:
                    
                    1. **Perturbing the Input Image**: LIME creates several versions of the input image by modifying small parts (called superpixels) of the image.
                    2. **Measuring the Model's Response**: For each perturbed version, LIME measures how much the model's prediction changes.
                    3. **Identifying Important Regions**: Based on the model's response, LIME identifies the regions of the image that positively or negatively contributed to the prediction.

                    ### What Do the Colors Mean?
                    - **Green Regions**: These areas **support** the model's prediction. They have a positive influence on the model's confidence for the predicted class.
                    - **Yellow Regions**: These areas **contradict** the model's prediction. They have a negative influence on the model's confidence for the predicted class.
                    - **Uncolored Regions**: These areas have little or no impact on the model's decision.

                    ### How to Interpret LIME Output
                    The LIME explanation highlights the regions of the image that were most influential in the model's decision. This helps users:
                    - Understand which features (e.g., anatomical regions in medical images) the model is focusing on.
                    - Evaluate whether the model's prediction aligns with expert knowledge.
                    - Detect potential biases or errors in the model.

                    For example:
                    - In an abdominal MRI, **green regions** might highlight a organ that supports the model's classification, while **yellow regions** could indicate artifacts that contradict it.

                    ### Advantages of LIME Over Anchors or SHAP:
                    - While Anchors provides rule-based explanations (e.g., "If this set of pixels is present, the model will always predict X"), LIME provides more detailed, graded explanations, showing how much each superpixel contributes to the final decision. This can be more insightful for image tasks where specific regions matter more than discrete "rules."
                    - SHAP is generally slower than LIME, especially when applied to deep neural networks like DenseNet. SHAP calculates Shapley values, which can be computationally expensive because it considers all feature combinations. LIME, in contrast, is faster and less computationally intensive because it approximates the behavior of the model locally.
                    - SHAP can give very precise pixel-level attributions, but these can be harder to interpret visually for non-technical users. LIME’s focus on superpixels gives you higher-level insights that are visually easier to comprehend.
                    
                    ### Limitations of LIME:
                    - Local explanations may not generalize.
                    - LIME creates a local surrogate model to explain the prediction, but these explanations are approximate and can vary with different perturbations. This means the results might not always be consistent across different runs, especially for images that are borderline cases.
                    - LIME provides explanations for individual predictions, but it does not give you a global understanding of how the model behaves across the entire dataset, unlike SHAP, which can offer both global and local explanations.
                    - Since LIME uses a linear surrogate model to approximate the complex decision boundary of DenseNet, it might oversimplify the true decision-making process of the neural network, especially for high-dimensional, non-linear image data.
                    - LIME is slower than anchors for large images, and may require many perturbations to approximate the model’s behavior for a given image. For high-resolution images or large test sets, this can be computationally expensive and time-consuming, although it is typically faster than SHAP.
            """