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

from scripts.process_tree import Processor 

from scripts.config import *
from scripts.utils import *
from scripts.cnn.cnn_inference import *


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
