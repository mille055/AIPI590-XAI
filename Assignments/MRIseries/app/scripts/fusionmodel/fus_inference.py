import numpy as np
import pandas as pd
import os
import torch
import pickle
import pydicom
from pydicom.errors   import InvalidDicomError
from pathlib import Path

from fus_model import FusionModel
from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer
from utils import *



# Load the models and create an instance of the ModelContainer
model_container_instance = ModelContainer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_fusion_inference(row, model_container, classes=classes, features=feats_to_keep, device=device, include_nlp=True):
    '''
    Gets predictions and probabilities based on the fusion model over the provided dataframe row
    Input: 
        row(pd.Series): row from the dataframe that contains the pixel_array which is used for this model
        model_container(ModelContainer class): holds the submodels, scaler for feature creation and the weights for the fusion model
        classes(int): the list of classes
        features(list): list of features to use for the metadata model
        device(cpu or gpu)
        include_nlp(bool): whether to include all 3 models or just the pixel (cnn) and metadata (RF)

    Output:
        predicted_class(int): int corresopnding to the predicted class
        confidence_score(float): highest probability which corresonds to the predicted class
        submodel_df: contains the preds/probs for the 3 submodels
    '''
    metadata_model = model_container.metadata_model
    cnn_model = model_container.cnn_model
    nlp_model = model_container.nlp_model
    scaler = model_container.metadata_scaler

    # Create FusionModel instance
    fusion_model = FusionModel(model_container=model_container, num_classes=len(classes), include_nlp=include_nlp)
    # Load the weights
    if include_nlp:
        weights_path = model_container.fusion_weights_path
    else:
        weights_path = model_container.partial_fusion_weights_path
        
    fusion_model.load_weights(weights_path)



    # get metadata preds,probs
    pred1, prob1 = get_meta_inference(row, scaler, metadata_model, features)
    prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()
    print(pred1)

    # get cnn preds, probs
    pred2, prob2 = pixel_inference(cnn_model, row['fname'].values.tolist()[0], classes=classes)
    prob2_tensor = torch.tensor(prob2, dtype=torch.float32)
    print(pred2)

    # get nlp preds, probs...if statement because thinking about assessing both ways
    if include_nlp:
        pred3, prob3 = get_NLP_inference(nlp_model, row['fname'].values.tolist()[0], device, classes=classes)
        prob3_tensor = torch.tensor(prob3, dtype=torch.float32)
        print(pred3)
        fused_output = fusion_model(prob1_tensor, prob2_tensor, prob3_tensor)
    else:
        fused_output = fusion_model(prob1_tensor, prob2_tensor)

    predicted_class = classes[torch.argmax(fused_output, dim=1).item()]
    confidence_score = torch.max(torch.softmax(fused_output, dim=1)).item()

    submodel_df = pd.DataFrame.from_dict({'meta_preds': pred1, 'meta_probs': prob1, 'pixel_preds': pred2, 'pixel_probs': prob2, 'nlp_preds': pred3, 'nlp_probs': prob3, 'SeriesD': row.SeriesDescription})

    return predicted_class, confidence_score, submodel_df

# def get_fusion_inference(self, row, classes=classes, features=feats_to_keep, device=device, include_nlp=True):
def get_fusion_inference_from_file(file_path, model_container, classes=classes, features=feats_to_keep, device=device, include_nlp=True):
    '''
    similar to above but gets the inference from a filename and constructs the row first'''
    
    # unpack the models
    metadata_model = model_container.metadata_model
    cnn_model = model_container.cnn_model
    nlp_model = model_container.nlp_model
    scaler = model_container.metadata_scaler
   
   # Create FusionModel instance
    fusion_model = FusionModel(model_container=model_container, num_classes=19)
    # Load the weights
    fusion_model.load_weights(model_container.fusion_weights_path)

    
    my_df = pd.DataFrame.from_dicoms([file_path])

    # Preprocess the metadata using the preprocess function
    preprocessed_metadata, _ = preprocess(my_df, scaler=model_container.metadata_scaler)
    
    # Get the preprocessed row
    row = preprocessed_metadata.iloc[0]

    # Call the original get_fusion_inference function
    predicted_class, confidence_score, troubleshoot_df = get_fusion_inference(row, model_container, classes, features, device, include_nlp)

    return predicted_class, confidence_score, troubleshoot_df



def load_fusion_model(model_path):
    with open(model_path, 'rb') as file:
        fusion_model = pickle.load(file)
    return fusion_model
