import torch
from transformer_lens import HookedTransformer, utils
from typing import List, Dict

# Function to capture activations for a given text
def capture_activations(model: HookedTransformer, tokens, layer_name: str):
    activations = {}

    def hook_fnn(activation, hook):
        activations[hook.name] = activation.detach().clone()

    # Add hook for the specified layer
    model.add_hook(layer_name, hook_fnn)

    # Run model on tokens to capture activations
    model(tokens)

    # Remove the hook after capturing
    model.reset_hooks()

    return activations[layer_name]

# Function to generate a CAV from positive and negative examples
def generate_cav(model: HookedTransformer, tokenizer, positive_texts: List[str], negative_texts: List[str], layer_name: str):
    # Tokenize positive and negative texts
    positive_tokens = [tokenizer(text, return_tensors="pt")["input_ids"] for text in positive_texts]
    negative_tokens = [tokenizer(text, return_tensors="pt")["input_ids"] for text in negative_texts]

    # Capture activations for positive examples
    positive_activations = [capture_activations(model, tokens, layer_name).mean(dim=1) for tokens in positive_tokens]

    # Capture activations for negative examples
    negative_activations = [capture_activations(model, tokens, layer_name).mean(dim=1) for tokens in negative_tokens]

    # Calculate the CAV as the mean difference between positive and negative activations
    cav = torch.mean(torch.stack(positive_activations), dim=0) - torch.mean(torch.stack(negative_activations), dim=0)
    return cav

# Function to calculate average similarity scores between CAV and text activations
def calculate_cav_similarity(model: HookedTransformer, tokenizer, text: str, cav: torch.Tensor, layer_name: str):
    # Tokenize the text to be analyzed
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]

    # Capture activations for the text at the specified layer
    text_activations = capture_activations(model, tokens, layer_name).mean(dim=1)

    # solve the issue of the extra dimension
    text_activations = text_activations.squeeze()
    print('text_activations:', text_activations.shape)
    cav = cav.squeeze()
    print('cav:', cav.shape)

    # Calculate the cosine similarity between the CAV and the text activations
    similarity = torch.cosine_similarity(text_activations, cav, dim=0).item()
    return similarity

# Function to calculate similarity scores between CAV and text activations for multiple layers
def calculate_layerwise_cav_similarity(model: HookedTransformer, tokenizer, text: str, cavs: dict, layers: List[str]):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    similarity_scores = []

    # Calculate similarity for each specified layer
    for layer_name in layers:
        text_activations = capture_activations(model, tokens, layer_name).mean(dim=1).squeeze()  # Average across tokens
        cav = cavs[layer_name].squeeze()  # Ensure CAV is also a 1D tensor
        similarity = torch.cosine_similarity(text_activations, cav, dim=0).item()
        similarity_scores.append(similarity)

    return similarity_scores