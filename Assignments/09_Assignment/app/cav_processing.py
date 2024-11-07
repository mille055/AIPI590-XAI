import torch
import matplotlib.pyplot as plt
import streamlit as st
from transformer_lens import HookedTransformer, utils

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
def generate_cav(model: HookedTransformer, tokenizer, positive_texts: list[str], negative_texts: list[str], layer_name: str):
    # Tokenize positive and negative texts
    positive_tokens = [tokenizer(text, return_tensors="pt")["input_ids"] for text in positive_texts]
    negative_tokens = [tokenizer(text, return_tensors="pt")["input_ids"] for text in negative_texts]

    # Capture activations for positive examples
    positive_activations = [capture_activations(model, tokens, layer_name).mean(dim=1) for tokens in positive_tokens]

    # Capture activations for negative examples
    negative_activations = [capture_activations(model, tokens, layer_name).mean(dim=1) for tokens in negative_tokens]

    # Calculate the CAV as the mean difference between positive and negative activations
    cav = torch.mean(torch.stack(positive_activations), dim=0) - torch.mean(torch.stack(negative_activations), dim=0)
    cav = cav.squeeze()
    cav = cav / torch.norm(cav)
    print('cav:', cav.shape)
    return cav

# Function to calculate average similarity scores between CAV and text activations
def calculate_cav_similarity(model: HookedTransformer, tokenizer, text: str, cav: torch.Tensor, layer_name: str):
    # Tokenize the text to be analyzed
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]

    # Capture activations for the text at the specified layer
    text_activations = capture_activations(model, tokens, layer_name).mean(dim=1)

    # solve the issue of the extra dimension
    text_activations = text_activations.squeeze()

    # Normalize text activations for each token for consistent comparison
    text_activations = text_activations / text_activations.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity between the CAV and the text activations
    similarity_scores = torch.cosine_similarity(text_activations, cav.unsqueeze(0), dim=-1)
    
    average_similarity = similarity_scores.mean().item()
    max_similarity = similarity_scores.max().item()
    print('max similarity:', max_similarity)
    print('similarity:', average_similarity)
    return average_similarity, max_similarity

# Function to calculate similarity scores between CAV and text activations for multiple layers
def calculate_layerwise_cav_similarity(model: HookedTransformer, tokenizer, text: str, cav: torch.Tensor, layers: list[str]):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    similarity_scores = []

    # Calculate similarity for each specified layer
    for layer_name in layers:
        # Capture activations for the text at the specified layer
        text_activations = capture_activations(model, tokens, layer_name).squeeze()

        # Normalize text activations for each token for consistent comparison
        text_activations = text_activations / text_activations.norm(dim=-1, keepdim=True)

        # Calculate token-wise cosine similarities and take the average
        similarity = torch.cosine_similarity(text_activations, cav.unsqueeze(0), dim=-1).mean().item()
        similarity_scores.append(similarity)

    return similarity_scores

# Function to calculate similarity scores between CAV and text activations for multiple layers
def calculate_cav_similarity_multiple_layers(model: HookedTransformer, tokenizer, text: str, cav: torch.Tensor, layers: list[str]):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    similarity_scores = []

    # Calculate similarity for each specified layer
    for layer_name in layers:
        # Capture activations for the text at the specified layer
        text_activations = capture_activations(model, tokens, layer_name).squeeze()

        # Normalize text activations for each token for consistent comparison
        text_activations = text_activations / text_activations.norm(dim=-1, keepdim=True)

        # Calculate token-wise cosine similarities and take the average
        similarity = torch.cosine_similarity(text_activations, cav.unsqueeze(0), dim=-1).mean().item()
        similarity_scores.append(similarity)

    # Calculate the overall average and max similarity across all layers
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    max_similarity = max(similarity_scores)  
    return average_similarity, max_similarity, similarity_scores


# Plotting function for CAV similarity
def plot_cav_similarity(similarity_score, layerwise_similarity, concept):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot single similarity score as a bar
    #ax.bar(["Overall Similarity"], [similarity_score], color='skyblue', label=f"Overall Similarity to '{concept}'")

    # Plot layerwise similarity as a line plot
    ax.plot(range(len(layerwise_similarity)), layerwise_similarity, marker='o', color='salmon', label="Layerwise Similarity")

    # Add axis labels and title
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"CAV Similarity to Concept '{concept}'")
    
    # Add integer layer labels on the x-axis for the line plot
    ax.set_xticks(range(len(layerwise_similarity)))
    ax.set_xticklabels([str(i) for i in range(len(layerwise_similarity))])

    # Add legend
    ax.legend(loc="upper right")

    # Display the plot in Streamlit
    st.pyplot(fig)
