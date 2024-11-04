import torch
import numpy as np
import plotly.graph_objects as go
from transformer_lens import HookedTransformer, utils

# Load GPT-2 model using TransformerLens
def load_model():
    return HookedTransformer.from_pretrained("gpt2")

# Hook function to capture activations
def hook_fn(value, hook):
    if hook.name not in activations:
        activations[hook.name] = []
    activations[hook.name].append(value.detach().cpu().numpy())
    #print('value is:', value)
    return value

# Function to capture activations using `run_with_hooks`
def capture_activations(model, tokens):
    global activations
    activations = {}

    # Define hooks for each transformer block layer
    fwd_hooks = [
        (utils.get_act_name("resid_post", layer), hook_fn) 
        for layer in range(model.cfg.n_layers)
    ]
    print('fwd_hooks:', fwd_hooks)
    # Run the model with hooks
    with torch.no_grad():
        model.run_with_hooks(tokens, return_type="loss", fwd_hooks=fwd_hooks)

    # Convert activations to numpy arrays for further processing
    for layer in activations:
        activations[layer] = np.array(activations[layer])
    for key, value in activations.items():
        print(f"{key} shape: {value.shape}")
    return activations

def capture_attention_weights(model, tokens):
    attention_weights = {}

    # Define hook function to capture attention patterns
    def attention_hook(value, hook):
        if hook.name not in attention_weights:
            attention_weights[hook.name] = []
        # Capture attention weights (shape: batch x heads x query_len x key_len)
        attention_weights[hook.name].append(value.detach().cpu().numpy())

    # Register hooks for each transformer block's attention output
    fwd_hooks = [
        (utils.get_act_name("attn", layer), attention_hook)
        for layer in range(model.cfg.n_layers)
    ]

    # Run the model with hooks to capture attention weights
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    # Convert to numpy arrays for easier processing
    for layer in attention_weights:
        attention_weights[layer] = np.array(attention_weights[layer])

    return attention_weights


# Function to generate Plotly visualization
def plot_memory_retention(activations, token_index=0):
    
    layer_activations = [activations[f"blocks.{i}.hook_resid_post"][0, 0, token_index] for i in range(len(activations))]
    layer_numbers = list(range(len(layer_activations)))

    # Plot activation strength across layers
    fig = go.Figure(data=go.Scatter(
        x=layer_numbers,
        y=[np.linalg.norm(layer) for layer in layer_activations],
        mode='lines+markers'
    ))
    fig.update_layout(title="Memory Retention Across Layers",
                      xaxis_title="Layer",
                      yaxis_title="Activation Strength (L2 Norm)")
    return fig