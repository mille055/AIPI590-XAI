# Memory and Context Retention Analysis App Using TransformerLens

This project uses TransformerLens, a powerful library for mechanistic interpretability of transformer models, to analyze memory retention and attention patterns in GPT-2. This Streamlit application provides an interactive environment to explore how the model processes context across layers and attention heads, allowing users to examine how specific tokens in the input are retained or attended to as they pass through the model's layers.


## Overview

This application uses TransofmerLens to visualize the attention patterns in a language model and allows users to observe memory retention across transformer layers. It provides an interactive way to explore specific layers and attention heads, giving insights into which tokens a model focuses on and how context is retained or transformed at different levels.

## What is TransformerLens? 

TransformerLens is a library developed specifically for exploring and interpreting transformer-based language models like GPT-2. TransformerLens provides tools to inspect, analyze, and intervene in different parts of the model by attaching hooks to layers, attention heads, and other components.

* TransformerLens allows attaching hooks at specific points in the model, making it possible to capture and analyze activations at each layer's residual streams and attention patterns.
* With hooks on the attention heads, we can extract attention weights and visualize how the model distributes attention across different tokens in the input sequence at various layers and heads.
* By using intervention hooks, we can zero out or modify specific activations, enabling experiments to study the effect of certain layers or heads on the output.

In this project, TransformerLens is used to capture and visualize residual activations at each layer, showing how information about each token is transformed across layers.


## Prerequisites
Python 3.9+
Streamlit
PyTorch (for running the model)
Plotly
TransformerLens

## Setup
1. Clone the repository:
```
git clone https://github.com/mille055/AIPI590-XAI.git
```
2. Create and activate a virtual environment

First, get into the subdirectory for this app:
```
cd Assignments/09_Assignment 
```
Then

On a Mac:
```
python -m venv venv
source venv/bin/activate
```
On a PC:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
Change the directory to the app and run the app with the following commands:
```
cd app
streamlit run my_app.py
```
# Acknowledgments
The code and ideas borrow from the TensorLens main demo colab notebook:
(https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb)

and the other demo pages:
(https://transformerlensorg.github.io/TransformerLens/content/tutorials.html)

Copilot used for autocomplete in the Visual Studio Code environment for this project.
