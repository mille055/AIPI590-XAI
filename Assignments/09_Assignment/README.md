# Attention Analysis and Concept Activation Vector (CAV) Analysis Using TransformerLens

This project uses TransformerLens, a powerful library for mechanistic interpretability of transformer models, to analyze memory retention and attention patterns in GPT-2. This Streamlit application provides an interactive environment to explore how the model processes context across layers and attention heads, allowing users to examine how specific tokens in the input are retained or attended to as they pass through the model's layers.


## Overview

This application provides two main analyses:

### Attention Pattern Analysis: 
Visualizes how GPT-2 attends to different tokens across layers and attention heads. This helps users understand which parts of the input the model focuses on when processing text and how attention shifts across layers.

### Concept Activation Vector (CAV) Analysis: 
Measures how well the model’s internal representations align with predefined concepts, such as "Respiratory Distress" or "Cardiac Symptoms." This analysis aims to quantify the model’s understanding of specific concepts, allowing users to evaluate the depth of concept representation in different layers.


## What is TransformerLens? 

TransformerLens is a library developed specifically for exploring and interpreting transformer-based language models like GPT-2. TransformerLens provides tools to inspect, analyze, and intervene in different parts of the model by attaching hooks to layers, attention heads, and other components.

* TransformerLens allows attaching hooks at specific points in the model, making it possible to capture and analyze activations at each layer's residual streams and attention patterns.
* With hooks on the attention heads, we can extract attention weights and visualize how the model distributes attention across different tokens in the input sequence at various layers and heads.
* By using intervention hooks, we can zero out or modify specific activations, enabling experiments to study the effect of certain layers or heads on the output.

In this project, TransformerLens is used to capture and visualize residual activations at each layer, showing how information about each token is transformed across layers.

## Application Details
### Attention Analysis
In attention analysis, the application allows users to:

1. Select a layer and head to observe token-level attention distributions.
2. Visualize attention patterns with an interactive heatmap, showing the importance of each token relative to others.
3. Explore how the model’s focus changes across different layers and attention heads, revealing where the model captures short- or long-term dependencies.

### Concept Analysis with CAVs
In CAV analysis, the application allows users to:

!. Define a concept with positive and negative examples, creating a contrastive concept vector.
2. Input a sentence and calculate similarity scores between the model's activations and the concept vector at each layer, providing insights into how well the model aligns with the concept across layers.

3. Visualize these scores across layers, helping users understand which layers best capture the concept and identify potential areas for improvement in model training or interpretability.

Potential use cases include examining model bias terms or comparing attention patterns across heads and layers for exploratory analysis.  

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
https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb

and the other demo pages:
https://transformerlensorg.github.io/TransformerLens/content/tutorials.html

Autocomplete with copilot in the Visual Studio Code environment was in use for this project.
