import streamlit as st
import plotly.express as px
import torch
import circuitsvis as cv
from circuitsvis.attention import attention_heads
from activation_processing import load_model, capture_activations, plot_memory_retention, capture_attention_weights
from cav_processing import generate_cav, calculate_cav_similarity, calculate_layerwise_cav_similarity, plot_cav_similarity, calculate_cav_similarity_multiple_layers

# Load the model
model = load_model()
tokenizer = model.tokenizer

MAX_TOKEN_LENGTH = 50

# Pre-selected text options
pre_selected_texts = {
    "Case Study 1: Respiratory Symptoms": "A patient with a history of chronic cough presents with new symptoms of difficulty breathing, particularly during exercise. No prior history of asthma is reported.",
    "Case Study 2: Chest CT Report": "There is subsegmental pulmonary embolism in the right lower lobe. No evidence of right heart strain. No pleural effusion.",
    "Case Study 3: Abdominal MRI Report": "MRI shows a 2 cm mass in the left hepatic lobe. The lesion is T2 hyperintense and has discontinuous peripheral nodular enhancement on post contrast imaging.", 
    "Case Study 4: General Medical Report": "The patient is a 45-year-old male presenting with fatigue, weight loss, and night sweats. Lab results show elevated white blood cell counts.",
    "Case Study 5: Shakespearean Sonnet": "Shall I compare thee to a summer’s day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May, And summer’s lease hath all too short a date. Sometime too hot the eye of heaven shines, And often is his gold complexion dimmed; And every fair from fair sometime declines, By chance or nature’s changing course untrimmed. But thy eternal summer shall not fade Nor lose possession of that fair thou ow’st, Nor shall Death brag thou wand’rest in his shade, When in eternal lines to time thou grow’st. So long as men can breathe or eyes can see, So long lives this, and this gives life to thee"
}

# Define positive and negative examples for CAV concepts
concept_examples = {
    "Respiratory Distress": {
        "positive": ["The patient is experiencing difficulty breathing.", "Respiratory rate is elevated.", "The oxygen saturation is low."],
        "negative": ["The patient is resting comfortably.", "The patient appears well.", "The patient has normal lung sounds."]
    },
    "Cardiac Symptoms": {
        "positive": ["The patient complains of chest pain during exertion.", "There is a family history of heart disease."],
        "negative": ["The patient reports no cardiovascular symptoms.", "The patient has a normal heart rate."]
    },
    "Liver Mass": {
        "positive": ["The MRI shows a mass in the cirrhotic liver.", "The lesion is mildly T2 hyperintense, hyperenhances on the arterial phase, and washes out on the equilibrium phase.", "The mass can be classified as LI-RADS 5 for hepatocellular carcinoma."],
        "negative": ["The liver is normal.", "No masses are detected.", "There is no evidence of liver disease."]
    }
}

# Generate CAVs for the specified concepts
cavs = {}
layer_name = "blocks.10.hook_resid_post"  
for concept, examples in concept_examples.items():
    cavs[concept] = generate_cav(model, tokenizer, examples["positive"], examples["negative"], layer_name)

def get_model_cache(model, text):
    # Tokenize the input text
    tokens = model.to_tokens(text)
    
    # Run the model with caching enabled
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    #print("Available keys in cache:", cache.keys())
    
    return logits, cache, tokens

# Extract and plot attention pattern for the selected layer and head
def plot_attention_pattern(attention_weights, selected_layer, selected_head):
    # Extract the single attention head pattern
    attention_all_heads = attention_weights[f"blocks.{selected_layer}.attn.hook_pattern"][0].squeeze()  # Should be (12, 28, 28)
    attention = attention_all_heads[selected_head]  # Shape should be (28, 28)

    # Get tokenized input text for labels
    #input_tokens = model.tokenizer.convert_ids_to_tokens(st.session_state["tokens"][0].tolist())
    input_tokens = [model.tokenizer.decode([token]).strip() for token in st.session_state["tokens"][0].tolist()]

    # For sizing the heatmap
    num_tokens = len(input_tokens)
    figure_size = max(300, 25* num_tokens)
            

    # Plot attention heatmap
    fig = px.imshow(
                attention,
                x=input_tokens,
                y=input_tokens,
                color_continuous_scale="Blues",
                labels={"x": "Key Tokens", "y": "Query Tokens", "color": "Attention Weight"},
                title=f"Attention Pattern in Layer {selected_layer}, Head {selected_head}"
                #width=figure_size,
                #height=figure_size
            )
    fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(len(input_tokens))), ticktext=input_tokens))
    fig.update_layout(yaxis=dict(tickmode="array", tickvals=list(range(len(input_tokens))), ticktext=input_tokens))
    return fig


# Function to visualize attention patterns using CircuitsVis
def visualize_attention_patterns(cache, model, tokens, layer_index):
    # Access attention pattern for the specified layer
    attention_pattern = cache["blocks." + str(layer_index) + ".attn.hook_pattern"]
    
    # Convert token IDs to strings for visualization
    token_strings = model.to_str_tokens(tokens)
    
    # Use circuitsvis to visualize the attention pattern
    #attention_html = cv.attention.attention_patterns(tokens=token_strings, attention=attention_pattern)
    attention_html_obj = cv.attention.attention_patterns(tokens=token_strings, attention=attention_pattern)
    attention_html = attention_html_obj.to_html() if hasattr(attention_html_obj, 'to_html') else str(attention_html_obj)

    # Render the HTML component in Streamlit
    st.components.v1.html(attention_html, height=600, scrolling=True)

# Streamlit UI
st.markdown("<style>body { background-color: white; }</style>", unsafe_allow_html=True)
st.title("Transformer Attention and Concept Analysis Using TransformerLens")
s
st.write("This project uses TransformerLens, a powerful library for mechanistic interpretability of transformer models, to analyze attention patterns in GPT-2 for those with a basic understanding. This Streamlit application provides an interactive environment to explore how the model processes context across layers and attention heads, allowing users to examine how specific tokens in the input are retained or attended to as they pass through the model's layers.")
st.write("Transformer models, like GPT-2, adjust their focus on words in a sentence based on context and task. By visualizing attention patterns, we can see which words the model prioritizes during text interpretation or prediction. \
Transformers have multiple layers and attention heads. Each layer refines focus across the input, while each head captures different aspects of the text. GPT-2’s causal attention restricts tokens to attend only to previous tokens, creating a sequential flow of information essential for predictive tasks.")

st.write("This application also includes Concept Analysis (CAV) to measure how well the model's internal representations align with a given concept. By comparing the model's activations to a concept's CAV, we can quantify the model's understanding of the concept. This can help identify biases or areas where the model may need further training or refinement.")
st.write("Note: Please use light mode or the custom theme for the best viewing experience.")

# Analysis Type Selection
analysis_type = st.radio("Choose an Analysis Type:", ["Attention Patterns: Select this to view how the model attends to each token in the text at various layers and heads.", "Concept Analysis (CAV): Click this button to compute the similarity score for the entered text, visualizing how the model's internal representations match the preselected concept."])

# Display options based on analysis type
if analysis_type.startswith("Attention Patterns"):


    # Dropdown for pre-selected texts
    selected_text = st.selectbox("Choose a pre-selected text case or enter custom text to explore how the model attends to each token:",
                             ["-- None --"] + list(pre_selected_texts.keys()))

    # Display selected text or allow for custom text entry
    if selected_text == "-- None --":
        input_text = st.text_area("Enter custom text for analysis.", "Type your own text here...")
    else:
        input_text = pre_selected_texts[selected_text]
        st.write("Selected Text:", input_text)

    if st.button("Analyze Attention Patterns"):
        # Tokenize input text
        tokens = model.tokenizer(input_text, return_tensors='pt')['input_ids']
        if tokens.shape[1] > MAX_TOKEN_LENGTH:
            st.warning("The input text is too long. It has been truncated for analysis.")
            tokens = tokens[:, :MAX_TOKEN_LENGTH]

        # Capture activations
        activations = capture_activations(model, tokens)
    
        # Capture attention weights and store in session state
        st.session_state["attention_weights"] = capture_attention_weights(model, tokens)
        st.session_state["tokens"] = tokens  

    if "attention_weights" in st.session_state and "tokens" in st.session_state:
        # Select layer and head for visualization
        selected_layer = st.slider("Select Layer", 0, model.cfg.n_layers - 1, 0)
        selected_head = st.slider("Select Attention Head", 0, model.cfg.n_heads - 1, 0)
   
        # Toggle between custom Plotly and CircuitsVis visualizations
        visualization_type = st.radio("Choose Visualization Method:", ["Custom Plotly: Custom visualization showing the attenuation weights for the key and query tokens for the selected layer and head.", "CircuitsVis: Interactive visualization of attention patterns using CircuitsVis."])

        if visualization_type.startswith("Custom Plotly"):
            # Generate the attention plot based on current slider positions
            attention_fig = plot_attention_pattern(st.session_state["attention_weights"], selected_layer, selected_head)
            st.plotly_chart(attention_fig)

        if visualization_type.startswith("CircuitsVis"):
            # Display attention patterns using CircuitsVis
            logits, cache, tokens = get_model_cache(model, input_text)
            visualize_attention_patterns(cache, model, tokens, selected_layer)

        st.subheader("Interpreting Attention Analysis Results")
        st.write("""
        In this section, you are visualizing the attention patterns of the selected transformer model. The attention heads and layers each serve different roles in interpreting the input text:
        - **Attention Head**: Each head in a layer has the ability to focus on different parts of the text, highlighting specific words or tokens. Selecting different heads can reveal which parts of the input are being emphasized by the model at each step.
        - **Layer**: Each layer in the transformer progressively refines the model's understanding of the input. Lower layers typically capture basic patterns (like word associations), while higher layers capture more abstract, context-aware relationships. 
        - **Interpretation**: By examining the attention heatmap, you can see which tokens are prioritized when the model processes the input. A darker color represents higher attention weights, indicating greater focus on that token. The attention patterns often help reveal dependencies, such as how nouns and verbs relate or how earlier and later tokens in a sequence influence each other.
        
        Use this visualization to explore how attention changes across layers and heads, allowing insights into the model's contextual understanding and focus on specific terms or ideas in your text.
        """)

if analysis_type.startswith("Concept Analysis (CAV)"):
    # Concept selection
    concept = st.selectbox("Select a Concept for Analysis", list(cavs.keys()))
    positive_texts = concept_examples[concept]["positive"]
    st.write(f"Selected positive examples for '{concept}':")
    for text in positive_texts:
        st.write(f"- {text}")

    # Text input for user to provide their own text
    input_text = st.text_area("Enter text for comparison with selected concept", "Type here...")
    
    

    if st.button("Calculate CAV Similarity"):
       # Define layers to analyze for layerwise similarity
        layers_to_analyze = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
        print('layers_to_analyze:', layers_to_analyze)
        
        # Calculate similarity between input text and the selected concept's CAV
        avg_similarity_score, max_similarity_score, layerwise_similarity_scores = calculate_cav_similarity_multiple_layers(
        model, tokenizer, input_text, cavs[concept], layers_to_analyze
    )
        
        # Display the similarity score
        st.write(f"Overall average similarity to '{concept}': {avg_similarity_score:.3f}")
        st.write(f"Maximum similarity to '{concept}': {max_similarity_score:.3f}")

        # Display bar chart
        plot_cav_similarity(max_similarity_score, layerwise_similarity_scores, concept)

        st.subheader("Interpreting Concept Analysis (CAV) Results")
        st.write("""
        In this section, you are assessing how well the model aligns with a specific concept, such as "Respiratory Distress" or "Cardiac Symptoms." This analysis uses **Concept Activation Vectors (CAVs)** to capture the essence of a concept based on positive and negative examples. Here’s how to interpret the results:
        - **Overall Average Similarity Score**: This score represents the average alignment between the model's activations and the concept across all layers. Higher values indicate stronger alignment, suggesting that the model understands or captures the concept well across the entire network.
        - **Maximum Similarity Score**: This score represents the highest layer-specific alignment with the concept. A high maximum score may indicate that certain layers specialize more in representing the concept.
        - **Layerwise Similarity Plot**: The line plot shows the similarity score at each layer, revealing which layers align most with the concept. This can help you identify where in the model’s hierarchy the concept is best represented. For example, if a concept is strongly aligned in higher layers, it may suggest that the model relies on more abstract representations to capture it.

        Together, these scores help you evaluate how well the model understands a concept and at which layers this understanding is strongest.
        """)