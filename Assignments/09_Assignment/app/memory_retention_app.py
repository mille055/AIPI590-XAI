import streamlit as st
import plotly.express as px
import torch
from activation_processing import load_model, capture_activations, plot_memory_retention, capture_attention_weights
from cav_processing import generate_cav, calculate_cav_similarity, calculate_layerwise_cav_similarity

# Load the model
model = load_model()
tokenizer = model.tokenizer

# Pre-selected text options
pre_selected_texts = {
    "Case Study 1: Respiratory Symptoms": "A patient with a history of chronic cough presents with new symptoms of difficulty breathing, particularly during exercise. No prior history of asthma is reported.",
    "Case Study 2: Cardiac Symptoms": "The patient complains of chest pain that occurs mostly during physical exertion and is relieved by rest. There is a family history of heart disease.",
    "General Medical Report": "The patient is a 45-year-old male presenting with fatigue, weight loss, and night sweats. Lab results show elevated white blood cell counts.",
    "Shakespearean Sonnet": "Shall I compare thee to a summer’s day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May, And summer’s lease hath all too short a date. Sometime too hot the eye of heaven shines, And often is his gold complexion dimmed; And every fair from fair sometime declines, By chance or nature’s changing course untrimmed. But thy eternal summer shall not fade Nor lose possession of that fair thou ow’st, Nor shall Death brag thou wand’rest in his shade, When in eternal lines to time thou grow’st. So long as men can breathe or eyes can see, So long lives this, and this gives life to thee"
}

# Define positive and negative examples for CAV concepts
concept_examples = {
    "Respiratory Distress": {
        "positive": ["The patient is experiencing difficulty breathing.", "Respiratory rate is elevated."],
        "negative": ["The patient is resting comfortably.", "There is no evidence of respiratory issues."]
    },
    "Cardiac Symptoms": {
        "positive": ["The patient complains of chest pain during exertion.", "There is a family history of heart disease."],
        "negative": ["The patient reports no cardiovascular symptoms.", "The patient has a normal heart rate."]
    }
}

# Generate CAVs for the specified concepts
cavs = {}
layer_name = "blocks.10.hook_resid_post"  
for concept, examples in concept_examples.items():
    cavs[concept] = generate_cav(model, tokenizer, examples["positive"], examples["negative"], layer_name)

# Streamlit UI
st.title("Memory and Context Retention Analysis")

# Analysis Type Selection
analysis_type = st.radio("Choose an Analysis Type:", ["Memory Retention", "Concept Analysis (CAV)"])

# Display options based on analysis type
if analysis_type == "Memory Retention":


    # Dropdown for pre-selected texts
    selected_text = st.selectbox("Choose a pre-selected text case or enter custom text:", 
                             ["-- None --"] + list(pre_selected_texts.keys()))

    # Display selected text or allow for custom text entry
    if selected_text == "-- None --":
        input_text = st.text_area("Enter custom text for analysis", "Type your own text here...")
    else:
        input_text = pre_selected_texts[selected_text]
        st.write("Selected Text:", input_text)

    if st.button("Analyze Memory Retention"):
        # Tokenize input text
        tokens = model.tokenizer(input_text, return_tensors='pt')['input_ids']
    
        # Capture activations
        activations = capture_activations(model, tokens)
    
        # Select a token index to visualize 
        token_index = len(tokens[0]) // 2
        fig = plot_memory_retention(activations, token_index)
    
        # Display Plotly chart in Streamlit
        st.plotly_chart(fig)

        # Capture attention weights and store in session state
        st.session_state["attention_weights"] = capture_attention_weights(model, tokens)
        st.session_state["tokens"] = tokens  

    if "attention_weights" in st.session_state and "tokens" in st.session_state:
        # Select layer and head for visualization
        selected_layer = st.slider("Select Layer", 0, model.cfg.n_layers - 1, 0)
        selected_head = st.slider("Select Attention Head", 0, model.cfg.n_heads - 1, 0)

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
        
        # Generate the attention plot based on current slider positions
        attention_fig = plot_attention_pattern(st.session_state["attention_weights"], selected_layer, selected_head)
        st.plotly_chart(attention_fig)

    # cavs = {}
    # for concept, examples in concept_examples.items():
    #     cavs[concept] = generate_layerwise_cavs(model, tokenizer, examples["positive"], examples["negative"], layers_to_analyze)


if analysis_type == "Concept Analysis (CAV)":
    # Concept selection
    concept = st.selectbox("Select a Concept for Analysis", list(cavs.keys()))
    
    # Text input for user to provide their own text
    input_text = st.text_area("Enter text for comparison with selected concept", "Type here...")
    # Define layers to analyze for layerwise similarity
    layers_to_analyze = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    print('layers_to_analyze:', layers_to_analyze)

    if st.button("Calculate CAV Similarity"):
        # Calculate similarity between input text and the selected concept's CAV
        similarity_score = calculate_cav_similarity(model, tokenizer, input_text, cavs[concept], layer_name)
        layerwise_similarity = calculate_layerwise_cav_similarity(model, tokenizer, input_text, cavs[concept], layers_to_analyze)
        
        # Display the similarity score
        st.write(f"Similarity to '{concept}': {similarity_score:.2f}")

        # Display bar chart
        #st.bar_chart([similarity_score])
        st.line_chart(layerwise_similarity, width=700, height=400)