# This is Python code -> using Streamlit to build a web page

# Import required packages
import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import get_response, format_output, keyword_extract,split_heading_blocks,generate_image_from_text,summary_output_for_images,compute_clip_score
from streamlit.components.v1 import html
import re
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import nltk
import os



#Load Clip model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set up the sidebar
with st.sidebar:
    api_key = st.text_input('Please input openai_API_Key', type='password')

    # Add connection hint using Markdown
    st.markdown('[get openai_Api_key](https://openai.com/api/)')

# Set the title
st.title("Aphasia-friendly content generator")

# Create session memory to store chat history
if "memory" not in st.session_state:
    # Create session memory buffer
    st.session_state['memory'] = ConversationBufferMemory()
    st.session_state['messages'] = [{'role': 'AI', 'content': 'Hello, I am your AI assistant. How can I help you?'}]

# Display the chat history from memory
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Create an input box to receive user input
prompt = st.chat_input('please input content')

# If the user input is not empty, display it
if prompt:
    if not api_key:
        st.warning('Please input API KEY')
        st.stop()

    # Append the user message to the session memory and display it
    st.session_state['messages'].append({'role': 'human', 'content': prompt})
    st.chat_message('human').write(prompt)

    # Show spinner while waiting for AI response
    with st.spinner("Please waiting for reply"):
        # Call custom function to get AI response
        content = get_response(prompt, st.session_state['memory'], api_key)

    # Extract keywords from the response content
    KeyWord = keyword_extract(content)

    # Append the AI response to session memory and display it
    st.session_state['messages'].append({'role': 'ai', 'content': content})

    # Format and render the AI response using custom HTML

    formatted_html = format_output(content)
    html(formatted_html, height=800, scrolling=True)

    #  ADD IMAGE GENERATION HERE
    blocks = split_heading_blocks(content)
    #  Show each heading with its image and simplified text
    if len(KeyWord) == len(blocks):
        st.markdown("### 🖼️ Visual Support")

        for i in range(len(blocks)):
            block = blocks[i]
            keyword_info = KeyWord[i]

            visual_words = keyword_info.get("visual_keywords", [])
            caption_text = "Keywords: " + ", ".join(visual_words).capitalize() if visual_words else "Image"

            try:
                summmary = summary_output_for_images(block['prompt'],api_key)
                image_url = generate_image_from_text(summmary, api_key)


                # # Left-right layout
                col1, col2 = st.columns([1, 1.2])
                with col1:

                    st.markdown(
                        f"""
                        <div style="font-family: Arial, sans-serif; font-size: 18pt; font-weight: bold; color: #0A66C2; margin-bottom: 0.8em;">
                            {block['heading']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Split and display lines that start with a dash ('-')
                    lines = block['content'].split("- ")
                    for line in lines:
                        clean = line.strip()
                        if clean:
                            st.markdown(
                                f"""
                                <div style="font-family: Arial, sans-serif; font-size: 16pt; line-height: 1.8; color: #000000; max-width: 480px; margin-bottom: 10px;">
                                    • {clean}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                with col2:
                    st.image(image_url, caption=caption_text, use_column_width=True)

                    clip_score = compute_clip_score(image_url, summmary, clip_model, clip_processor)
                    print(f'Number{i}Clip score: {clip_score}')
            except Exception as e:
                st.error(f" Failed to generate image for {block['heading']}: {e}")
    else:
        st.error(" Mismatch: The number of content blocks and keyword blocks do not align.")



