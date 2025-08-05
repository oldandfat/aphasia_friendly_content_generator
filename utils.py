# Core business logic:
# Accept the user input (prompt), call the model, and retrieve the result.

"""
In Python and AI development, `utils.py` or a `utils` directory is typically used for:
    1. Encapsulating common functions (e.g., string processing, data cleaning)
    2. Code sharing (to avoid duplication)
    3. Improving code maintainability (functional decomposition)
"""

# 1. Import packages
from langchain_openai import ChatOpenAI  # LLM model
from langchain.chains import ConversationChain  # Conversation chain: LLM + memory
from langchain.prompts import ChatPromptTemplate  # Chat prompt template
from langchain.memory import ConversationBufferMemory  # Conversation memory
from langchain.chains import LLMChain
import re
from keybert import KeyBERT
import nltk
import json
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
import openai
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO

def get_response(prompt, memory, api_key):
    # Create the LLM model
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        openai_api_key=api_key
    )

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a speech and language therapist. You help people with aphasia understand written information."),

        ("human",
         """Please simplify the following text for people with aphasia.
    Use the following guidelines very strictly.

    ### ✳️ Language Rules
    1. Use very short sentences. One idea per sentence.
    2. Use simple and clear grammar. Avoid pronouns (like he, she, it, they). Use names or nouns instead.
    3. Use easy words (CEFR A1-A2, and some B1). Avoid abstract or difficult words.

    ### ✳️ Structure and Layout
    4. Add a title at the top. Write it like this: **Title: ...**
    5. Break the text into clear sections. Use section headings. Write headings like this: **Heading: ...**
    6. Use bullet points (- ...) when listing items or key facts.
    7. Leave an empty line between each paragraph or bullet list.
    8. Each paragraph should be 2–3 short sentences only.

    ### ✳️ Formatting Style
    9. Use a sans-serif font (like Arial or Calibri).
    10. Use font size 16–18 pt.
    11. Leave extra space between lines.
    12. Make sure the layout is not too wide. It should fit in half of an A4 page.
    13. Use blue for headings and bold text.
    
    

    ### ✳️ Emotional and Content Safety
    14. Do not use violent, legal, or distressing details.
    15. Be gentle, calm, and emotionally clear.
    
    ### ✳️ Vocabulary Preference
    - Try to use simple, high-frequency words.
    - Here are some good examples: go, get, eat, make, doctor, house, book, phone, water, help.
    - Avoid complicated or rare words like: consume, residence, physician, legal, commence.

    ### ✳️ Output Format
    - Output as plain text with markdown-style formatting:
        - Start with a clear **Title: ...**
        - Then use **Heading: ...** and normal paragraphs.
        - Use bullet points (- ...) where appropriate.
    - Do not include explanations or extra notes.
    - Only output the simplified text.
    
    

    Now simplify the following text:
    {text}
    """),

        ("ai", "Sure. Here is the simplified version:")
    ])

    # Chain the prompt template with the model
    chain = prompt_template | llm

    # Pass input text to the model
    response = chain.invoke({"text": prompt})
    return response.content


def format_output(raw_text: str) -> str:
    """
    Format the GPT output into HTML for display in Streamlit or web.
    - Wrap each content block in a bordered box
    - Align all text top-left
    - Detect titles, headings, bullet points
    """
    lines = raw_text.strip().split("\n")
    html_output = """<style>
    .container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        font-family: Arial, sans-serif;
        font-size: 16pt;
        line-height: 1.6;
    }
    .box {
        border: 1px solid #ccc;
        padding: 1em;
        margin-bottom: 1em;
        width: 100%;
        max-width: 480px;
        background-color: #fdfdfd;
        text-align: left;
    }
    .section-title {
        font-weight: bold;
        color: blue;
        font-size: 20pt;
        margin-bottom: 0.5em;
    }
    .text-block {
        margin-bottom: 0.5em;
    }
    ul {
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        padding-left: 1.2em;
    }
    li {
        margin-bottom: 0.3em;
    }
    </style>
    <div class="container">
    """

    for line in lines:
        line = line.strip()
        if not line:
            continue

        html_output += '<div class="box">\n'

        # Title or heading
        match_title = re.match(r"^\*{1,2}(.+?)\*{1,2}$", line)
        if match_title:
            clean_title = match_title.group(1).strip()
            html_output += f'<div class="section-title">{clean_title}</div>\n'
        elif line.startswith("- "):  # bullet point
            item = line[2:].strip()
            html_output += f"<ul><li>{item}</li></ul>\n"
        elif re.match(r"^\d+\.\s", line):  # numbered
            html_output += f'<div class="text-block"><strong>{line}</strong></div>\n'
        else:
            html_output += f'<div class="text-block">{line}</div>\n'

        html_output += "</div>\n"  # close box

    html_output += "</div>"  # close container
    return html_output


def is_visual_concept(word):
    synsets = wn.synsets(word)
    for syn in synsets:
        for hyper in syn.closure(lambda s: s.hypernyms()):
            if hyper.name().split('.')[0] in {
                'person', 'human', 'location', 'building', 'room',
                'action', 'event', 'motion', 'communication', 'emotion', 'artifact'
            }:
                return True
    return False

# Choose the best keywords to guide image generation
def select_visual_keywords_auto(core_keywords, extra_keywords, max_num=5):
    selected = [kw for kw in core_keywords if is_visual_concept(kw)]

    if len(selected) < max_num:
        selected += [kw for kw in extra_keywords if is_visual_concept(kw) and kw not in selected]

    return selected[:max_num]

#  Main function: extract keywords from text and suggest image prompts
def keyword_extract(content):
    # # Load high-frequency word list
    df = pd.read_csv("highfreqwords.csv")
    high_freq_words = set(df['word'].str.lower())

    text = content
    blocks = re.split(r"(Heading: .+)", text)
    results = []

    for i in range(1, len(blocks), 2):
        heading = blocks[i].replace("Heading: ", "").strip()
        prag_text = blocks[i + 1].strip()
        keywords = set()

        # Split sentences and perform POS tagging
        sentences = sent_tokenize(prag_text)
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            for word, tag in pos_tags:
                word_lower = word.lower()
                if tag.startswith("NN") or tag.startswith("VB"):
                    keywords.add(word_lower)

        # Separate high-frequency words and expanded keywords
        core_keywords = {kw for kw in keywords if kw in high_freq_words}
        extra_keywords = keywords - core_keywords

        # Automatically suggest image keywords
        visual_keywords = select_visual_keywords_auto(core_keywords, extra_keywords)

        results.append({
            "heading": heading,
            "core_keywords": sorted(core_keywords),
            "extra_keywords": sorted(extra_keywords),
            "visual_keywords": visual_keywords
        })


    print(json.dumps(results, indent=2))
    return results




def split_heading_blocks(simplified_text):
    pattern = r"\*\*Heading: (.+?)\*\*"
    parts = re.split(pattern, simplified_text)

    blocks = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip()
        full_prompt = f"{heading}: {content}"
        blocks.append({
            "heading": heading,
            "content": content,
            "prompt": full_prompt
        })
    return blocks



def generate_image_from_text(prompt_text, api_key):

    openai.api_key = api_key

    response = openai.images.generate(
        model="dall-e-3",
        prompt= (

        f"Create a clean, realistic illustration that clearly shows the scene described below. "
        f"Make the image easy to understand, with clear objects and people. "
        f"Use an adult-oriented style — avoid childish or cartoon elements. "
        f"Keep the style consistent with modern, flat or realistic visuals. "
        f"Do not include any text in the image."
        f"Scene description:  {prompt_text}"

    ),
        n=1,
        size="1024x1024"
    )
    return response.data[0].url

def summary_output_for_images(text_output,api_key):

        # Create the LLM model
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=api_key
        )




        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ('system','You can summarize the text to guide image generation'),
            ('human',"You are a visual scene summarizer. Read the following simplified paragraph, "
            "and rewrite it as one sentence that clearly describes what should appear in an illustration. "
            "Focus on visible actions and settings. Do not include opinions or abstract ideas.\n\n"
            "Paragraph: {text}\n"
            "Visual Summary:")
        ])


        # Chain the prompt template with the model
        chain = prompt_template | llm

        # Pass input text to the model
        response = chain.invoke({"text":text_output})
        return response.content.strip()

#Calculate image-text matching score
from torch.nn.functional import cosine_similarity

def compute_clip_score(image_url, text, model, processor):
    # 加载图像
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # 处理图文
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # shape: (1, 512)
        text_embeds = outputs.text_embeds    # shape: (1, 512)

        # 计算 cosine similarity 并映射为 0~1
        cos_sim = cosine_similarity(image_embeds, text_embeds).item()
        normalized_score = (cos_sim + 1) / 2

    return normalized_score






if __name__ == '__main__':
    # Example input prompt
    prompt = """
    The Legend of Robin Hood

The Origin

In the time of King Richard the Lionheart, when England was ruled by corrupt nobles while the king was away on crusade, there lived a young nobleman named Robin of Locksley. Robin was known for his exceptional skill with a bow and his strong sense of justice.

When Robin's father died, the Sheriff of Nottingham seized the family lands through false accusations and legal trickery. Robin found himself an outlaw, stripped of his title and possessions, forced to flee into the depths of Sherwood Forest.
"""

    # Create conversation memory buffer; `returnmessages` controls whether to return conversation history
    memory = ConversationBufferMemory(returnmessages=True)

    # Set your API key
    api_key = "sk-proj-..."

    # Call the response function
    result = get_response(prompt, memory, api_key)
    keyword_extract(result)
    print(result)