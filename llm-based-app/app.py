import gradio as gr
import trafilatura
from openai import OpenAI
import dotenv
import json

# Load environment variables
dotenv.load_dotenv()

MODEL_NAME = "gemma3:4b"  # Model name for OpenAI API
# MODEL_NAME = "gpt-4o-mini"  # Model name for OpenAI API

# Initialize OpenAI client
# client = OpenAI()
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Function to extract article from URL
def extract_article(url):
    print("Downloading webpage...")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {"error": "Failed to download the webpage."}

    print("Extracting article content...")
    result = trafilatura.extract(downloaded, output_format="json", with_metadata=True)
    
    if result:
        data = json.loads(result)
        title = data.get("title", "")
        author = data.get("author", "")
        text = data.get("text", "")
        date = data.get("date", "")
        
        return {
            "title": title,
            "author": author,
            "date": date,
            "text": text
        }
    else:
        return {"error": "No article found on the webpage."}


# System prompt for GPT
system_prompt = f"""
You are an AI English language learning assistant for native Bangla speakers. Your goal is to help users learn English through real-world texts (such as articles, essays, or reports). When a user submits an English article, you must perform the following tasks:

1. **Bangla Summary**:
   - Summarize the entire article in clear and concise Bangla. Constraint the summary under 100 words.
   - Focus on the main ideas, arguments, and conclusions presented in the article.
   - Avoid complex or overly academic language.
   - Preserve the original message, including major facts and logical flow.

2. **Important Vocabulary Extraction**:
   - Identify a list of 5-10 key English words or phrases from the article that are essential for understanding the content.
   - Prioritize words that are:
     - Contextually important
     - Likely to improve reading comprehension
     - Useful in academic, journalistic, or civic discourse
   - For each word or phrase, provide:
     - **English term**
     - **Bangla meaning (translated accurately)**
     - **Context or usage note** (optional but recommended for clarity)

3. **Optional**:
   - Provide example sentences in both English and Bangla using the important vocabulary words.
   - Include grammar tips or idiomatic usage, if applicable.

**Formatting Guidelines**:
- Use clear section headers: “১. বাংলা সারাংশ”, “২. গুরুত্বপূর্ণ শব্দসমূহ ও অর্থ”
- Present vocabulary in a clean, tabular or bullet-point format
- Keep the tone educational, supportive, and suitable for self-learners

Important: Do not skip steps unless explicitly asked. Assume the user has a basic grasp of English and wants to learn through real-world comprehension.

The input text will always be in English. Your response should include both Bangla and English outputs as per the above structure.
"""

# Build user prompt with article
def build_user_prompt(article):
    return f"""
I want to learn English using the article below. Please perform the following tasks:

1. Summarize the article in **Bangla** in a clear and concise way. The summary should be easy to understand for someone with intermediate Bangla language skills.

2. Identify **important English words or phrases** from the article that are essential to understand its meaning. For each word or phrase, provide:
- English word/phrase
- Bangla meaning
- Contextual usage or explanation (if needed)
- Example sentences in both English and Bangla using the important vocabulary words.
3. Provide a **grammar tip** or idiomatic usage related to the vocabulary words, if applicable.

Here is the article:
Title: {article['title']}

Article: {article['text']}
"""


# Get GPT response
def get_gpt_response(user_prompt):
    print("Generating summary and vocabulary...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


# Main processing function for Gradio
def process_article(url):
    # Step 1: Extract article
    article = extract_article(url)
    if "error" in article:
        return article["error"]
    
    # Step 2: Build user prompt
    user_prompt = build_user_prompt(article)
    
    # Step 3: Get GPT response
    result = get_gpt_response(user_prompt)
    
    return article['text'], result


# Create Gradio interface
interface = gr.Interface(
    fn=process_article,
    inputs=gr.Textbox(label="Enter Article URL", placeholder="https://www.thedailystar.net/... "),
    outputs=[
        gr.Textbox(label="Article Text", placeholder="Extracted article text will appear here..."),
        gr.Markdown(label="Learning Output")
        ],
    title="English Learning Assistant for Bangla Speakers",
    description="Enter any English article URL. Get a Bangla summary and important vocabulary with meanings and examples.",
    theme="soft",
    allow_flagging="never"
)

# Launch app
if __name__ == "__main__":
    interface.launch(share=True)