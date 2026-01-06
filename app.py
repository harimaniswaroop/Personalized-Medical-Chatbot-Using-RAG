import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.auth import init_user_storage, hash_password, check_password, get_user, add_user
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import ollama
import time
import json
import re
import pandas as pd
import requests

# Flask App Initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

FIRECRAWL_API_KEY = "fc-0126c9f027574e3897a696c7517ca00b"

# Load Translation Model (Optimized for Speed)
MODEL_PATH = "C:/Users/hp/Desktop/Mini/medical_assistant/medical_assistant/olla"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to(device)
model = torch.compile(model)  # Compile model for faster inference

print("✅ Translation Model Loaded Successfully")


# ================= Utility Functions =================

def get_user_language(username):
    """Fetch user language preference from users.xlsx."""
    df = pd.read_excel("users.xlsx")
    user_row = df[df['username'] == username]
    return user_row['language'].values[0] if not user_row.empty else "eng_Latn"  # Default English


def translate(texts, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    """Batch translates multiple sentences to the target language."""
    if src_lang == tgt_lang:
        return texts  # No translation needed if languages match
    
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=300
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# ================= Routes =================

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user:
            stored_password = user[2]  # [email, username, password, language]
            if check_password(password, stored_password):
                session['user'] = username
                return redirect(url_for('chat'))
        return 'Invalid Credentials', 401
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        # Strong password policy
        if len(password) < 8 or not any(c.isupper() for c in password) or not any(c.isdigit() for c in password):
            return 'Weak Password', 400

        success = add_user(email, username, password)
        if not success:
            return 'Username already exists', 400

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/chat', methods=['GET'])
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/settings', methods=['GET'])
def settings():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('settings.html')


@app.route('/update_language', methods=['POST'])
def update_language():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    new_language = request.form['language']

    try:
        df = pd.read_excel("users.xlsx")
        if username in df['username'].values:
            df.loc[df['username'] == username, 'language'] = new_language
            df.to_excel("users.xlsx", index=False)
            flash("Language updated successfully!", "success")
        else:
            flash("User not found in database.", "error")
    except Exception as e:
        flash(f"Error updating language: {str(e)}", "error")

    return redirect(request.referrer or url_for('chat'))


# ================= Firecrawl Search =================

def firecrawl_search(query, num_of_searches=5):
    """Searches the web using Firecrawl API and returns results with sources."""
    url = "https://api.firecrawl.dev/v1/search"
    payload = {
        "limit": num_of_searches,
        "query": query,
        "extractorOptions": {
            "mode": "llm-extraction",
            "extractionSchema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "source": {"type": "string"}
                }
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        results = response.json().get('data', [])
        sources = []
        for result in results:
            if 'llm_extraction' in result:
                sources.append({
                    "text": result['llm_extraction']['summary'],
                    "url": result['llm_extraction']['source']
                })
        return sources[:4]  # Return top 4 sources
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []


# ================= Prompt Restructuring =================

def restructure(user_prompt):
    """Enhances prompt clarity and determines if web search is necessary."""
    system_prompt = """
    You are a medical prompt optimizer. Your tasks:
    1. Correct grammar and spelling while preserving meaning.
    2. Make it concise.
    3. Identify if external medical information is required:
       - Return 1 for fact-based medical queries.
       - Return 0 for greetings or non-medical topics.
    Respond ONLY in JSON format: {"digit": 0 or 1, "restructured_prompt": "..."}
    """

    try:
        response = ollama.chat(model='llama3.2:latest', messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        raw_json = re.sub(r'json|', '', response['message']['content']).strip()
        result = json.loads(raw_json)
        return {
            "digit": int(result["digit"]),
            "restructured_prompt": result["restructured_prompt"][:150]  # Limit length
        }
    except Exception as e:
        print(f"Restructure error: {str(e)}")
        return {"digit": 0, "restructured_prompt": user_prompt}


# ================= Main Chat Route =================

@app.route('/get_response', methods=['POST'])
def get_response():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    username = session['user']
    user_lang = get_user_language(username)
    user_prompt = request.json['prompt']

    # Step 1: Translate input if not English
    if user_lang != "eng_Latn":
        try:
            translated_input = translate(user_prompt, src_lang=user_lang, tgt_lang="eng_Latn")
            translated_input = translated_input[0] if isinstance(translated_input, list) else translated_input
            processing_prompt = translated_input
        except Exception as e:
            print(f"Translation error: {str(e)}")
            processing_prompt = user_prompt
    else:
        processing_prompt = user_prompt

    # Step 2: Restructure the prompt
    restructured = restructure(processing_prompt)
    digit, modified_prompt = restructured['digit'], restructured['restructured_prompt']

    # Step 3: Web search if needed
    sources = []
    context = ""
    if digit == 1:
        try:
            sources = firecrawl_search(modified_prompt)
            context = "\n".join([f"{s['text']}" for s in sources])
        except Exception as e:
            print(f"Search error: {str(e)}")

    # Step 4: Generate response
    try:
        start_time = time.time()
        response = ollama.chat(
            model='llama3.2:latest',
            messages=[{"role": "user", "content": f"""
            As a medical expert, respond to this query: {modified_prompt}
            Context: {context}
            - Keep response under 12 lines
            - Use simple language
            - Include emojis where appropriate
            - Format with clear paragraphs
            """}],
            options={'temperature': 0.4}
        )
        english_answer = response['message']['content']

        # Step 5: Translate back to user language if needed
        if user_lang != "eng_Latn":
            translated_answer = translate(english_answer, src_lang="eng_Latn", tgt_lang=user_lang)
            translated_answer = translated_answer[0] if isinstance(translated_answer, list) else translated_answer
        else:
            translated_answer = english_answer

        return jsonify({
            'response': translated_answer,
            'inference_time': round(time.time() - start_time, 2),
            'sources': [s['url'] for s in sources],
            'original_response': english_answer
        })

    except Exception as e:
        error_msg = translate(f"Error: {str(e)}", "eng_Latn", user_lang) if user_lang != "eng_Latn" else str(e)
        return jsonify({'error': error_msg}), 500


# ================= Run =================
if __name__ == '__main__':
    app.run(debug=True)
