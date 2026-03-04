import os
import re
import random
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rephrase_with_mistral import BloomsMistralRephraser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for frontend integration
CORS(app)

# ─── Model Configs ───
MODEL_CONFIGS = {
    "6level": {
        "path": "/home/harsh/projects/majorProject/question_gen_blooms_6level_model",
        "levels": {
            1: "Remember",
            2: "Understand",
            3: "Apply",
            4: "Analyze",
            5: "Evaluate",
            6: "Create"
        }
    },
    "4level": {
        "path": "/home/harsh/projects/majorProject/question_gen_blooms_4level_model",
        "levels": {
            1: "Remember & Understand",
            2: "Apply",
            3: "Analyze",
            4: "Evaluate & Create"
        }
    }
}

# Global cache to keep models loaded in memory
cache = {
    "model_path": None,
    "model": None,
    "tokenizer": None,
    "device": None,
    "rephraser": None
}

def load_ai_model(model_name_key):
    """Loads the specified local T5 model into memory."""
    config = MODEL_CONFIGS.get(model_name_key)
    if not config:
        raise ValueError(f"Invalid model selected: {model_name_key}. Choose '6level' or '4level'.")
        
    model_path = config["path"]
    
    # Check if already loaded
    if cache["model_path"] == model_path and cache["model"] is not None:
        return cache["model"], cache["tokenizer"], cache["device"]

    print(f"Loading local model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small",
        torch_dtype=torch.float32,
        device_map=None
    ).to(device)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Save to cache
    cache["model_path"] = model_path
    cache["model"] = model
    cache["tokenizer"] = tokenizer
    cache["device"] = device
    
    return model, tokenizer, device

def get_rephraser():
    """Initializes and returns the Mistral Rephraser."""
    if cache["rephraser"] is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key or api_key == "your_mistral_api_key_here":
            return None # API key not properly configured
            
        print("Initializing Mistral Rephraser...")
        try:
            cache["rephraser"] = BloomsMistralRephraser(api_key=api_key)
        except Exception as e:
            print(f"Failed to load Mistral Rephraser: {e}")
            return None
            
    return cache["rephraser"]

def chunk_text(text, max_chars=800):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current, current_len = [], 0
    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [sent], len(sent)
        else:
            current.append(sent)
            current_len += len(sent)
    if current: chunks.append(" ".join(current))
    return chunks

def generate_question(model, tokenizer, device, text, level_name):
    prompt = f"generate a {level_name.lower()} level question: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns available model configurations."""
    return jsonify({
        "success": True,
        "models": MODEL_CONFIGS
    })

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Main endpoint for generating questions.
    Expects JSON:
    {
        "text": "The source paragraph or document text...",
        "model_type": "6level" or "4level",
        "level_index": 1, (int, matches the model's level config keys)
        "num_questions": 3,
        "use_rephraser": true/false
    }
    """
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"success": False, "error": "Missing 'text' in request body"}), 400
        
    text = data.get("text", "").strip()
    model_type = data.get("model_type", "4level")
    level_index = int(data.get("level_index", 1))
    num_questions = int(data.get("num_questions", 1))
    use_rephraser = bool(data.get("use_rephraser", True))
    
    if not text:
        return jsonify({"success": False, "error": "Provided text is empty"}), 400
        
    if model_type not in MODEL_CONFIGS:
        return jsonify({"success": False, "error": f"Invalid model_type. Valid options: {list(MODEL_CONFIGS.keys())}"}), 400
        
    config = MODEL_CONFIGS[model_type]
    if level_index not in config["levels"]:
        return jsonify({"success": False, "error": f"Invalid level_index {level_index} for model_type '{model_type}'"}), 400
        
    level_name = config["levels"][level_index]
    
    try:
        # 1. Load Model dynamically based on request
        model, tokenizer, device = load_ai_model(model_type)
        
        # 2. Chunk Source Text
        chunks = chunk_text(text)
        sampled_chunks = random.sample(chunks, min(len(chunks), num_questions))
        
        # 3. Process Generation
        results = []
        rephraser = get_rephraser() if use_rephraser else None
        
        for chunk in sampled_chunks:
            # Generate via local T5
            raw_question = generate_question(model, tokenizer, device, chunk, level_name)
            
            # Predict tag for UI styling payload
            tag_match = re.match(r'^\[(.*?)\]', raw_question)
            tag_name = tag_match.group(1).lower() if tag_match else "unknown"
            
            # Rephrase via Mistral
            refined_question = None
            if rephraser:
                refined_question = rephraser.rephrase(chunk, raw_question)
                
            results.append({
                "source_chunk": chunk,
                "raw_question": raw_question,
                "refined_question": refined_question,
                "tag": tag_name,
                "target_level": level_name
            })
            
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask API for Bloom's Taxonomy QGen")
    app.run(host="0.0.0.0", port=5000, debug=True)
