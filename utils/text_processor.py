import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from googletrans import Translator
import langdetect
from textblob import TextBlob
import spacy
import emoji
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from collections import defaultdict

# Initialize translation cache
CACHE_FILE = "cache/translation_cache.json"
os.makedirs("cache", exist_ok=True)
translation_cache = {}

# Load existing cache if available
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        translation_cache = json.load(f)

# Initialize NLP components
translator = Translator()
try:
    nlp = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Emotion mapping with emojis
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨",
    "happy": "🤗",
    "neutral": "😐",
    "sad": "😔",
    "surprise": "😮"
}

# Load emotion detection model and tokenizer
model_path = "emotion_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Get emotion mapping from model config
with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
    label_mapping = json.load(f)
    id_to_emotion = label_mapping
    emotion_to_id = {v: int(k) for k, v in label_mapping.items()}

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

def is_english(text: str) -> bool:
    """Detect if text is in English."""
    try:
        return langdetect.detect(text) == 'en'
    except:
        return True

def translate_to_english(text: str) -> str:
    """Translate text to English with caching."""
    if not text or len(text.strip()) == 0:
        return text
        
    # Check cache first
    if text in translation_cache:
        return translation_cache[text]
        
    # If text is already in English, return as is
    if is_english(text):
        translation_cache[text] = text
        return text
        
    try:
        translated = translator.translate(text, src='auto', dest='en')
        translation_cache[text] = translated.text
        
        # Save cache periodically
        if len(translation_cache) % 50 == 0:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=2)
                
        return translated.text
    except Exception as e:
        print(f"⚠ Translation failed: {e}")
        return text

def get_text_stats(text: str) -> Dict:
    """Get detailed text statistics."""
    doc = nlp(text)
    
    # Basic stats
    stats = {
        "word_count": len(doc),
        "sentence_count": len(list(doc.sents)),
        "character_count": len(text),
        "emoji_count": len([c for c in text if c in emoji.EMOJI_DATA]),
        "emoji_list": [c for c in text if c in emoji.EMOJI_DATA],
    }
    
    # Part of speech stats
    pos_counts = defaultdict(int)
    for token in doc:
        pos_counts[token.pos_] += 1
    stats["pos_counts"] = dict(pos_counts)
    
    # Sentiment analysis
    blob = TextBlob(text)
    stats["polarity"] = blob.sentiment.polarity
    stats["subjectivity"] = blob.sentiment.subjectivity
    
    return stats

def get_detailed_emotion_analysis(text: str) -> Dict:
    """Get detailed emotion analysis using our trained model."""
    try:
        # Tokenize and prepare input
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            probabilities = probabilities.cpu().numpy()  # Convert to numpy array
        
        # Convert predictions to emotion labels with probabilities
        emotions_list = []
        for idx, prob in enumerate(probabilities):
            emotion = id_to_emotion[str(idx)]  # Convert idx to string to match config format
            emotions_list.append({
                "label": emotion,
                "score": float(prob),
                "emoji": emotions_emoji_dict.get(emotion, "")
            })
        
        # Sort by probability
        emotions_list = sorted(emotions_list, key=lambda x: x['score'], reverse=True)
        
        return {
            "emotions": emotions_list,
            "dominant_emotion": emotions_list[0],
            "confidence": float(max(probabilities))
        }
    except Exception as e:
        print(f"Error in emotion analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "emotions": [],
            "dominant_emotion": {"label": "unknown", "score": 0, "emoji": "❓"},
            "confidence": 0
        }

def process_text_batch(texts: List[str]) -> List[Dict]:
    """Process a batch of texts."""
    results = []
    for text in texts:
        translated = translate_to_english(text)
        stats = get_text_stats(translated)
        emotion_analysis = get_detailed_emotion_analysis(translated)
        results.append({
            "original_text": text,
            "translated_text": translated,
            "stats": stats,
            "emotion_analysis": emotion_analysis
        })
    return results 