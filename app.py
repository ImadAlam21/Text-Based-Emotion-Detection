import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Advanced Emotion Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import altair as alt
import os
from googletrans import Translator
import langdetect
import json
from pathlib import Path
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils.text_processor import process_text_batch, translate_to_english, get_detailed_emotion_analysis, get_text_stats
from utils.visualizations import (
    create_emotion_radar_chart,
    create_pos_bar_chart,
    create_sentiment_gauge,
    display_text_stats,
    create_confidence_indicator,
    create_historical_emotions_chart
)
from datetime import datetime
from textblob import TextBlob
import plotly.graph_objects as go
import shutil
import gdown
import zipfile

# Create necessary directories
os.makedirs("cache", exist_ok=True)
os.makedirs(".streamlit/model_cache", exist_ok=True)

def download_and_extract_model():
    """Download the model from Google Drive and extract it."""
    model_url = "https://drive.google.com/uc?id=1-gnJaYMM59ClEdOMXjYZS2PNNLmy0988"
    output = "emotion_model.zip"
    
    if not os.path.exists("emotion_model"):
        try:
            st.info("⏳ Downloading model... This will take a few minutes but only happens once.")
            gdown.download(model_url, output, quiet=False)
            
            st.info("📦 Extracting model files...")
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall("emotion_model")
            
            # Clean up zip file
            os.remove(output)
            st.success("✅ Model downloaded and extracted successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            return False
    return True

# Initialize model and tokenizer
@st.cache_resource
def load_model():
    model_path = 'emotion_model'
    cached_model_path = '.streamlit/model_cache'
    
    # First, ensure model is downloaded
    if not os.path.exists(model_path):
        if not download_and_extract_model():
            st.error("Could not load the model. Please try again later.")
            st.stop()
    
    # Check if model is already in persistent storage
    if not os.path.exists(os.path.join(cached_model_path, 'pytorch_model.bin')):
        st.info("🔄 First time setup: Optimizing and caching the model... This will take a few minutes but only happens once.")
        
        # Load label mapping first
        with open(os.path.join(model_path, 'label_mapping.json'), 'r') as f:
            label_mapping = json.load(f)
        
        # Update config with proper label mappings
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['id2label'] = label_mapping
        config['label2id'] = {v: int(k) for k, v in label_mapping.items()}
        
        # Save config to persistent storage
        os.makedirs(cached_model_path, exist_ok=True)
        with open(os.path.join(cached_model_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
            
        # Copy necessary files to persistent storage
        for file in ['vocab.txt', 'label_mapping.json']:
            shutil.copy2(os.path.join(model_path, file), os.path.join(cached_model_path, file))
        
        # Load and quantize model
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(label_mapping),
            problem_type="single_label_classification"
        )
        
        # Quantize the model to reduce size (8-bit quantization)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Save quantized model to persistent storage
        model.save_pretrained(cached_model_path)
        
        # Save tokenizer to persistent storage
        tokenizer = BertTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(cached_model_path)
        
        st.success("✅ Model optimization and caching completed!")
    else:
        # Load from persistent storage
        with open(os.path.join(cached_model_path, 'label_mapping.json'), 'r') as f:
            label_mapping = json.load(f)
        
        tokenizer = BertTokenizer.from_pretrained(cached_model_path)
        model = BertForSequenceClassification.from_pretrained(cached_model_path)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, tokenizer, label_mapping, device

model, tokenizer, label_mapping, device = load_model()

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .emotion-box {
        background-color: #1e1e1e;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .emotion-box h2 {
        color: white;
        margin: 0;
        font-size: 1.8em;
    }
    .emotion-box p {
        color: #e0e0e0;
        margin: 10px 0 0 0;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'batch_state' not in st.session_state:
    st.session_state.batch_state = {
        'use_sample': False,
        'results': None,
        'has_analyzed': False
    }

# Sidebar
with st.sidebar:
    st.title("🎭 Navigation")
    page = st.radio(
        "Choose a feature:",
        ["Single Text Analysis", "Batch Analysis", "Analysis History", "Model Performance"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This advanced emotion analysis tool uses state-of-the-art NLP models to:
    - Detect emotions in any language
    - Provide detailed text analysis
    - Track emotion patterns over time
    - Generate visual insights
    """)

# Initialize translation cache
CACHE_FILE = "cache/translation_cache.json"
translation_cache = {}

# Load existing cache if available
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        translation_cache = json.load(f)

translator = Translator()

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def is_english(text):
    try:
        return langdetect.detect(text) == 'en'
    except:
        return True

def translate_to_english(text):
    if not text or len(text.strip()) == 0:
        return text
        
    # Check cache first
    if text in translation_cache:
        return translation_cache[text]
        
    try:
        lang = langdetect.detect(text)
        if lang == 'en':
            translation_cache[text] = text
            return text
            
        translated = translator.translate(text, src=lang, dest='en')
        translation_cache[text] = translated.text
        
        # Save cache periodically
        if len(translation_cache) % 50 == 0:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=2)
                
        return translated.text
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

def analyze_text(text):
    """Analyze text using the trained model."""
    # Translate if necessary
    text = translate_to_english(text)
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        predictions = {label_mapping[str(i)]: float(prob) for i, prob in enumerate(probs)}
        
    return predictions

def single_text_analysis():
    st.title("🔍 Single Text Analysis")
    
    text = st.text_area(
        "Enter your text (any language):",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🎯 Analyze Emotions")
    with col2:
        clear_button = st.button("🔄 Clear")
        if clear_button:
            st.experimental_rerun()
    
    if analyze_button and text:
        with st.spinner("Analyzing text..."):
            # Get detailed emotion analysis
            analysis_result = get_detailed_emotion_analysis(text)
            
            # Main results section
            st.markdown("### 📊 Emotion Analysis Results")
            
            # Create three columns for the main metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎯 Primary Emotion")
                dominant_emotion = analysis_result["dominant_emotion"]
                st.markdown(
                    f"""
                    <div class="emotion-box">
                        <h2>{dominant_emotion['emoji']} {dominant_emotion['label'].title()}</h2>
                        <p>Confidence: {dominant_emotion['score']:.1%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown("#### 🌡️ Confidence Score")
                confidence_chart = create_confidence_indicator(dominant_emotion['score'])
                st.plotly_chart(confidence_chart, use_container_width=True)
            
            with col3:
                st.markdown("#### 💭 Text Statistics")
                stats = get_text_stats(text)
                st.metric("Words", stats["word_count"])
                st.metric("Sentences", stats["sentence_count"])
                st.metric("Emojis", stats["emoji_count"])
            
            # Detailed Analysis Section
            st.markdown("### 🔍 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Emotion Distribution")
                # Create radar chart for all emotions
                emotions_dict = {emotion['label']: emotion['score'] for emotion in analysis_result['emotions']}
                radar_chart = create_emotion_radar_chart(emotions_dict)
                st.plotly_chart(radar_chart, use_container_width=True)
            
            with col2:
                st.markdown("#### Sentiment Analysis")
                # Get sentiment from TextBlob
                blob_sentiment = TextBlob(text).sentiment
                sentiment_gauge = create_sentiment_gauge(blob_sentiment.polarity)
                st.plotly_chart(sentiment_gauge, use_container_width=True)
                st.markdown(f"""
                    - **Polarity**: {blob_sentiment.polarity:.2f} (-1 negative to 1 positive)
                    - **Subjectivity**: {blob_sentiment.subjectivity:.2f} (0 objective to 1 subjective)
                """)
            
            # Language Analysis
            st.markdown("### 🗣️ Language Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Parts of Speech Distribution")
                pos_chart = create_pos_bar_chart(stats["pos_counts"])
                st.altair_chart(pos_chart, use_container_width=True)
            
            with col2:
                st.markdown("#### Text Information")
                if not is_english(text):
                    st.info(f"Original text was translated from another language for analysis.")
                
                # Display emojis if present
                if stats["emoji_list"]:
                    st.markdown("#### Emojis Used")
                    st.markdown(" ".join(stats["emoji_list"]))
            
            # Save to history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "analysis": analysis_result,
                "stats": stats,
                "sentiment": blob_sentiment._asdict()
            }
            st.session_state.history.append(history_entry)
            
            # Show success message
            st.success("Analysis completed successfully! Check out the Analysis History tab to see trends over time.")

def batch_analysis():
    st.title("📊 Batch Text Analysis")
    
    st.markdown("""
    ### Instructions
    Upload a CSV file containing texts to analyze. The file should have a column named 'text' or 'Text'.
    You can also use the sample file for testing.
    """)
    
    # Add option to use sample file
    use_sample = st.checkbox("Use sample texts file", value=st.session_state.batch_state['use_sample'])
    st.session_state.batch_state['use_sample'] = use_sample
    
    # Only load data if we haven't analyzed yet or if the sample choice changed
    if not st.session_state.batch_state['has_analyzed'] or use_sample != st.session_state.batch_state['use_sample']:
        if use_sample:
            try:
                df = pd.read_csv('sample_texts.csv')
                st.success("Successfully loaded sample texts!")
            except Exception as e:
                st.error(f"Error loading sample file: {str(e)}")
                return
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type="csv"
            )
            
            if not uploaded_file:
                return
                
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        
        # Check for text column
        text_column = None
        if 'text' in df.columns:
            text_column = 'text'
        elif 'Text' in df.columns:
            text_column = 'Text'
        
        if not text_column:
            st.error("CSV file must contain a column named 'text' or 'Text'")
            return
        
        # Show sample of data
        st.markdown("### Preview of uploaded data")
        st.dataframe(df.head())
        
        if st.button("🚀 Start Batch Analysis"):
            with st.spinner("Analyzing texts..."):
                results = []
                progress_bar = st.progress(0)
                
                # Process texts
                total_texts = len(df[text_column])
                for idx, text in enumerate(df[text_column]):
                    if pd.isna(text):  # Skip empty/NA values
                        continue
                        
                    analysis = get_detailed_emotion_analysis(str(text))
                    stats = get_text_stats(str(text))
                    sentiment = TextBlob(str(text)).sentiment
                    
                    result = {
                        'text': text,
                        'dominant_emotion': analysis['dominant_emotion']['label'],
                        'confidence': analysis['dominant_emotion']['score'],
                        'word_count': stats['word_count'],
                        'sentiment_polarity': sentiment.polarity,
                        'sentiment_subjectivity': sentiment.subjectivity
                    }
                    
                    results.append(result)
                    
                    # Add to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().isoformat(),
                        "text": text,
                        "analysis": analysis,
                        "stats": stats,
                        "sentiment": sentiment._asdict(),
                        "source": "batch_analysis"
                    })
                    
                    # Update progress
                    progress = (idx + 1) / total_texts
                    progress_bar.progress(progress)
                
                # Store results in session state
                st.session_state.batch_state['results'] = results
                st.session_state.batch_state['has_analyzed'] = True
    
    # Display results if they exist
    if st.session_state.batch_state['has_analyzed'] and st.session_state.batch_state['results']:
        results = st.session_state.batch_state['results']
        results_df = pd.DataFrame(results)
        
        # Display results
        st.markdown("### Analysis Results")
        st.dataframe(results_df)
        
        # Create summary visualizations
        st.markdown("### Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion distribution
            emotion_counts = results_df['dominant_emotion'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=emotion_counts.index,
                values=emotion_counts.values,
                hole=.3
            )])
            fig.update_layout(title="Distribution of Dominant Emotions")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = go.Figure(data=[go.Histogram(x=results_df['confidence'])])
            fig.update_layout(title="Distribution of Confidence Scores")
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            "emotion_analysis_results.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Clear results button
        if st.button("🔄 Clear Results and Analyze New Data"):
            st.session_state.batch_state['has_analyzed'] = False
            st.session_state.batch_state['results'] = None
            st.experimental_rerun()

def analysis_history():
    st.title("📈 Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history available yet. Try analyzing some text first!")
        return
    
    # Convert history to DataFrame
    history_data = []
    for entry in st.session_state.history:
        history_data.append({
            'timestamp': pd.to_datetime(entry['timestamp']),
            'text': entry['text'],
            'emotion': entry['analysis']['dominant_emotion']['label'],
            'confidence': entry['analysis']['dominant_emotion']['score'],
            'sentiment': entry.get('sentiment', {}).get('polarity', 0),
            'subjectivity': entry.get('sentiment', {}).get('subjectivity', 0)
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Display summary metrics
    st.markdown("### 📊 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", len(history_df))
    with col2:
        if not history_df.empty:
            most_common_emotion = history_df['emotion'].mode()[0]
            emotion_count = history_df['emotion'].value_counts()[most_common_emotion]
            st.metric("Most Common Emotion", f"{most_common_emotion} ({emotion_count} times)")
    with col3:
        if not history_df.empty:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
    
    # Emotion trends over time
    st.markdown("### 📈 Emotion Trends")
    if not history_df.empty:
        # Emotion distribution
        emotion_counts = history_df['emotion'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=emotion_counts.index,
            values=emotion_counts.values,
            hole=.3
        )])
        fig.update_layout(
            title="Distribution of Emotions",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['confidence'],
            mode='lines+markers',
            name='Confidence'
        ))
        fig.update_layout(
            title="Confidence Trend Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment analysis
        st.markdown("### 🎭 Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig = go.Figure(data=[go.Histogram(x=history_df['sentiment'])])
            fig.update_layout(
                title="Distribution of Sentiment Scores",
                xaxis_title="Sentiment (-1 to 1)",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment vs. Confidence scatter
            fig = go.Figure(data=[go.Scatter(
                x=history_df['sentiment'],
                y=history_df['confidence'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=history_df['subjectivity'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Subjectivity")
                )
            )])
            fig.update_layout(
                title="Sentiment vs. Confidence",
                xaxis_title="Sentiment Score",
                yaxis_title="Confidence Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # History table
    st.markdown("### 📝 Analysis History")
    if not history_df.empty:
        history_df['formatted_time'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df = history_df[['formatted_time', 'text', 'emotion', 'confidence', 'sentiment']].copy()
        display_df.columns = ['Timestamp', 'Text', 'Emotion', 'Confidence', 'Sentiment']
        st.dataframe(
            display_df.sort_values('Timestamp', ascending=False),
            use_container_width=True
        )
        
        # Export option
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download History CSV",
            csv,
            "emotion_analysis_history.csv",
            "text/csv",
            key='download-history'
        )

def model_performance():
    st.title("🎯 Model Performance")
    
    st.markdown("""
    ### Model Information
    - Base Model: BERT (bert-base-uncased)
    - Training Data: Custom emotion dataset
    - Validation Accuracy: 73.59%
    
    ### Supported Emotions
    """)
    
    # Display supported emotions
    emotions_df = pd.DataFrame(list(label_mapping.items()), columns=['ID', 'Emotion'])
    st.dataframe(emotions_df.sort_values('Emotion'))

# Main app logic
if page == "Single Text Analysis":
    single_text_analysis()
elif page == "Batch Analysis":
    batch_analysis()
elif page == "Analysis History":
    analysis_history()
else:
    model_performance()

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit and state-of-the-art NLP models"
)
