# Text-based Emotion Detection Project

Hey there! 👋 This is my text emotion detection project that I built using Python, BERT, and Streamlit. It's not perfect (still gets confused sometimes, like thinking someone being sick is "joy" 😅), but it's been a fun learning experience!

## What Does It Do? 🤔

Basically, you can input any text (like tweets, messages, or whatever), and it tries to figure out the emotion behind it. It works with:
- Single text analysis
- Batch processing (if you've got a CSV full of texts)
- Multiple languages (it'll translate them to English first)

### Main Features

1. **Single Text Analysis**
   - Drop in any text, hit analyze
   - Shows the main emotion with confidence score
   - Gives you a cool radar chart of all possible emotions
   - Breaks down the text structure (words, sentences, etc.)
   - Even shows sentiment analysis (how positive/negative it is)

2. **Batch Analysis**
   - Upload a CSV file with lots of texts
   - Processes them all at once
   - Shows nice visualizations of the results
   - You can download the analysis as a CSV
   - Great for analyzing customer feedback or social media posts

3. **Analysis History**
   - Keeps track of everything you've analyzed
   - Shows trends over time
   - Helps spot patterns in emotions
   - Everything's saved until you close the app

## Tech Stack 🛠

Here's what I used to build this:
- **Python** (3.8+) - The backbone of everything
- **BERT** - For the actual emotion detection (using HuggingFace's transformers)
- **Streamlit** - For the web interface (super easy to use!)
- **Plotly & Altair** - For those nice-looking charts
- **TextBlob** - Extra help with sentiment analysis
- **Pandas** - Data handling (especially for batch processing)

## How to Run It 🚀

1. First, clone this repo:
   ```bash
   git clone <your-repo-url>
   cd text-based-emotion-detection
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

That's it! The app should open in your browser.

## Project Structure 📁

```
├── app.py                  # Main Streamlit application
├── train_emotion_model.py  # Model training script
├── utils/
│   ├── text_processor.py   # Text processing functions
│   └── visualizations.py   # Visualization components
├── emotion_model/         # Trained model files
├── cache/                # Translation cache
└── requirements.txt      # Project dependencies
```

## Known Issues & Future Improvements 🔧

Look, I'll be honest - it's not perfect. Here are some quirks and things I want to fix:

1. **Accuracy Issues**:
   - Sometimes gets confused with health-related texts
   - Can be overconfident with wrong predictions
   - Needs more training data for better context understanding

2. **Technical Stuff**:
   - Batch upload sometimes throws 403 errors (working on it!)
   - Translation can be slow for large batches
   - Memory usage could be optimized

3. **Future Plans**:
   - Retrain the model with better data
   - Add more emotion categories
   - Improve confidence scoring
   - Add export options for visualizations

## Model Details 🤖

The emotion detection model is based on BERT (bert-base-uncased) and was trained on a custom dataset. Current performance:
- Validation accuracy: ~73.59%
- Supported emotions: joy, sadness, anger, fear, surprise, neutral
- Can handle texts up to 512 tokens

## Usage Tips 💡

1. **Single Text Analysis**:
   - Works best with 1-3 sentences
   - Try different phrasings if results seem off
   - Check the confidence score - lower means less certain

2. **Batch Analysis**:
   - Use CSV files with a 'text' or 'Text' column
   - Keep files under 50MB for best performance
   - Check the sample file format if unsure

3. **General Tips**:
   - Translations work, but native English gives best results
   - Emojis are analyzed too! 😊
   - The longer the text, the more complex the analysis

## Dependencies

Here are the main packages you'll need:
```
streamlit>=1.8.0
torch>=1.9.0
transformers>=4.15.0
pandas>=1.3.0
plotly>=5.3.0
textblob>=0.15.3
googletrans>=3.1.0a0
```

## Contributing

Found a bug? Got an idea? Feel free to:
1. Open an issue
2. Submit a pull request
3. Reach out with suggestions

## License

This project is under the MIT License - do whatever you want with it! Just remember to credit me if you use it somewhere cool 😉

---

Built with ❤️ and lots of coffee ☕️ 

P.S. If the model thinks you're happy when you're sick, don't worry - it's not you, it's the AI 😅
