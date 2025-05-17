#!/bin/bash

# Make sure the script stops on first error
set -e

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install spacy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p ~/.streamlit/

# Create Streamlit config
echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml 