#!/bin/bash

# Make sure the script stops on first error
set -e

echo "Starting setup..."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify spaCy model installation
echo "Verifying spaCy model..."
if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "Installing spaCy model..."
    python -m spacy download en_core_web_sm
fi

# Create necessary directories
echo "Setting up Streamlit configuration..."
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

echo "Setup completed successfully!" 