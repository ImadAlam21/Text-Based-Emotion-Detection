#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install spacy model
python -m spacy download en_core_web_sm 