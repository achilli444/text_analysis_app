# Text Analysis Application

A Flask-based application that analyzes text from multiple sources and identifies similar facts across them.

## Features

- Analyze text from three different sources (A, B, C)
- Extract independent facts using natural language processing
- Resolve pronouns to their proper nouns
- Identify similar facts across different sources
- Visualize fact frequency with interactive horizontal bar charts
- Show variations of each fact when clicking on bars

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Application

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and go to: http://localhost:8080

## Usage

1. Paste different text blocks into the three source text areas (A, B, C)
2. Click "Analyze Texts" to process the input
3. View the horizontal bar chart showing fact frequencies
4. Click on any bar to see:
   - The original fact text
   - Which sources it appeared in (A, B, C)
   - All variations of how the fact was expressed

## Technical Details

- Uses spaCy for natural language processing
- Implements custom pronoun resolution
- Handles compound and complex sentences
- Uses similarity matching to identify related facts
- Interactive visualization with Plotly
