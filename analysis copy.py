import spacy
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load emotion analysis model
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Load readability model
readability_tokenizer = AutoTokenizer.from_pretrained("agentlans/deberta-v3-xsmall-readability")
readability_model = AutoModelForSequenceClassification.from_pretrained("agentlans/deberta-v3-xsmall-readability")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Readability Helper Functions --- #

def get_readability_score(text):
    inputs = readability_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = readability_model(**inputs)
    return outputs.logits.item()

def normalize_readability_to_complexity(score, min_grade=1, max_grade=20):
    clamped = max(min(score, max_grade), min_grade)
    return round(1 + ((clamped - min_grade) / (max_grade - min_grade)) * 9, 1)

def interpret_readability(score):
    if score < 4:
        return "Very easy to read (early primary level)"
    elif score < 7:
        return "Simple and readable (upper primary)"
    elif score < 10:
        return "Moderate complexity (middle school level)"
    elif score < 13:
        return "High school level reading"
    elif score < 17:
        return "College-level writing complexity"
    else:
        return "Postgraduate or academic-level writing"

# --- Main Analysis Function --- #

def analyze_journal_text(journal_entry):
    doc = nlp(journal_entry)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct]
    sentences = [sent.text for sent in doc.sents]

    word_counts = Counter(tokens)
    sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    unique_words = set(tokens)
    ttr_score = len(unique_words) / len(tokens) if tokens else 0
    syntax_depths = []
    for sent in doc.sents:
        max_depth = 0
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        syntax_depths.append(max_depth)
    avg_syntax_depth = sum(syntax_depths) / len(syntax_depths) if syntax_depths else 0

    word_count = len(tokens)

    # Readability
    grade_score = get_readability_score(journal_entry)
    complexity_score = normalize_readability_to_complexity(grade_score)
    interpretation = interpret_readability(grade_score)

    complexity_data = {
        "complexity_score": complexity_score,
        "readability_grade_level": round(grade_score, 2),
        "readability_interpretation": interpretation,
        "components": {
            "word_count": word_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "lexical_diversity": round(ttr_score, 2),
            "avg_syntactic_depth": round(avg_syntax_depth, 2)
        }
    }

    return complexity_data

# --- Emotion Analysis --- #

def analyze_emotions(text):
    results = emotion_analyzer(text)
    emotion_scores = {e['label']: e['score'] for e in results[0]}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    return {
        "emotion_scores": emotion_scores,
        "dominant_emotion": dominant_emotion
    }

# --- Combined Entry Analysis --- #

def analyze_entry(journal_entry):
    complexity = analyze_journal_text(journal_entry)
    emotion = analyze_emotions(journal_entry)
    return {
        "original_text": journal_entry,
        "complexity": complexity,
        "emotions": emotion
    }

# --- Flask Routes --- #

@app.route('/analyze', methods=['POST'])
def analyze_entry_endpoint():
    try:
        data = request.get_json()
        if not data or 'journal_text' not in data:
            return jsonify({'error': 'Missing required field: journal_text'}), 400
        journal_text = data['journal_text']
        result = analyze_entry(journal_text)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error in /analyze endpoint")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/emotions', methods=['POST'])
def analyze_emotions_endpoint():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        text = data['text']
        result = analyze_emotions(text)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error in /analyze/emotions endpoint")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/complexity', methods=['POST'])
def analyze_complexity_endpoint():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        text = data['text']
        result = analyze_journal_text(text)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error in /analyze/complexity endpoint")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
