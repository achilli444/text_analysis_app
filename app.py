from flask import Flask, render_template, request, jsonify
import spacy
import plotly.express as px
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher
import os

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

def resolve_pronouns(text):
    """
    Replace pronouns with their corresponding proper nouns or subjects.
    """
    doc = nlp(text)
    resolved_text = text
    
    # Create a mapping of pronouns to their most recent proper noun antecedents
    antecedents = {}
    current_proper_noun = None
    
    for token in doc:
        if token.pos_ in ['PROPN', 'NOUN'] and token.dep_ in ['nsubj', 'nsubjpass']:
            current_proper_noun = token.text
        elif token.pos_ == 'PRON' and current_proper_noun:
            antecedents[token.i] = current_proper_noun
    
    # Replace pronouns with their antecedents, from last to first to maintain indices
    for idx in sorted(antecedents.keys(), reverse=True):
        pronoun_token = doc[idx]
        resolved_text = resolved_text[:pronoun_token.idx] + antecedents[idx] + resolved_text[pronoun_token.idx + len(pronoun_token.text):]
    
    return resolved_text

def extract_single_clause(doc):
    """
    Extract a single independent clause from a doc object.
    """
    # Find the root verb and its associated subject and object
    root = None
    subject = None
    obj = None
    
    for token in doc:
        if token.dep_ == 'ROOT':
            root = token
        elif token.dep_ in ['nsubj', 'nsubjpass']:
            subject = token
        elif token.dep_ in ['dobj', 'pobj']:
            obj = token
    
    if root:
        # Collect all tokens that are part of the independent clause
        clause_tokens = set()
        clause_tokens.add(root)
        if subject:
            clause_tokens.add(subject)
        if obj:
            clause_tokens.add(obj)
        
        # Add essential modifiers
        for token in doc:
            if token.head in clause_tokens and token.dep_ not in ['mark', 'advcl', 'relcl']:
                clause_tokens.add(token)
            # Include certain types of modifiers that complete the meaning
            elif token.dep_ in ['xcomp', 'ccomp', 'acomp', 'attr']:
                clause_tokens.add(token)
                # Add any words that modify these completions
                for child in token.children:
                    if child.dep_ not in ['mark', 'advcl', 'relcl']:
                        clause_tokens.add(child)
        
        # Sort tokens by their original position
        clause_tokens = sorted(clause_tokens, key=lambda x: x.i)
        
        # Reconstruct the clause text
        clause_text = ' '.join(token.text for token in clause_tokens)
        return clause_text.strip()
    
    return None

def extract_independent_clause(sent):
    """
    Extract independent clauses from a sentence, handling compound and complex sentences.
    """
    doc = nlp(sent)
    clauses = []
    
    # Find all coordinating conjunctions (but, and, or, etc.)
    coord_conjunctions = [token for token in doc if token.dep_ == 'cc']
    
    if not coord_conjunctions:
        # Simple sentence - process as before
        clause = extract_single_clause(doc)
        if clause:
            clauses.append(clause)
    else:
        # Compound sentence - split on coordinating conjunctions
        prev_index = 0
        subjects = {}  # Keep track of subjects for each part
        
        # First, identify all subjects and their associated words
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                subjects[token.i] = token.text
        
        # Process each part of the compound sentence
        for conj in coord_conjunctions:
            # Find the right boundary of the current clause
            right_boundary = conj.i
            
            # Extract the clause from prev_index to right_boundary
            first_part = doc[prev_index:right_boundary].text
            clause1 = extract_single_clause(nlp(first_part))
            if clause1:
                clauses.append(clause1)
            
            # Find the start of the next independent clause
            next_start = conj.i + 1
            while next_start < len(doc) and doc[next_start].dep_ in ['cc', 'punct']:
                next_start += 1
            
            # If we're at the last conjunction, process the remainder
            if conj == coord_conjunctions[-1]:
                second_part = doc[next_start:].text
                
                # If no explicit subject in second part, use the subject from first part
                if not any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc[next_start:]):
                    # Find the most recent subject
                    recent_subject = None
                    for i in sorted(subjects.keys()):
                        if i < next_start:
                            recent_subject = subjects[i]
                    
                    if recent_subject:
                        second_part = f"{recent_subject} {second_part}"
                
                clause2 = extract_single_clause(nlp(second_part))
                if clause2:
                    clauses.append(clause2)
            
            prev_index = next_start
    
    return clauses

def extract_facts(text):
    """
    Extract independent clauses as facts from the text.
    """
    # First resolve pronouns
    resolved_text = resolve_pronouns(text)
    doc = nlp(resolved_text)
    
    facts = []
    for sent in doc.sents:
        # Extract all independent clauses from the sentence
        clauses = extract_independent_clause(sent.text)
        facts.extend(clauses)
    
    return facts

def compare_facts(fact1, fact2, threshold=0.8):
    """
    Compare two facts to determine if they are similar.
    This function can be enhanced with more sophisticated comparison methods:
    - Semantic similarity using word embeddings
    - Entity matching
    - Dependency parsing comparison
    - Topic modeling
    - etc.

    Args:
        fact1 (str): First fact to compare
        fact2 (str): Second fact to compare
        threshold (float): Similarity threshold (0-1)

    Returns:
        bool: True if facts are considered similar
    """
    # Current implementation uses sequence matching
    # This can be replaced or enhanced with more sophisticated methods
    similarity = SequenceMatcher(None, fact1.lower(), fact2.lower()).ratio()
    
    # You can also add spaCy's similarity comparison
    doc1 = nlp(fact1)
    doc2 = nlp(fact2)
    semantic_similarity = doc1.similarity(doc2)
    
    # Combine different similarity metrics
    combined_similarity = (similarity + semantic_similarity) / 2
    
    return combined_similarity >= threshold

def find_or_add_fact(fact, unique_facts, threshold=0.7):
    """
    Find a matching fact in the existing set or add as new.
    
    Args:
        fact (str): New fact to check
        unique_facts (list): List of existing unique facts
        threshold (float): Similarity threshold
    
    Returns:
        str: The matching fact or the new fact if no match found
    """
    for existing_fact in unique_facts:
        if compare_facts(fact, existing_fact, threshold):
            return existing_fact
    return fact

def truncate_text(text, max_length=50):
    """
    Truncate text to a maximum length and add ellipsis if needed.
    """
    return text if len(text) <= max_length else text[:max_length-3] + "..."

def get_shortest_variant(variants):
    """
    Get the shortest, most concise variant from a list of similar statements.
    """
    return min(variants, key=len)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    texts = {
        'A': request.form.get('source_a', ''),
        'B': request.form.get('source_b', ''),
        'C': request.form.get('source_c', '')
    }
    
    # Extract facts from each source
    all_facts = []
    fact_sources = {}
    unique_facts = []
    original_facts = {}  # Store all original variations of each fact
    
    for source, text in texts.items():
        facts = extract_facts(text)
        for fact in facts:
            # Find similar fact or add new one
            matching_fact = find_or_add_fact(fact, unique_facts)
            if matching_fact not in unique_facts:
                unique_facts.append(matching_fact)
            
            # Store the original version of this fact
            if matching_fact not in original_facts:
                original_facts[matching_fact] = []
            original_facts[matching_fact].append(fact)
            
            all_facts.append(matching_fact)
            if matching_fact not in fact_sources:
                fact_sources[matching_fact] = []
            if source not in fact_sources[matching_fact]:
                fact_sources[matching_fact].append(source)
    
    # Count fact frequencies
    fact_counts = Counter(all_facts)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Fact': [get_shortest_variant(set(original_facts[fact])) for fact in fact_counts.keys()],
        'Count': list(fact_counts.values()),
        'Original': list(fact_counts.keys()),
        'Sources': [', '.join(fact_sources[fact]) for fact in fact_counts.keys()],
        'Variations': [', '.join(set(original_facts[fact])) for fact in fact_counts.keys()]
    })
    
    # Create horizontal bar plot
    fig = px.bar(df, 
                 x='Count', 
                 y='Fact',
                 custom_data=['Original', 'Sources', 'Variations'],
                 title='Fact Frequency Across Sources',
                 orientation='h')  # Set orientation to horizontal
    
    # Update layout for better readability
    fig.update_layout(
        yaxis={'automargin': True},  # Automatically adjust margin for fact labels
        xaxis_title="Frequency",
        yaxis_title="Facts",
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
        height=max(400, len(df) * 40),  # Dynamic height based on number of facts
        showlegend=False
    )
    
    # Update bar appearance
    fig.update_traces(
        width=0.7,  # Make bars slightly thinner
        marker_color='rgb(55, 83, 109)'  # Use a professional color
    )
    
    return jsonify({
        'plot': fig.to_json(),
        'facts': df.to_dict('records')
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
