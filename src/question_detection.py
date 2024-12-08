import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def identify_questions(text):
    """
    Identify questions from the transcribed text using NLP techniques.
    Returns a list of sentences identified as questions.
    """
    # Process the text with spaCy
    doc = nlp(text)

    questions = []
    question_words = {"who", "what", "when", "where", "why", "how", "which", "whom", "whose", "is", "are", "can", "do", "does", "did", "could", "would", "will", "shall", "may", "might", "should"}

    for sent in doc.sents:
        # Clean sentence text
        sentence = sent.text.strip()

        # Skip short or incomplete sentences
        if len(sentence) < 3:
            continue

        # Check for explicit question structure:
        # 1. Ends with a question mark
        # 2. Starts with a question word or auxiliary verb
        if sentence.endswith("?") or sentence.split()[0].lower() in question_words:
            questions.append(sentence)

        # Advanced fallback: Identify questions based on dependency analysis
        else:
            if any(tok.dep_ == "aux" and tok.head.tag_ == "VB" for tok in sent):
                questions.append(sentence)

    return questions
