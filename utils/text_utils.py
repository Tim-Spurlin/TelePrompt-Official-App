def clean_text(text):
    """Cleans the input text by removing unnecessary whitespace and punctuation."""
    import re
    text = text.strip()  # Remove leading and trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

def format_text(text, max_length=80):
    """Formats the input text to fit within a specified maximum length."""
    if len(text) <= max_length:
        return text
    else:
        return '\n'.join(text[i:i + max_length] for i in range(0, len(text), max_length))

def analyze_text(text):
    """Analyzes the input text and returns basic statistics."""
    word_count = len(text.split())
    char_count = len(text)
    return {
        'word_count': word_count,
        'char_count': char_count,
        'text': text
 }