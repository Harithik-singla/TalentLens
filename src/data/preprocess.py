import re
from transformers import BertTokenizer

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove non-ASCII characters (PDF artifacts, weird bullets, fancy quotes)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # Normalize all whitespace (newlines, tabs, multiple spaces) to single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    # Lowercase
    text = text.lower()
    return text

def tokenize_texts(df, tokenizer, max_length):
    """
    Tokenize resume and JD separately.
    Returns a dict of tensors ready for the bi-encoder.
    """
    resume_encodings = tokenizer(
        df['resume_clean'].tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    jd_encodings = tokenizer(
        df['jd_clean'].tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    labels = torch.tensor(df['label_id'].tolist(), dtype=torch.long)

    return {
        'resume_input_ids'      : resume_encodings['input_ids'],
        'resume_attention_mask' : resume_encodings['attention_mask'],
        'jd_input_ids'          : jd_encodings['input_ids'],
        'jd_attention_mask'     : jd_encodings['attention_mask'],
        'labels'                : labels
    }