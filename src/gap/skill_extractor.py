def extract_skills(text, extractor, threshold=0.5):
    """
    Takes raw text, runs NER skill extraction,
    returns a list of unique skill strings above confidence threshold.
    """
    text  = clean_text(text)
    raw   = extractor(text)

    skills = []
    for entity in raw:
        if entity['score'] >= threshold:
            skill = entity['word'].strip().lower()
            # Skip subword tokens and very short extractions
        if skill.startswith('##') or len(skill) <= 1:
            continue
        if skill not in skills:
            skills.append(skill)
    return skills

# Test it
skills = extract_skills(sample_text, skill_extractor)
print("Clean skill list:", skills)