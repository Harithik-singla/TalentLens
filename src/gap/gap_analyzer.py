def match_skills(resume_skills, jd_skills, model, threshold=0.85, partial_threshold=0.6):
    """
    Compares JD skills against resume skills using embedding similarity.
    Returns matched, partial matches, and missing skills.
    """
    if not resume_skills or not jd_skills:
        return [], [], jd_skills

    # Embed all skills
    resume_embeddings = model.encode(resume_skills)
    jd_embeddings     = model.encode(jd_skills)

    # Cosine similarity matrix — shape (n_jd_skills, n_resume_skills)
    sim_matrix = cosine_similarity(jd_embeddings, resume_embeddings)

    matched  = []
    partial  = []
    missing  = []

    for i, jd_skill in enumerate(jd_skills):
        best_score = sim_matrix[i].max()
        best_match = resume_skills[sim_matrix[i].argmax()]

        if best_score >= threshold:
            matched.append({
                'jd_skill'     : jd_skill,
                'resume_match' : best_match,
                'score'        : round(float(best_score), 3)
            })
        elif best_score >= partial_threshold:
            partial.append({
                'jd_skill'     : jd_skill,
                'resume_match' : best_match,
                'score'        : round(float(best_score), 3)
            })
        else:
            missing.append(jd_skill)

    return matched, partial, missing

print("match_skills function defined")


def generate_gap_report(resume_text, jd_text, skill_extractor, embedding_model):
    """
    End-to-end gap analysis:
    1. Extract skills from resume and JD
    2. Match skills using embeddings
    3. Return structured gap report
    """
    # Extract skills
    resume_skills = extract_skills(resume_text, skill_extractor)
    jd_skills     = extract_skills(jd_text, skill_extractor)

    # Match skills
    matched, partial, missing = match_skills(
        resume_skills, jd_skills, embedding_model
    )

    # Build report
    report = {
        'resume_skills'  : resume_skills,
        'jd_skills'      : jd_skills,
        'matched'        : matched,
        'partial'        : partial,
        'missing'        : missing,
        'match_rate'     : round(len(matched) / len(jd_skills) * 100, 1) if jd_skills else 0
    }

    return report


def print_gap_report(report):
    print("=" * 55)
    print("GAP ANALYSIS REPORT")
    print("=" * 55)

    print(f"\nResume skills found : {report['resume_skills']}")
    print(f"JD required skills  : {report['jd_skills']}")
    print(f"Match rate          : {report['match_rate']}%")

    print("\n✓ MATCHED SKILLS:")
    if report['matched']:
        for m in report['matched']:
            print(f"  {m['jd_skill']:<20} ← {m['resume_match']} ({m['score']})")
    else:
        print("  None")

    print("\n~ PARTIAL MATCHES:")
    if report['partial']:
        for p in report['partial']:
            print(f"  {p['jd_skill']:<20} ≈ {p['resume_match']} ({p['score']})")
    else:
        print("  None")

    print("\n✗ MISSING SKILLS:")
    if report['missing']:
        for s in report['missing']:
            print(f"  {s}")
    else:
        print("  None — strong match!")

print("Gap report functions defined")