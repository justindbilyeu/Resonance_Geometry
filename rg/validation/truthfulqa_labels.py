import re

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def label_hallucination(model_output, correct_answers, incorrect_answers):
    """
    Returns:
      0 = clean (matches any correct answer, fuzzy substring)
      1 = hallucinated (matches any incorrect answer, fuzzy substring)
     -1 = borderline/ambiguous (exclude from λ_ref)
    """
    out = _norm(model_output)
    if not out:
        return -1

    for c in (correct_answers or []):
        c = _norm(c)
        if c and (c in out or out in c):
            return 0

    for ic in (incorrect_answers or []):
        ic = _norm(ic)
        if ic and (ic in out or out in ic):
            return 1

    return -1
