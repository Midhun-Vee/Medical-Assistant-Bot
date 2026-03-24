import re

# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

def mask_pii(query: str) -> str:
    query = re.sub(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', '[EMAIL]', query)
    query = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', query)
    query = re.sub(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', '[AADHAAR]', query)
    query = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN]', query)
    return query


# ---------------------------------------------------------------------------
# Prompt injection detection
# ---------------------------------------------------------------------------

INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "forget everything",
    "you are now",
    "act as",
    "pretend to be",
    "jailbreak",
    "dan mode",
    "developer mode",
    "override instructions",
    "disregard instructions",
    "new system prompt",
    "do anything now",
]

def has_injection(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in INJECTION_PHRASES)


# ---------------------------------------------------------------------------
# Banned words
# ---------------------------------------------------------------------------

BANNED_WORDS = [
    "suicide", "kill myself", "self harm", "self-harm",
    "make meth", "synthesize drugs", "synthesise drugs",
    "how to kill", "poison someone",
]

def has_banned_words(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in BANNED_WORDS)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def check_input(query: str) -> dict:
    """
    Returns {"blocked": False, "query": ...} if safe,
    or {"blocked": True, "reason": ...} if not.
    """

    if len(query.strip()) < 2:
        return {"blocked": True, "reason": "Query is too short."}

    if len(query) > 1500:
        return {"blocked": True, "reason": "Query is too long. Please shorten your message."}

    if has_injection(query):
        return {"blocked": True, "reason": "Your message contains disallowed instructions."}

    if has_banned_words(query):
        return {"blocked": True, "reason": "Your message contains content that cannot be processed. If you are in distress, please contact emergency services."}

    cleaned = mask_pii(query)

    return {"blocked": False, "query": cleaned}