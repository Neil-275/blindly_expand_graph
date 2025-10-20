


def extract_numbers(t):
    """Recursively extract all numbers from nested tuples into a flat list."""
    numbers = []
    if isinstance(t, int):
        numbers.append(t)
    elif isinstance(t, (tuple, list)):
        for item in t:
            numbers.extend(extract_numbers(item))
    return numbers

def extract_strings(t):
    """Recursively extract all strings from nested tuples into a flat list."""
    strings = []
    if isinstance(t, str):
        strings.append(t)
    elif isinstance(t, (tuple, list)):
        for item in t:
            strings.extend(extract_strings(item))
    return strings

def extract_notations(t):
    """Recursively extract all notations (entities and relations) from nested tuples into a flat list."""
    notations = []
    t = str(t)
    for c in t:
        if c.isalpha():
            notations.append(c)
    return notations
