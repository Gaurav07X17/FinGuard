"""
Simple token check helper. In production use a robust auth system.
"""
def check_token(provided: str, expected: str) -> bool:
    return provided == expected
