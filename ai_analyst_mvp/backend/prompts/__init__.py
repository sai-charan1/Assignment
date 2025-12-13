from pathlib import Path

def load_prompt(name):
    p = Path(__file__).parent / name
    return p.read_text()
