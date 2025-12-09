import json
from server import build_prompt

def test_prompt_builder():
    sp = "системная инструкция"
    u = "подбери низкую группировку"
    p = build_prompt(sp, u)
    assert "assistant" in p
