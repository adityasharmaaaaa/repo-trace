from typing import Any

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    text_parts.append(text)
            elif isinstance(block, str):
                text_parts.append(block)

        return "".join(text_parts)
    return str(content)