from __future__ import annotations

from typing import Any, Dict, List


def system(text: str) -> Dict:
    return {"role": "system", "content": text}


def user_text(text: str) -> Dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


def user_image_base64(image_base64: str, image_format: str = "jpeg") -> Dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{image_base64}",
                },
            }
        ],
    }


def user_content(parts: List[Dict]) -> Dict:
    return {
        "role": "user",
        "content": parts,
    }


def text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(image_base64: str, image_format: str = "png") -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"},
    }


def build_messages(system_prompt: str, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [system(system_prompt), user_content(parts)]
