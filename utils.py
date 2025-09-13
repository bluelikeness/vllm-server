import os
import io
import time
import json
import base64
from typing import Any, Dict, Optional
from PIL import Image

MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1280"))


# ===== JSON 헬퍼 =====
def strip_code_fences(text: str) -> str:
    lines = text.splitlines()
    return "\n".join([ln for ln in lines if not ln.strip().startswith("```")])


def sanitize_json_like(s: str) -> str:
    import re
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|[^:])//.*$", r"\1", s, flags=re.M)
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\r\n", "\n")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def extract_first_json_object(text: str) -> Optional[str]:
    s = text
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end != -1:
        return s[start:end + 1]
    last = s.rfind('}')
    if last == -1 or last <= start:
        return None
    return s[start:last + 1]


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    stripped = strip_code_fences(text)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    candidate = extract_first_json_object(stripped)
    if candidate is None:
        return None
    for attempt in (candidate, sanitize_json_like(candidate)):
        try:
            return json.loads(attempt)
        except Exception:
            continue
    return None


# ===== 이미지 처리 =====
def _resize_image(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        max_side = max(w, h)
        if max_side <= MAX_IMAGE_SIDE:
            return img
        scale = MAX_IMAGE_SIDE / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.LANCZOS)
    except Exception:
        return img


def process_image_data(image_data: str) -> Image.Image:
    if image_data.startswith("data:"):
        base64_data = image_data.split(",")[1]
    else:
        base64_data = image_data
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = _resize_image(image)
    w, h = image.size
    if w < 32 or h < 32:
        new_size = max(32, max(w, h))
        new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
        offset = ((new_size - w) // 2, (new_size - h) // 2)
        new_image.paste(image, offset)
        image = new_image
    return image
