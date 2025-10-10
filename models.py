from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    lora_adapter: Optional[str] = None  # 사용할 LoRA 어댑터 이름


class VisionRequest(BaseModel):
    message: str
    image_data: str
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 512
    json_only: Optional[bool] = False
    lora_adapter: Optional[str] = None  # 사용할 LoRA 어댑터 이름


class MultimodalRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    image_list: Optional[List[str]] = None
    file_data: Optional[str] = None
    file_type: Optional[str] = None
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 512
    json_only: Optional[bool] = False
    lora_adapter: Optional[str] = None  # 사용할 LoRA 어댑터 이름


class MultiVisionRequest(BaseModel):
    message: str
    image_list: List[str]
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 512
    json_only: Optional[bool] = False
    lora_adapter: Optional[str] = None
    image_count: Optional[int] = None


class GenerationResponse(BaseModel):
    response: str
    conversation_id: str
    generation_time: float
    model_info: Dict[str, Any]
    response_json: Optional[Dict[str, Any]] = None
    response_is_json: bool = False


class ServerStatus(BaseModel):
    status: str
    model_loaded: bool
    active_conversations: int
    gpu_memory_used: float
    gpu_memory_total: float
    uptime: float
    vllm_stats: Dict[str, Any]


# ===== 대화 상태 =====
active_conversations: Dict[str, List[Dict[str, Any]]] = {}


def add_to_conversation(conversation_id: str, role: str, content: str, image_info: Optional[str] = None):
    if conversation_id not in active_conversations:
        active_conversations[conversation_id] = []
    message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    if image_info:
        message["image_info"] = image_info
    active_conversations[conversation_id].append(message)


def get_or_create_conversation(conversation_id: Optional[str]) -> str:
    import uuid
    if conversation_id and conversation_id in active_conversations:
        return conversation_id
    new_id = str(uuid.uuid4())
    active_conversations[new_id] = []
    return new_id


def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    if conversation_id not in active_conversations:
        return []
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in active_conversations[conversation_id]]


def format_chat_prompt(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_vision_prompt(message: str, json_only: bool = False) -> str:
    system_prompt = """이미지를 분석하고 사용자의 질문에 답해주세요. 
이미지의 내용을 정확하게 인식하고 상세히 설명해주세요.
한국어로 답변해주세요."""
    if json_only:
        system_prompt += """
\n응답은 반드시 다음 형식의 JSON으로만 제공해주세요:
{
    "analysis": "이미지 분석 결과",
    "details": ["세부사항1", "세부사항2", ...],
    "summary": "요약"
}"""
    parts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>",
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{message}<|im_end|>",
        "<|im_start|>assistant\n",
    ]
    return "\n".join(parts)


def format_multi_vision_prompt(message: str, image_count: int, json_only: bool = False) -> str:
    system_prompt = """여러 이미지를 순서대로 분석하고 사용자의 질문에 답해주세요.
각 이미지에 대해 관찰한 내용을 명시하고, 필요한 경우 비교하거나 종합하세요.
한국어로 답변해주세요."""
    if json_only:
        system_prompt += """

응답은 반드시 다음 형식의 JSON으로만 제공해주세요:
{
    "analysis": [
        {"image_index": 1, "details": ["항목1", "항목2"], "summary": "요약"},
        ...
    ],
    "overall_summary": "전체 요약"
}"""

    user_parts = ["<|vision_start|><|image_pad|><|vision_end|>" for _ in range(max(1, image_count))]
    user_payload = "\n".join(user_parts)
    parts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>",
        f"<|im_start|>user\n{user_payload}\n{message}<|im_end|>",
        "<|im_start|>assistant\n",
    ]
    return "\n".join(parts)
