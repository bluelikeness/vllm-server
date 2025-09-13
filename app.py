import os
import time
import base64
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from . import engine
from .models import (
    ChatRequest,
    VisionRequest,
    MultimodalRequest,
    GenerationResponse,
    active_conversations,
    get_or_create_conversation,
    get_conversation_messages,
    add_to_conversation,
    format_chat_prompt,
    format_vision_prompt,
)
from .utils import process_image_data, try_parse_json
from .file_io import process_uploaded_file


server_start_time = time.time()
MAX_TOKENS_CAP = int(os.getenv("MAX_TOKENS_CAP", "512"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # .env 로드 (필요 시 환경변수 적용)
    try:
        load_dotenv()
    except Exception:
        pass
    print("🔄 vLLM 서버 시작 중...")
    success = await engine.initialize_vllm_engine()
    if not success:
        print("❌ vLLM 엔진 초기화 실패로 서버를 종료합니다.")
        raise RuntimeError("vLLM 엔진 초기화 실패")
    print("✅ vLLM 서버 시작 완료!")
    yield
    print("🔄 vLLM 서버 종료 중...")
    active_conversations.clear()
    print("✅ vLLM 서버 종료 완료!")


app = FastAPI(
    title="CloudLLM vLLM Server",
    description="vLLM 기반 고성능 멀티모달 LLM 서버",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "CloudLLM vLLM Server v3.1 - 고성능 멀티모달 LLM 서비스",
        "engine": "vLLM",
        "version": "3.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health")
async def health_check():
    loaded = engine.vllm_engine is not None
    gpu_status = engine.get_gpu_status()
    try:
        vllm_stats = await engine.get_vllm_stats() if loaded else {}
    except Exception:
        vllm_stats = {}
    device_info = "N/A"
    if torch.cuda.is_available() and loaded:
        device_info = f"cuda:{torch.cuda.current_device()}"
    elif loaded:
        device_info = "cpu"
    return {
        "status": "healthy" if loaded else "loading",
        "model_loaded": loaded,
        "model_name": os.getenv("MODEL_NAME", "N/A"),
        "engine_type": "vLLM",
        "device": device_info,
        "gpu_memory_used": gpu_status["memory_used"],
        "gpu_memory_total": gpu_status["memory_total"],
        "vllm_stats": vllm_stats,
    "engine_config": engine.engine_config,
        "active_conversations": len(active_conversations),
        "uptime": round(time.time() - server_start_time, 2),
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_text_endpoint(request: ChatRequest):
    if engine.vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM 엔진이 초기화되지 않았습니다")
    start_time = time.time()
    conversation_id = get_or_create_conversation(request.conversation_id)
    messages = get_conversation_messages(conversation_id)
    messages.append({"role": "user", "content": request.message})
    try:
        prompt = format_chat_prompt(messages)
        response_text, gen_timings = await engine.generate_with_vllm(
            prompt=prompt, 
            max_tokens=request.max_tokens, 
            temperature=request.temperature,
            lora_adapter=request.lora_adapter  # 🆕 LoRA 어댑터 전달
        )
        add_to_conversation(conversation_id, "user", request.message)
        add_to_conversation(conversation_id, "assistant", response_text)
        generation_time = time.time() - start_time
        t_json0 = time.time()
        parsed = try_parse_json(response_text)
        json_parse_ms = round((time.time() - t_json0) * 1000, 1)
        timings_api = {
            "endpoint_total_ms": round(generation_time * 1000, 1),
            "json_parse_ms": json_parse_ms,
            **gen_timings,
        }
        return GenerationResponse(
            response=response_text,
            conversation_id=conversation_id,
            generation_time=round(generation_time, 2),
            model_info={
                "model_name": os.getenv("MODEL_NAME", "unknown"),
                "engine": "vLLM",
                "max_tokens": min(request.max_tokens or 512, MAX_TOKENS_CAP),
                "temperature": request.temperature,
                "lora_adapter": request.lora_adapter or os.getenv("DEFAULT_LORA_ADAPTER", "base"),  # 🆕 사용된 어댑터 정보
                "timings": timings_api,
            },
            response_json=parsed,
            response_is_json=parsed is not None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"생성 오류: {str(e)}")


@app.post("/vision", response_model=GenerationResponse)
async def analyze_vision(request: VisionRequest):
    if engine.vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM 엔진이 초기화되지 않았습니다")
    if not engine.MULTIMODAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="멀티모달 기능이 사용할 수 없습니다")
    start_time = time.time()
    conversation_id = get_or_create_conversation(request.conversation_id)
    try:
        image = process_image_data(request.image_data)
        prompt = format_vision_prompt(request.message, request.json_only)
        response_text, gen_timings = await engine.generate_with_vllm(
            prompt=prompt, 
            max_tokens=request.max_tokens, 
            temperature=0.7, 
            images=[image],
            lora_adapter=request.lora_adapter  # 🆕 LoRA 어댑터 전달
        )
        add_to_conversation(conversation_id, "user", request.message, "이미지 포함")
        add_to_conversation(conversation_id, "assistant", response_text)
        generation_time = time.time() - start_time
        t_json0 = time.time()
        parsed = try_parse_json(response_text) if request.json_only else None
        json_parse_ms = round((time.time() - t_json0) * 1000, 1)
        timings_api = {
            "endpoint_total_ms": round(generation_time * 1000, 1),
            "json_parse_ms": json_parse_ms,
            **gen_timings,
        }
        return GenerationResponse(
            response=response_text,
            conversation_id=conversation_id,
            generation_time=round(generation_time, 2),
            model_info={
                "model_name": os.getenv("MODEL_NAME", "unknown"),
                "engine": "vLLM+Vision",
                "max_tokens": min(request.max_tokens or 512, MAX_TOKENS_CAP),
                "temperature": 0.7,
                "lora_adapter": request.lora_adapter or os.getenv("DEFAULT_LORA_ADAPTER", "base"),  # 🆕 사용된 어댑터 정보
                "multimodal": True,
                "timings": timings_api,
            },
            response_json=parsed,
            response_is_json=parsed is not None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 오류: {str(e)}")


@app.post("/multimodal", response_model=GenerationResponse)
async def multimodal_analysis(request: MultimodalRequest):
    if engine.vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM 엔진이 초기화되지 않았습니다")
    start_time = time.time()
    conversation_id = get_or_create_conversation(request.conversation_id)
    try:
        enhanced_message = request.message
        images = None

        if request.image_data:
            if engine.MULTIMODAL_AVAILABLE:
                try:
                    image = process_image_data(request.image_data)
                    images = [image]
                    enhanced_message += "\n\n[이미지가 제공되었습니다. 이미지를 분석해 주세요.]"
                except Exception as e:
                    enhanced_message += f"\n\n[이미지 처리 실패: {str(e)}]"
            else:
                # 멀티모달 미지원 시 이미지는 무시
                pass

        if request.file_data and request.file_type:
            try:
                file_content = base64.b64decode(request.file_data)
                file_text = process_uploaded_file(file_content, request.file_type)
                if file_text:
                    enhanced_message += f"\n\n=== {request.file_type.upper()} 파일 내용 ===\n{file_text}"
                else:
                    enhanced_message += f"\n\n[{request.file_type} 파일 처리 실패]"
            except Exception as e:
                enhanced_message += f"\n\n[파일 처리 실패: {str(e)}]"

        messages = [{"role": "user", "content": enhanced_message}]
        prompt = format_chat_prompt(messages)
        response_text, gen_timings = await engine.generate_with_vllm(
            prompt=prompt, max_tokens=request.max_tokens, temperature=0.7, images=images
        )
        context_info: List[str] = []
        if request.image_data:
            context_info.append("이미지")
        if request.file_data and request.file_type:
            context_info.append(f"{request.file_type} 파일")
        context_str = " + ".join(context_info) if context_info else "텍스트만"
        add_to_conversation(conversation_id, "user", request.message, context_str)
        add_to_conversation(conversation_id, "assistant", response_text)
        generation_time = time.time() - start_time
        t_json0 = time.time()
        parsed = try_parse_json(response_text) if request.json_only else None
        json_parse_ms = round((time.time() - t_json0) * 1000, 1)
        timings_api = {
            "endpoint_total_ms": round(generation_time * 1000, 1),
            "json_parse_ms": json_parse_ms,
            **gen_timings,
        }
        return GenerationResponse(
            response=response_text,
            conversation_id=conversation_id,
            generation_time=round(generation_time, 2),
            model_info={
                "model_name": os.getenv("MODEL_NAME", "unknown"),
                "engine": "vLLM+Multimodal",
                "max_tokens": min(request.max_tokens or 512, MAX_TOKENS_CAP),
                "temperature": 0.7,
                "multimodal": True,
                "has_image": request.image_data is not None,
                "has_file": request.file_data is not None,
                "file_type": request.file_type,
                "timings": timings_api,
            },
            response_json=parsed,
            response_is_json=parsed is not None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"멀티모달 분석 오류: {str(e)}")


@app.post("/upload", response_model=GenerationResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(512),
):
    if engine.vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM 엔진이 초기화되지 않았습니다")
    try:
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"파일 크기가 제한({MAX_UPLOAD_BYTES} bytes)를 초과했습니다")
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        supported_types = ['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'bmp']
        if file_extension not in supported_types:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_types)}")
        image_types = ['png', 'jpg', 'jpeg', 'gif', 'bmp']
        is_image = file_extension in image_types
        conversation_id = get_or_create_conversation(conversation_id)
        if is_image:
            image_base64 = base64.b64encode(file_content).decode('utf-8')
            image_data_url = f"data:image/{file_extension};base64,{image_base64}"
            req = VisionRequest(message=message, image_data=image_data_url, conversation_id=conversation_id, max_tokens=max_tokens)
            return await analyze_vision(req)
        else:
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            req = MultimodalRequest(message=message, file_data=file_base64, file_type=file_extension, conversation_id=conversation_id, max_tokens=max_tokens)
            return await multimodal_analysis(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 처리 오류: {str(e)}")


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
    return {
        "conversation_id": conversation_id,
        "messages": active_conversations[conversation_id],
        "message_count": len(active_conversations[conversation_id]),
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
    del active_conversations[conversation_id]
    return {"message": f"대화 {conversation_id}가 삭제되었습니다"}


@app.get("/conversations")
async def list_conversations():
    conversations = []
    for conv_id, messages in active_conversations.items():
        last_message = messages[-1] if messages else None
        conversations.append({
            "conversation_id": conv_id,
            "message_count": len(messages),
            "last_message_time": last_message["timestamp"] if last_message else None,
            "last_message_preview": last_message["content"][:50] + "..." if last_message and len(last_message["content"]) > 50 else last_message["content"] if last_message else None,
        })
    return {"conversations": conversations, "total_conversations": len(conversations)}


@app.get("/status/detailed")
async def detailed_status():
    gpu_status = engine.get_gpu_status()
    vllm_stats = await engine.get_vllm_stats()
    return {
        "server_info": {
            "version": "3.1.0",
            "engine": "vLLM",
            "multimodal_support": "enabled" if engine.MULTIMODAL_AVAILABLE else "disabled",
            "uptime_seconds": round(time.time() - server_start_time, 2),
            "python_version": "3.10+",
            "pytorch_version": torch.__version__ if torch else "N/A",
        },
        "model_info": {
            "loaded": engine.vllm_engine is not None,
            "name": os.getenv("MODEL_NAME", "N/A"),
            "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "6144"),
            "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "8"),
            "limit_mm_per_prompt": {"image": 1} if engine.MULTIMODAL_AVAILABLE else None,
        },
        "gpu_info": {
            "available": torch.cuda.is_available() if torch else False,
            "device_count": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0,
            "memory_used_gb": gpu_status["memory_used"],
            "memory_total_gb": gpu_status["memory_total"],
            "memory_usage_percent": round((gpu_status["memory_used"] / gpu_status["memory_total"]) * 100, 2) if gpu_status["memory_total"] > 0 else 0,
            "utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        },
        "vllm_info": vllm_stats,
        "engine_config": engine.engine_config,
        "conversation_info": {
            "active_conversations": len(active_conversations),
            "total_messages": sum(len(msgs) for msgs in active_conversations.values()),
        },
        "features": {
            "text_generation": True,
            "vision_analysis": engine.MULTIMODAL_AVAILABLE,
            "multimodal_analysis": engine.MULTIMODAL_AVAILABLE,
            "file_upload": True,
            "supported_file_types": ['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'bmp'],
        },
    }


@app.get("/lora/adapters")
async def list_lora_adapters():
    """등록된 LoRA 어댑터 목록 반환"""
    try:
        from .lora_manager import LoRAManager
        manager = LoRAManager()
        adapters = manager.list_adapters()
        default_adapter = manager.adapters_config.get("default_adapter")
        
        return {
            "adapters": adapters,
            "default_adapter": default_adapter,
            "total_count": len(adapters)
        }
    except Exception as e:
        return {
            "adapters": [],
            "default_adapter": None,
            "total_count": 0,
            "error": str(e)
        }


@app.post("/lora/set-default")
async def set_default_lora_adapter(adapter_name: str):
    """기본 LoRA 어댑터 설정"""
    try:
        from .lora_manager import LoRAManager
        manager = LoRAManager()
        success = manager.set_default_adapter(adapter_name)
        if success:
            # .env 파일 업데이트
            manager.update_env_file()
            return {"success": True, "message": f"기본 어댑터가 '{adapter_name}'으로 설정되었습니다"}
        else:
            raise HTTPException(status_code=404, detail=f"어댑터 '{adapter_name}'을 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"기본 어댑터 설정 실패: {str(e)}")


@app.get("/models")
async def list_models():
    """현재 로드된 모델 정보"""
    base_model = os.getenv("MODEL_NAME", "unknown")
    default_lora = os.getenv("DEFAULT_LORA_ADAPTER", "")
    
    return {
        "base_model": base_model,
        "default_lora_adapter": default_lora,
        "model_loaded": engine.vllm_engine is not None,
        "multimodal_available": engine.MULTIMODAL_AVAILABLE
    }


if __name__ == "__main__":
    host = os.getenv("LLM_HOST", "0.0.0.0")
    port = int(os.getenv("LLM_PORT", "8001"))
    workers = int(os.getenv("LLM_SERVER_WORKERS", "1"))
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=False,
    )


def run():
    host = os.getenv("LLM_HOST", "0.0.0.0")
    port = int(os.getenv("LLM_PORT", "8001"))
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info").lower(), access_log=True)
