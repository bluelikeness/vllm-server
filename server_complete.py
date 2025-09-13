        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"파일 크기가 제한({MAX_UPLOAD_BYTES} bytes)를 초과했습니다")
        
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        supported_types = ['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'bmp']
        if file_extension not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_types)}"
            )
        
        image_types = ['png', 'jpg', 'jpeg', 'gif', 'bmp']
        is_image = file_extension in image_types
        
        conversation_id = get_or_create_conversation(conversation_id)
        
        if is_image:
            image_base64 = base64.b64encode(file_content).decode('utf-8')
            image_data_url = f"data:image/{file_extension};base64,{image_base64}"
            
            request = VisionRequest(
                message=message,
                image_data=image_data_url,
                conversation_id=conversation_id,
                max_tokens=max_tokens
            )
            
            return await analyze_vision(request)
        else:
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            request = MultimodalRequest(
                message=message,
                file_data=file_base64,
                file_type=file_extension,
                conversation_id=conversation_id,
                max_tokens=max_tokens
            )
            
            return await multimodal_analysis(request)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 처리 오류: {str(e)}")

# === 대화 관리 엔드포인트 ===
@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
    
    return {
        "conversation_id": conversation_id,
        "messages": active_conversations[conversation_id],
        "message_count": len(active_conversations[conversation_id])
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
            "last_message_preview": last_message["content"][:50] + "..." if last_message and len(last_message["content"]) > 50 else last_message["content"] if last_message else None
        })
    
    return {
        "conversations": conversations,
        "total_conversations": len(conversations)
    }

@app.get("/status/detailed")
async def detailed_status():
    gpu_status = get_gpu_status()
    vllm_stats = await get_vllm_stats()
    
    return {
        "server_info": {
            "version": "3.1.0",
            "engine": "vLLM",
            "multimodal_support": "enabled",
            "uptime_seconds": round(time.time() - server_start_time, 2),
            "python_version": "3.10+",
            "pytorch_version": torch.__version__ if torch else "N/A"
        },
        "model_info": {
            "loaded": vllm_engine is not None,
            "name": os.getenv("MODEL_NAME", "N/A"),
            "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "4096"),
            "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "8"),
            "limit_mm_per_prompt": {"image": 1}
        },
        "gpu_info": {
            "available": torch.cuda.is_available() if torch else False,
            "device_count": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0,
            "memory_used_gb": gpu_status["memory_used"],
            "memory_total_gb": gpu_status["memory_total"],
            "memory_usage_percent": round((gpu_status["memory_used"] / gpu_status["memory_total"]) * 100, 2) if gpu_status["memory_total"] > 0 else 0,
            "utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
        },
        "vllm_info": vllm_stats,
        "conversation_info": {
            "active_conversations": len(active_conversations),
            "total_messages": sum(len(msgs) for msgs in active_conversations.values())
        }
    }

if __name__ == "__main__":
    host = os.getenv("LLM_HOST", "0.0.0.0")
    port = int(os.getenv("LLM_PORT", "8001"))
    
    print(f"🚀 CloudLLM vLLM 멀티모달 서버 시작")
    print(f"📍 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"⚡ Engine: vLLM with Multimodal Support")
    print(f"🔢 Max Batch Size: {os.getenv('VLLM_MAX_NUM_SEQS', '8')}")
    print(f"📏 Max Model Length: {os.getenv('VLLM_MAX_MODEL_LEN', '4096')}")
    print(f"🖼️ 멀티모달: 이미지 + 텍스트 + 파일 지원")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )