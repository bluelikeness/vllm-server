"""
향상된 로깅 시스템
- 환경변수로 로그 레벨 제어 (LOG_LEVEL)
- DEBUG 모드에서 상세한 입출력 정보 로깅
- 구조화된 로그 포맷
- 이미지 정보, 프롬프트, 응답 등 상세 정보 기록
"""

import os
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
import base64
from PIL import Image
import io


class ColoredFormatter(logging.Formatter):
    """컬러 로그 포맷터 (터미널에서 보기 좋게)"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (선택)
    
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    
    # 환경변수에서 로그 레벨 읽기 (기본값: INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 포맷 설정
    detailed_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    simple_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    format_str = detailed_format if level == logging.DEBUG else simple_format
    
    # 컬러 포맷터 적용
    colored_formatter = ColoredFormatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        plain_formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    
    return logger


class RequestLogger:
    """요청별 상세 로깅 클래스"""
    
    def __init__(self, logger: logging.Logger, request_id: str):
        self.logger = logger
        self.request_id = request_id
        self.start_time = datetime.now()
        
    def log_request_start(self, endpoint: str, **kwargs):
        """요청 시작 로깅"""
        self.logger.info(f"🎯 [{self.request_id}] ===== 요청 시작: {endpoint} =====")
        self.logger.info(f"⏰ [{self.request_id}] 시작 시각: {self.start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        if self.logger.level <= logging.DEBUG:
            for key, value in kwargs.items():
                self.logger.debug(f"📋 [{self.request_id}] {key}: {value}")
    
    def log_prompt(self, prompt: str, max_length: int = 500):
        """프롬프트 로깅"""
        prompt_length = len(prompt)
        self.logger.info(f"📝 [{self.request_id}] 프롬프트 길이: {prompt_length}자")
        
        if self.logger.level <= logging.DEBUG:
            if prompt_length <= max_length:
                self.logger.debug(f"💬 [{self.request_id}] 전체 프롬프트:\n{'-'*80}\n{prompt}\n{'-'*80}")
            else:
                preview = prompt[:max_length]
                self.logger.debug(f"💬 [{self.request_id}] 프롬프트 미리보기 ({max_length}자):\n{'-'*80}\n{preview}\n... (생략) ...\n{'-'*80}")
    
    def log_image(self, image: Image.Image, image_data: Optional[str] = None):
        """이미지 정보 로깅"""
        self.logger.info(f"🖼️ [{self.request_id}] 이미지 포함: 예")
        self.logger.info(f"📐 [{self.request_id}] 이미지 크기: {image.size[0]}x{image.size[1]}")
        self.logger.info(f"🎨 [{self.request_id}] 이미지 모드: {image.mode}")
        
        if self.logger.level <= logging.DEBUG and image_data:
            # Base64 데이터 길이 로깅
            if image_data.startswith('data:'):
                header, data = image_data.split(',', 1) if ',' in image_data else (image_data, '')
                data_length = len(data)
                self.logger.debug(f"📦 [{self.request_id}] Base64 데이터 길이: {data_length}자")
                self.logger.debug(f"🏷️ [{self.request_id}] 데이터 헤더: {header}")
            
            # 이미지 파일 크기 추정
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'PNG')
            img_size_kb = len(img_byte_arr.getvalue()) / 1024
            self.logger.debug(f"💾 [{self.request_id}] 이미지 예상 크기: {img_size_kb:.2f}KB")
    
    def log_generation_params(self, **params):
        """생성 파라미터 로깅"""
        self.logger.info(f"⚙️ [{self.request_id}] 생성 파라미터:")
        for key, value in params.items():
            self.logger.info(f"   - {key}: {value}")
    
    def log_response(self, response: str, max_length: int = 500):
        """응답 로깅"""
        response_length = len(response)
        self.logger.info(f"📤 [{self.request_id}] 응답 길이: {response_length}자")
        
        if self.logger.level <= logging.DEBUG:
            if response_length <= max_length:
                self.logger.debug(f"💡 [{self.request_id}] 전체 응답:\n{'-'*80}\n{response}\n{'-'*80}")
            else:
                preview = response[:max_length]
                self.logger.debug(f"💡 [{self.request_id}] 응답 미리보기 ({max_length}자):\n{'-'*80}\n{preview}\n... (생략) ...\n{'-'*80}")
        else:
            # INFO 레벨에서는 짧은 미리보기만
            preview_length = 100
            preview = response[:preview_length]
            if response_length > preview_length:
                preview += "..."
            self.logger.info(f"💡 [{self.request_id}] 응답 미리보기: {preview}")
    
    def log_json_response(self, json_data: Optional[Dict[str, Any]]):
        """JSON 응답 로깅"""
        if json_data:
            self.logger.info(f"📋 [{self.request_id}] JSON 파싱: 성공")
            
            if self.logger.level <= logging.DEBUG:
                try:
                    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                    self.logger.debug(f"📊 [{self.request_id}] JSON 데이터:\n{'-'*80}\n{json_str}\n{'-'*80}")
                except Exception as e:
                    self.logger.debug(f"⚠️ [{self.request_id}] JSON 직렬화 실패: {e}")
        else:
            self.logger.info(f"📋 [{self.request_id}] JSON 파싱: 실패 또는 텍스트 응답")
    
    def log_timings(self, timings: Dict[str, Any]):
        """타이밍 정보 로깅"""
        self.logger.info(f"⏱️ [{self.request_id}] 성능 지표:")
        for key, value in timings.items():
            if 'ms' in key:
                self.logger.info(f"   - {key}: {value}ms")
            elif 'per_second' in key:
                self.logger.info(f"   - {key}: {value}")
            else:
                self.logger.info(f"   - {key}: {value}")
    
    def log_gpu_status(self, memory_used: float, memory_total: float, stage: str = ""):
        """GPU 상태 로깅"""
        memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
        stage_label = f" ({stage})" if stage else ""
        self.logger.info(f"🖥️ [{self.request_id}] GPU 메모리{stage_label}: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_percent:.1f}%)")
    
    def log_lora_adapter(self, adapter_name: Optional[str], is_default: bool = False):
        """LoRA 어댑터 로깅"""
        if adapter_name:
            prefix = "🌟" if is_default else "🎯"
            label = "기본 LoRA 어댑터" if is_default else "LoRA 어댑터"
            self.logger.info(f"{prefix} [{self.request_id}] {label}: {adapter_name}")
        else:
            self.logger.info(f"🎯 [{self.request_id}] LoRA 어댑터: 사용 안함 (베이스 모델)")
    
    def log_error(self, error: Exception, context: str = ""):
        """에러 로깅"""
        context_label = f" - {context}" if context else ""
        self.logger.error(f"❌ [{self.request_id}] 오류 발생{context_label}: {type(error).__name__}: {str(error)}")
        
        if self.logger.level <= logging.DEBUG:
            import traceback
            self.logger.debug(f"🔍 [{self.request_id}] 스택 트레이스:\n{traceback.format_exc()}")
    
    def log_request_end(self, success: bool = True):
        """요청 종료 로깅"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        status_icon = "✅" if success else "❌"
        status_text = "성공" if success else "실패"
        
        self.logger.info(f"{status_icon} [{self.request_id}] ===== 요청 {status_text} =====")
        self.logger.info(f"⏰ [{self.request_id}] 종료 시각: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self.logger.info(f"⏱️ [{self.request_id}] 총 소요 시간: {duration:.3f}초")


def log_multimodal_content(logger: logging.Logger, request_id: str, 
                          has_image: bool, has_file: bool, file_type: Optional[str] = None):
    """멀티모달 컨텐츠 로깅"""
    content_parts = []
    if has_image:
        content_parts.append("이미지")
    if has_file and file_type:
        content_parts.append(f"{file_type.upper()} 파일")
    
    if content_parts:
        content_str = " + ".join(content_parts)
        logger.info(f"📦 [{request_id}] 멀티모달 컨텐츠: {content_str}")
    else:
        logger.info(f"📝 [{request_id}] 컨텐츠: 텍스트만")


def log_conversation_context(logger: logging.Logger, request_id: str, 
                             conversation_id: str, message_count: int):
    """대화 컨텍스트 로깅"""
    logger.info(f"💬 [{request_id}] 대화 ID: {conversation_id}")
    logger.info(f"📚 [{request_id}] 대화 메시지 수: {message_count}")


# 전역 로거 인스턴스
app_logger = setup_logger('vllm_app', log_file='/tmp/vllm_app.log')
engine_logger = setup_logger('vllm_engine', log_file='/tmp/vllm_engine.log')
