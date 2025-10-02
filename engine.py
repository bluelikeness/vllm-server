import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
import re
from PIL import Image

import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.inputs import TextPrompt

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [GPU] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/vllm_engine.log')
    ]
)
logger = logging.getLogger(__name__)


MULTIMODAL_AVAILABLE = True
try:
    from vllm.multimodal import MultiModalDataDict  # noqa: F401
    MULTIMODAL_AVAILABLE = True
except Exception:
    try:
        from vllm.multimodal import MultiModalData  # noqa: F401
        MULTIMODAL_AVAILABLE = True
    except Exception:
        MULTIMODAL_AVAILABLE = False


vllm_engine: Optional[AsyncLLMEngine] = None
engine_config: Dict[str, Any] = {}


async def initialize_vllm_engine() -> bool:
    global vllm_engine, engine_config
    init_start = time.time()

    logger.info("🚀 vLLM 엔진 초기화 시작")

    try:
        # 안전 기본값: 텐서 병렬 미지정 시 1로 동작
        tensor_parallel_size = 1
        load_mode = os.getenv("VLLM_LOAD_MODE", "fp16").lower()

        # Determine mode key for presets
        mode_key = "FP16"
        if load_mode in ("int4", "4bit"):
            mode_key = "INT4"
        elif load_mode in ("int8", "8bit"):
            mode_key = "INT8"
        elif load_mode in ("bf16", "bfloat16"):
            mode_key = "BF16"

        def pick_env(name: str, default: Optional[str] = None) -> Optional[str]:
            """Pick value with priority: MODE_NAME > generic NAME > default.
            Example: name='MAX_MODEL_LEN' checks f"{mode_key}_{name}", then f"VLLM_{name}".
            """
            mode_var = f"{mode_key}_{name}"
            generic_var = f"VLLM_{name}"
            val = os.getenv(mode_var)
            if val is not None and str(val).strip() != "":
                return val
            val = os.getenv(generic_var)
            if val is not None and str(val).strip() != "":
                return val
            return default

        quant_method_env = pick_env("QUANTIZATION_METHOD", "").lower()
        kv_cache_dtype = pick_env("KV_CACHE_DTYPE", "")
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-32B-Instruct")

        # LoRA 어댑터 설정 읽기
        lora_adapters = os.getenv("LORA_ADAPTERS", "").strip()
        lora_adapter_names = os.getenv("LORA_ADAPTER_NAMES", "").strip()
        default_lora_adapter = os.getenv("DEFAULT_LORA_ADAPTER", "").strip()

        logger.info(f"📂 모델: {model_name}")
        logger.info(f"⚙️ 로드 모드: {load_mode}")
        logger.info(f"🔧 양자화 방법: {quant_method_env or '없음'}")
        logger.info(f"💾 KV 캐시 타입: {kv_cache_dtype or 'auto'}")

        # LoRA 어댑터 정보 로깅
        if lora_adapters:
            adapter_list = [a.strip() for a in lora_adapters.split(',') if a.strip()]
            name_list = [n.strip() for n in lora_adapter_names.split(',') if n.strip()] if lora_adapter_names else []
            logger.info(f"🎯 LoRA 어댑터: {len(adapter_list)}개 감지")
            for i, adapter_path in enumerate(adapter_list):
                adapter_name = name_list[i] if i < len(name_list) else f"adapter_{i+1}"
                logger.info(f"  - {adapter_name}: {adapter_path}")
            if default_lora_adapter:
                logger.info(f"🌟 기본 어댑터: {default_lora_adapter}")
        else:
            logger.info("🎯 LoRA 어댑터: 없음 (베이스 모델만 사용)")

        quantization = None
        # KV 캐시는 모든 모드에서 적용 가능하도록 공통 처리
        kv_cache_dtype_setting = "auto"
        if kv_cache_dtype and kv_cache_dtype.lower() in ("fp8", "int8"):
            kv_cache_dtype_setting = kv_cache_dtype.lower()

        if load_mode in ("int4", "4bit"):
            # 4bit 양자화 설정
            if quant_method_env in ("awq", "gptq", "bitsandbytes", "bnb"):
                quantization = "bitsandbytes" if quant_method_env in ("bitsandbytes", "bnb") else quant_method_env
            else:
                quantization = "bitsandbytes"  # 기본값을 bitsandbytes로 변경

            logger.info(f"🔢 적용된 양자화: {quantization}")
            logger.info(f"🗄️ KV 캐시 최적화: {kv_cache_dtype_setting}")
        elif load_mode in ("int8", "8bit"):
            # 8bit 양자화 설정
            if quant_method_env in ("bitsandbytes", "bnb", "gptq", "awq"):
                # INT8 기본은 bitsandbytes 권장
                quantization = "bitsandbytes" if quant_method_env in ("", "bitsandbytes", "bnb") else quant_method_env
            else:
                quantization = "bitsandbytes"

            logger.info(f"🔢 적용된 양자화: {quantization}")
            logger.info(f"🗄️ KV 캐시 최적화: {kv_cache_dtype_setting}")

        env_max_len = pick_env("MAX_MODEL_LEN")
        env_gpu_util = pick_env("GPU_MEMORY_UTILIZATION")

        # 4bit 모드에서 더 큰 컨텍스트와 더 많은 시퀀스 처리 가능
        if load_mode in ("int4", "4bit"):
            default_max_len = 12288  # 4bit에서 더 큰 컨텍스트
            default_gpu_util = 0.85   # 4bit에서 안정적 메모리 사용
        elif load_mode in ("int8", "8bit"):
            default_max_len = 10240  # 8bit에서 적절한 컨텍스트
            default_gpu_util = 0.90   # 8bit에서 다소 높은 활용률 가능
        else:  # fp16/bf16
            default_max_len = 8192    # FP16/BF16 기본값
            default_gpu_util = 0.92   # FP16/BF16에서 높은 메모리 사용

        chosen_max_len = int(env_max_len) if env_max_len else default_max_len
        chosen_gpu_util = float(env_gpu_util) if env_gpu_util else default_gpu_util

        # 4bit 모드에서 시퀀스 수 최적화
        if load_mode in ("int4", "4bit"):
            default_max_seqs = 48  # 4bit에서 더 많은 동시 처리
            default_block_size = 32  # 더 큰 블록 크기
        elif load_mode in ("int8", "8bit"):
            default_max_seqs = 36   # 8bit에서 동시 처리 확대
            default_block_size = 32
        else:  # fp16/bf16
            default_max_seqs = 24   # FP16/BF16 기본값
            default_block_size = 32

        max_num_seqs = int(pick_env("MAX_NUM_SEQS", str(default_max_seqs)))
        block_size = int(pick_env("BLOCK_SIZE", str(default_block_size)))
        swap_space = int(pick_env("SWAP_SPACE", "4"))

        # Tensor parallel size (support mode preset or generic)
        tensor_parallel_size = int(
            pick_env("TENSOR_PARALLEL_SIZE", str(int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))))
        )

        logger.info(f"📏 최대 모델 길이: {chosen_max_len}")
        logger.info(f"🎯 GPU 메모리 사용률: {chosen_gpu_util}")
        logger.info(f"🔗 텐서 병렬 크기: {tensor_parallel_size}")
        logger.info(f"📊 최대 시퀀스 수: {max_num_seqs}")
        logger.info(f"🧱 블록 크기: {block_size}")
        logger.info(f"💽 스왑 공간: {swap_space}GB")

        # trust_remote_code 설정 (.env 제어 가능; 기본 True)
        trc_val = os.getenv("VLLM_TRUST_REMOTE_CODE", "1").strip().lower()
        trust_remote_code = trc_val in ("1", "true", "yes", "y")

        # vLLM 엔진 인자 구성
        engine_args_dict = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": chosen_max_len,
            "max_num_seqs": max_num_seqs,
            "block_size": block_size,
            "swap_space": swap_space,
            "gpu_memory_utilization": chosen_gpu_util,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": False,
            "limit_mm_per_prompt": {"image": 1} if MULTIMODAL_AVAILABLE else {"image": 0},
        }

        # LoRA 어댑터 설정 추가
        lora_paths_list: List[str] = []
        if lora_adapters:
            adapter_list = [a.strip() for a in lora_adapters.split(',') if a.strip()]
            name_list = [n.strip() for n in lora_adapter_names.split(',') if n.strip()] if lora_adapter_names else []

            if adapter_list:
                # LoRA 어댑터 경로들 설정
                engine_args_dict["enable_lora"] = True
                engine_args_dict["lora_modules"] = []
                lora_paths_list = []

                for i, adapter_path in enumerate(adapter_list):
                    adapter_name = name_list[i] if i < len(name_list) else f"adapter_{i+1}"
                    engine_args_dict["lora_modules"].append({
                        "name": adapter_name,
                        "path": adapter_path
                    })
                    lora_paths_list.append(adapter_path)

                # 기본 어댑터 설정
                if default_lora_adapter and default_lora_adapter in [n.get("name") for n in engine_args_dict["lora_modules"]]:
                    engine_args_dict["default_lora_adapter"] = default_lora_adapter

                logger.info(f"✅ LoRA 지원 활성화: {len(adapter_list)}개 어댑터")

        # 양자화 설정 추가
        if quantization:
            engine_args_dict["quantization"] = quantization

        # KV 캐시 dtype 설정 (모든 모드에서 적용; vLLM 버전에 따라 지원 여부 확인)
        if kv_cache_dtype_setting != "auto":
            try:
                engine_args_dict["kv_cache_dtype"] = kv_cache_dtype_setting
                logger.info(f"💾 KV 캐시 타입 설정: {kv_cache_dtype_setting}")
            except Exception as e:
                logger.warning(f"⚠️ KV 캐시 타입 설정 실패 (vLLM 버전 호환성): {e}")

        # vLLM 버전 호환: 지원되지 않는 키가 있으면 제거/대체하며 재시도
        attempt_dict = dict(engine_args_dict)
        for _ in range(5):
            try:
                engine_args = AsyncEngineArgs(**attempt_dict)
                break
            except TypeError as e:
                msg = str(e)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad_key = m.group(1)
                logger.warning(f"⚠️ AsyncEngineArgs에서 지원되지 않는 인자 감지: {bad_key} → 제거/대체 후 재시도")
                # lora_modules 미지원: lora_paths로 대체 시도
                if bad_key == "lora_modules" and lora_paths_list:
                    attempt_dict.pop("lora_modules", None)
                    attempt_dict["lora_paths"] = list(lora_paths_list)
                else:
                    attempt_dict.pop(bad_key, None)
        else:
            # 반복 시도 후에도 실패 시 마지막으로 예외 발생
            engine_args = AsyncEngineArgs(**attempt_dict)

        logger.info(f"🖼️ 멀티모달 지원: {'활성화' if MULTIMODAL_AVAILABLE else '비활성화'}")
        logger.info(f"⚡ 모드별 최적화: {'4bit/8bit 메모리 효율성' if load_mode in ('int4', '4bit', 'int8', '8bit') else 'FP16/BF16 고성능'}")
        logger.info(f"💾 KV 캐시 데이터 타입 설정(요청): {kv_cache_dtype or 'auto'} → 적용: {kv_cache_dtype_setting}")

        logger.info("🔄 vLLM 엔진 생성 중...")
        engine_create_start = time.time()

        try:
            vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            engine_create_time = time.time() - engine_create_start
            logger.info(f"✅ vLLM 엔진 생성 완료 - {engine_create_time:.2f}초")
        except Exception as e:
            logger.warning(f"⚠️ 첫 번째 시도 실패: {str(e)}")
            if "bitsandbytes" in str(e) or "quantizer" in str(e).lower():
                logger.info("🔄 양자화 없이 재시도...")
                quantization = None
                engine_args.quantization = None
                engine_args.gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
                vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                logger.info("✅ 양자화 없이 엔진 생성 성공")
            else:
                raise

        engine_config.update({
            "model": engine_args.model,
            "tensor_parallel_size": engine_args.tensor_parallel_size,
            "max_model_len": engine_args.max_model_len,
            "max_num_seqs": engine_args.max_num_seqs,
            "gpu_memory_utilization": engine_args.gpu_memory_utilization,
            "load_mode": load_mode,
            "quantization": quantization or "none",
            "kv_cache_dtype": kv_cache_dtype or None,
        })

        # GPU 상태 로깅
        gpu_status = get_gpu_status()
        logger.info(f"🖥️ GPU 메모리: {gpu_status['memory_used']:.2f}GB / {gpu_status['memory_total']:.2f}GB 사용")

        init_time = time.time() - init_start
        logger.info(f"🎯 vLLM 엔진 초기화 완료 - 총 {init_time:.2f}초")

        return True
    except Exception as e:
        init_time = time.time() - init_start
        logger.error(f"❌ vLLM 엔진 초기화 실패 - {init_time:.2f}초: {e}")
        return False


async def get_vllm_stats() -> Dict[str, Any]:
    global vllm_engine
    if vllm_engine is None:
        return {"error": "vLLM 엔진이 초기화되지 않았습니다"}
    return {
        "engine_status": "running",
        "pending_requests": 0,
        "running_requests": 0,
    }


def get_gpu_status() -> Dict[str, float]:
    try:
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024 ** 3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return {"memory_used": round(memory_used, 2), "memory_total": round(memory_total, 2)}
        else:
            return {"memory_used": 0.0, "memory_total": 0.0}
    except Exception as e:
        print(f"GPU 상태 확인 오류: {e}")
        return {"memory_used": 0.0, "memory_total": 0.0}


async def generate_with_vllm(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    images: Optional[List[Image.Image]] = None,
    lora_adapter: Optional[str] = None,  # 🆕 요청별 LoRA 어댑터 선택
) -> Tuple[str, Dict[str, Any]]:
    global vllm_engine
    if vllm_engine is None:
        raise RuntimeError("vLLM 엔진이 초기화되지 않았습니다")

    request_id = random_uuid()[:8]  # 짧은 요청 ID
    start_time = time.time()
    timings: Dict[str, Any] = {}
    original_prompt = prompt

    logger.info(f"🎯 [GPU-{request_id}] 텍스트 생성 시작")
    logger.info(f"📝 [GPU-{request_id}] 프롬프트 길이: {len(prompt)}자")
    logger.info(f"🖼️ [GPU-{request_id}] 이미지 포함: {'예' if images and len(images) > 0 else '아니오'}")
    logger.info(f"🎛️ [GPU-{request_id}] 최대 토큰: {max_tokens}, 온도: {temperature}")

    # LoRA 어댑터 정보 로깅
    if lora_adapter:
        logger.info(f"🎯 [GPU-{request_id}] 요청된 LoRA 어댑터: {lora_adapter}")
    else:
        default_adapter = os.getenv("DEFAULT_LORA_ADAPTER", "")
        if default_adapter:
            lora_adapter = default_adapter
            logger.info(f"🌟 [GPU-{request_id}] 기본 LoRA 어댑터 사용: {lora_adapter}")
        else:
            logger.info(f"🎯 [GPU-{request_id}] LoRA 어댑터: 사용 안함 (베이스 모델)")

    eff_tokens = max(1, int(min(max_tokens, int(os.getenv("MAX_TOKENS_CAP", "512")))))
    if eff_tokens != max_tokens:
        logger.info(f"⚙️ [GPU-{request_id}] 토큰 수 조정: {max_tokens} -> {eff_tokens}")

    sampling_params = SamplingParams(
        max_tokens=eff_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
        stop_token_ids=[],
    )

    # 🆕 LoRA 어댑터가 지정된 경우 sampling_params에 추가
    if lora_adapter:
        try:
            # vLLM 0.2.0+ 에서 지원하는 LoRA 요청별 지정
            sampling_params.lora_request = lora_adapter
            logger.info(f"✅ [GPU-{request_id}] LoRA 어댑터 설정 완료: {lora_adapter}")
        except AttributeError:
            # 구버전 vLLM에서는 지원하지 않을 수 있음
            logger.warning(f"⚠️ [GPU-{request_id}] 현재 vLLM 버전에서 요청별 LoRA 지정 미지원")
        except Exception as e:
            logger.warning(f"⚠️ [GPU-{request_id}] LoRA 어댑터 설정 실패: {e}")

    use_multimodal = False
    if images and MULTIMODAL_AVAILABLE:
        try:
            logger.info(f"🖼️ [GPU-{request_id}] 멀티모달 프롬프트 준비 중...")
            if "<|image_pad|>" not in prompt and "<|vision_start|>" not in prompt:
                prompt = f"<|vision_start|><|image_pad|><|vision_end|>\n{prompt}"
                logger.info(f"📄 [GPU-{request_id}] 비전 태그 추가됨")

            # 이미지 정보 로깅
            if images[0]:
                img_size = images[0].size
                img_mode = images[0].mode
                logger.info(f"🖼️ [GPU-{request_id}] 이미지 정보: {img_size}, 모드: {img_mode}")

            prompt = TextPrompt({"prompt": prompt, "multi_modal_data": {"image": images[0]} if images else {}})
            use_multimodal = True
            logger.info(f"✅ [GPU-{request_id}] 멀티모달 프롬프트 준비 완료")
        except Exception as e:
            logger.warning(f"⚠️ [GPU-{request_id}] 멀티모달 준비 실패, 텍스트 모드로 전환: {e}")
            prompt = original_prompt
            use_multimodal = False

    # GPU 상태 로깅
    gpu_status = get_gpu_status()
    logger.info(f"🖥️ [GPU-{request_id}] 생성 전 GPU 메모리: {gpu_status['memory_used']:.2f}GB / {gpu_status['memory_total']:.2f}GB")

    t_gen_start = time.time()
    logger.info(f"🚀 [GPU-{request_id}] vLLM 생성 시작...")

    try:
        results_generator = vllm_engine.generate(prompt, sampling_params, request_id)
    except Exception as e:
        logger.error(f"❌ [GPU-{request_id}] 생성 시작 실패: {e}")
        if use_multimodal:
            logger.info(f"🔄 [GPU-{request_id}] 텍스트 모드로 재시도...")
            prompt = original_prompt
            results_generator = vllm_engine.generate(prompt, sampling_params, request_id)
        else:
            raise

    final_output = None
    try:
        logger.info(f"⏳ [GPU-{request_id}] 응답 스트리밍 중...")
        async for request_output in results_generator:
            final_output = request_output
        logger.info(f"✅ [GPU-{request_id}] 응답 스트리밍 완료")
    except Exception as e:
        logger.error(f"❌ [GPU-{request_id}] 스트리밍 실패: {e}")
        if use_multimodal:
            logger.info(f"🔄 [GPU-{request_id}] 텍스트 모드로 재시도...")
            prompt = original_prompt
            results_generator = vllm_engine.generate(prompt, sampling_params, request_id)
            async for request_output in results_generator:
                final_output = request_output
        else:
            raise

    timings["generation_ms"] = round((time.time() - t_gen_start) * 1000, 1)

    if final_output is None:
        logger.error(f"❌ [GPU-{request_id}] 생성 결과가 없습니다")
        raise RuntimeError("생성 결과가 없습니다")

    response_text = "".join(o.text for o in final_output.outputs)
    timings["total_ms"] = round((time.time() - start_time) * 1000, 1)
    timings["tokens_generated"] = len(final_output.outputs[0].token_ids) if final_output.outputs else 0

    # 타입 안전한 토큰/초 계산
    generation_time_seconds = float(timings["generation_ms"]) / 1000.0 if timings.get("generation_ms", 0) > 0 else 0
    timings["tokens_per_second"] = round(
        float(timings["tokens_generated"]) / generation_time_seconds, 1
    ) if generation_time_seconds > 0 else 0

    # 생성 완료 로깅
    logger.info(f"🎯 [GPU-{request_id}] 텍스트 생성 완료")
    logger.info(f"⏱️ [GPU-{request_id}] 생성 시간: {timings['generation_ms']}ms")
    logger.info(f"📊 [GPU-{request_id}] 생성 토큰: {timings['tokens_generated']}개")
    logger.info(f"🚀 [GPU-{request_id}] 속도: {timings['tokens_per_second']} tokens/sec")
    logger.info(f"📤 [GPU-{request_id}] 응답 길이: {len(response_text)}자")
    logger.info(f"💡 [GPU-{request_id}] 응답 미리보기: {response_text[:100]}...")

    # 최종 GPU 상태 로깅
    final_gpu_status = get_gpu_status()
    logger.info(f"🖥️ [GPU-{request_id}] 생성 후 GPU 메모리: {final_gpu_status['memory_used']:.2f}GB / {final_gpu_status['memory_total']:.2f}GB")

    return response_text.strip(), timings

