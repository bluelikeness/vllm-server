#!/usr/bin/env python3
"""
LoRA 어댑터 관리 유틸리티
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LoRAManager:
    """LoRA 어댑터 관리 클래스"""
    
    def __init__(self, adapters_dir: str = None):
        # 환경변수 우선, 없으면 기본값 사용
        if adapters_dir is None:
            adapters_dir = os.getenv("LORA_ADAPTERS_HOME", "/data/huggingface_models/lora_adapters")
        
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)  # parents=True 추가
        self.config_file = self.adapters_dir / "adapters.json"
        self.adapters_config = self._load_config()
    
    def _load_config(self) -> Dict:
        """어댑터 설정 로드"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"어댑터 설정 로드 실패: {e}")
        
        return {
            "adapters": [],
            "default_adapter": None,
            "last_updated": None
        }
    
    def _save_config(self):
        """어댑터 설정 저장"""
        import datetime
        self.adapters_config["last_updated"] = datetime.datetime.now().isoformat()
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.adapters_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"어댑터 설정 저장 실패: {e}")
    
    def add_adapter(self, name: str, path: str, description: str = "", is_default: bool = False) -> bool:
        """LoRA 어댑터 추가"""
        adapter_path = Path(path)
        
        # 경로 검증
        if not adapter_path.exists():
            logger.error(f"어댑터 경로가 존재하지 않음: {path}")
            return False
        
        # 중복 확인
        for adapter in self.adapters_config["adapters"]:
            if adapter["name"] == name:
                logger.warning(f"동일한 이름의 어댑터가 이미 존재: {name}")
                return False
        
        # 어댑터 추가
        adapter_info = {
            "name": name,
            "path": str(adapter_path.absolute()),
            "description": description,
            "created_at": None
        }
        
        # 생성 시간 확인 (adapter_config.json 파일이 있는 경우)
        config_file = adapter_path / "adapter_config.json"
        if config_file.exists():
            try:
                import datetime
                stat = config_file.stat()
                adapter_info["created_at"] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            except Exception:
                pass
        
        self.adapters_config["adapters"].append(adapter_info)
        
        # 기본 어댑터 설정
        if is_default or not self.adapters_config["default_adapter"]:
            self.adapters_config["default_adapter"] = name
        
        self._save_config()
        logger.info(f"✅ LoRA 어댑터 추가됨: {name}")
        return True
    
    def remove_adapter(self, name: str) -> bool:
        """LoRA 어댑터 제거"""
        for i, adapter in enumerate(self.adapters_config["adapters"]):
            if adapter["name"] == name:
                # 기본 어댑터인 경우 null로 설정
                if self.adapters_config["default_adapter"] == name:
                    self.adapters_config["default_adapter"] = None
                
                del self.adapters_config["adapters"][i]
                self._save_config()
                logger.info(f"✅ LoRA 어댑터 제거됨: {name}")
                return True
        
        logger.warning(f"어댑터를 찾을 수 없음: {name}")
        return False
    
    def list_adapters(self) -> List[Dict]:
        """어댑터 목록 반환"""
        return self.adapters_config["adapters"]
    
    def set_default_adapter(self, name: str) -> bool:
        """기본 어댑터 설정"""
        for adapter in self.adapters_config["adapters"]:
            if adapter["name"] == name:
                self.adapters_config["default_adapter"] = name
                self._save_config()
                logger.info(f"✅ 기본 어댑터 설정됨: {name}")
                return True
        
        logger.warning(f"어댑터를 찾을 수 없음: {name}")
        return False
    
    def get_env_config(self) -> tuple[str, str, str]:
        """vLLM .env 파일용 설정 생성"""
        adapters = self.adapters_config["adapters"]
        
        if not adapters:
            return "", "", ""
        
        # 어댑터 경로들
        adapter_paths = [adapter["path"] for adapter in adapters]
        adapter_names = [adapter["name"] for adapter in adapters]
        default_adapter = self.adapters_config["default_adapter"] or ""
        
        return (
            ",".join(adapter_paths),
            ",".join(adapter_names), 
            default_adapter
        )
    
    def update_env_file(self, env_file_path: str = ".env") -> bool:
        """vLLM .env 파일 업데이트"""
        env_path = Path(env_file_path)
        
        if not env_path.exists():
            logger.error(f".env 파일이 존재하지 않음: {env_file_path}")
            return False
        
        try:
            # 현재 .env 파일 읽기
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # LoRA 설정 생성
            adapter_paths, adapter_names, default_adapter = self.get_env_config()
            
            # 기존 LoRA 설정 제거 및 새 설정 추가
            new_lines = []
            skip_next = False
            
            for line in lines:
                if line.startswith("LORA_"):
                    continue  # LoRA 관련 라인 건너뛰기
                new_lines.append(line)
            
            # 새 LoRA 설정 추가
            lora_section = [
                f"LORA_ADAPTERS={adapter_paths}\n",
                f"LORA_ADAPTER_NAMES={adapter_names}\n", 
                f"DEFAULT_LORA_ADAPTER={default_adapter}\n"
            ]
            
            # LoRA 설정을 MODEL_NAME 다음에 삽입
            insert_idx = -1
            for i, line in enumerate(new_lines):
                if line.startswith("MODEL_NAME=") or "로컬 모델 경로" in line:
                    insert_idx = i + 1
                    break
            
            if insert_idx > 0:
                new_lines[insert_idx:insert_idx] = ["\n# LoRA 어댑터 설정 (자동 생성)\n"] + lora_section + ["\n"]
            else:
                new_lines.extend(["\n# LoRA 어댑터 설정 (자동 생성)\n"] + lora_section)
            
            # 파일 쓰기
            with open(env_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            logger.info(f"✅ .env 파일 업데이트 완료: {env_file_path}")
            return True
            
        except Exception as e:
            logger.error(f".env 파일 업데이트 실패: {e}")
            return False

def scan_for_adapters(training_dir: str = "../training") -> List[Dict]:
    """학습 디렉토리에서 LoRA 어댑터 자동 스캔"""
    training_path = Path(training_dir)
    adapters = []
    
    if not training_path.exists():
        return adapters
    
    # models 및 checkpoints 디렉토리 스캔
    scan_dirs = [
        training_path / "models",
        training_path / "checkpoints", 
        training_path / "lora_outputs"
    ]
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
            
        for item in scan_dir.iterdir():
            if item.is_dir():
                # adapter_config.json 파일이 있는지 확인
                config_file = item / "adapter_config.json"
                if config_file.exists():
                    adapters.append({
                        "name": item.name,
                        "path": str(item.absolute()),
                        "description": f"Auto-detected from {scan_dir.name}",
                        "auto_detected": True
                    })
    
    return adapters

if __name__ == "__main__":
    # CLI 인터페이스
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA 어댑터 관리")
    parser.add_argument("action", choices=["list", "add", "remove", "set-default", "scan", "update-env"])
    parser.add_argument("--name", help="어댑터 이름")
    parser.add_argument("--path", help="어댑터 경로")
    parser.add_argument("--description", help="어댑터 설명", default="")
    parser.add_argument("--default", action="store_true", help="기본 어댑터로 설정")
    
    args = parser.parse_args()
    
    manager = LoRAManager()
    
    if args.action == "list":
        adapters = manager.list_adapters()
        if adapters:
            print("📋 등록된 LoRA 어댑터:")
            for adapter in adapters:
                default_mark = " (기본)" if adapter["name"] == manager.adapters_config["default_adapter"] else ""
                print(f"  - {adapter['name']}{default_mark}: {adapter['path']}")
        else:
            print("📋 등록된 LoRA 어댑터가 없습니다.")
    
    elif args.action == "add":
        if not args.name or not args.path:
            print("❌ --name과 --path가 필요합니다.")
        else:
            manager.add_adapter(args.name, args.path, args.description, args.default)
    
    elif args.action == "remove":
        if not args.name:
            print("❌ --name이 필요합니다.")
        else:
            manager.remove_adapter(args.name)
    
    elif args.action == "set-default":
        if not args.name:
            print("❌ --name이 필요합니다.")
        else:
            manager.set_default_adapter(args.name)
    
    elif args.action == "scan":
        print("🔍 LoRA 어댑터 자동 스캔 중...")
        found_adapters = scan_for_adapters()
        if found_adapters:
            print(f"📦 {len(found_adapters)}개의 어댑터를 발견했습니다:")
            for adapter in found_adapters:
                print(f"  - {adapter['name']}: {adapter['path']}")
                choice = input(f"    추가하시겠습니까? (y/N): ").lower()
                if choice == 'y':
                    manager.add_adapter(
                        adapter['name'], 
                        adapter['path'], 
                        adapter['description']
                    )
        else:
            print("📦 LoRA 어댑터를 찾을 수 없습니다.")
    
    elif args.action == "update-env":
        manager.update_env_file()
