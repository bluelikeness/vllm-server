#!/usr/bin/env python3
"""
vLLM 서버 멀티모달 기능 테스트 스크립트
"""

import requests
import base64
import json
import time
from pathlib import Path

# 서버 설정
BASE_URL = "http://localhost:8001"

def test_health():
    """서버 상태 확인"""
    print("🔍 서버 상태 확인...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data['status']}")
            print(f"📦 모델: {data['model_name']}")
            print(f"🖼️ 멀티모달: {'지원됨' if data.get('engine_type') == 'vLLM' else '불명'}")
            return True
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return False

def test_text_generation():
    """텍스트 생성 테스트"""
    print("\n📝 텍스트 생성 테스트...")
    
    payload = {
        "message": "안녕하세요! 간단한 인사말을 해주세요.",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 텍스트 생성 성공! (소요시간: {elapsed:.2f}초)")
            print(f"📄 응답: {data['response'][:100]}...")
            print(f"⏱️ 생성 시간: {data['generation_time']:.2f}초")
            return True
        else:
            print(f"❌ 텍스트 생성 실패: {response.status_code}")
            print(f"오류: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 텍스트 생성 오류: {e}")
        return False

def encode_image_to_base64(image_path):
    """이미지를 base64로 인코딩"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 이미지 인코딩 실패: {e}")
        return None

def test_vision():
    """비전 기능 테스트"""
    print("\n🖼️ 비전 기능 테스트...")
    
    # 테스트용 간단한 이미지 생성 (1x1 픽셀 PNG)
    # 실제 사용 시에는 실제 이미지 파일을 사용하세요
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    payload = {
        "message": "이 이미지에 대해 설명해주세요.",
        "image_data": f"data:image/png;base64,{test_image_b64}",
        "max_tokens": 150,
        "json_only": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/vision", json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 비전 분석 성공! (소요시간: {elapsed:.2f}초)")
            print(f"📄 응답: {data['response'][:100]}...")
            print(f"⏱️ 생성 시간: {data['generation_time']:.2f}초")
            print(f"🔧 엔진: {data['model_info'].get('engine', 'N/A')}")
            return True
        else:
            print(f"❌ 비전 분석 실패: {response.status_code}")
            print(f"오류: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 비전 분석 오류: {e}")
        return False

def test_multimodal():
    """멀티모달 기능 테스트"""
    print("\n🔗 멀티모달 기능 테스트...")
    
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    payload = {
        "message": "이미지와 텍스트를 함께 분석해주세요.",
        "image_data": f"data:image/png;base64,{test_image_b64}",
        "max_tokens": 150
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/multimodal", json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 멀티모달 분석 성공! (소요시간: {elapsed:.2f}초)")
            print(f"📄 응답: {data['response'][:100]}...")
            print(f"⏱️ 생성 시간: {data['generation_time']:.2f}초")
            return True
        else:
            print(f"❌ 멀티모달 분석 실패: {response.status_code}")
            print(f"오류: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 멀티모달 분석 오류: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 vLLM 서버 멀티모달 기능 테스트 시작")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    if not test_health():
        print("❌ 서버가 실행되지 않았거나 응답하지 않습니다.")
        return
    
    # 2. 텍스트 생성 테스트
    test_text_generation()
    
    # 3. 비전 기능 테스트
    test_vision()
    
    # 4. 멀티모달 기능 테스트
    test_multimodal()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료!")

if __name__ == "__main__":
    main()
