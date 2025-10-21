import os
import cv2
import json
import base64
from openai import OpenAI
from local_langchain import run_pipeline
from movie import render_shorts_from_timeline
from dotenv import load_dotenv

# ==============================
# 0. 설정
# ==============================
MEDIA_DIR = "media"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 환경변수나 직접 API Key 지정
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==============================
# 1. OpenAI Vision 이미지 분석
# ==============================
def analyze_image_openai(image_path: str) -> dict:
    print(f"🖼️ 이미지 분석 중: {image_path}")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "이 이미지를 자세히 분석해줘. "
        "무엇이 보이는지, 사람/사물/텍스트가 있으면 구체적으로 설명하고, "
        "이미지의 전체적인 분위기나 상황을 요약해줘."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},  # ✅ 수정: input_text → text
                        {
                            "type": "image_url",  # ✅ 수정: input_image → image_url
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                    }
                        },
                    ],
                }
            ],
            temperature=0.2,
        )

        description = response.choices[0].message.content
        print(f"✅ 분석 완료: {os.path.basename(image_path)}")
        return {"type": "image", "filename": os.path.basename(image_path), "description": description}
    except Exception as e:
        print(f"⚠️ 이미지 분석 오류: {e}")
        return {"type": "image", "filename": os.path.basename(image_path), "description": None}


# ==============================
# 2. OpenAI Vision 기반 영상 분석
# ==============================
def analyze_video_openai(video_path: str, num_frames: int = 5) -> dict:
    print(f"🎥 영상 분석 중: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(num_frames):
        idx = int(total * (i + 1) / (num_frames + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf).decode("utf-8")
        frames.append(b64)
    cap.release()

    prompt = (
        "이 영상의 장면 변화와 동작을 설명해줘. "
        "프레임 간의 움직임, 등장하는 인물/사물, 전환된 장면들 요약해줘."
    )

    try:
        content = [{"type": "text", "text": prompt}]
        for b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        description = response.choices[0].message.content
        print(f"✅ 영상 분석 완료: {os.path.basename(video_path)}")
        return {"type": "video", "filename": os.path.basename(video_path), "description": description}
    except Exception as e:
        print(f"⚠️ 영상 분석 오류: {e}")
        return {"type": "video", "filename": os.path.basename(video_path), "description": None}



# ==============================
# 3. 모든 미디어 파일 통합 분석
# ==============================
def analyze_all_media() -> list:
    print("📂 media 폴더에서 파일을 불러옵니다...")
    files = [
        os.path.join(MEDIA_DIR, f)
        for f in os.listdir(MEDIA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))
    ]

    print(f"📦 총 {len(files)}개의 파일 감지됨")
    all_results = []

    for path in files:
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            result = analyze_image_openai(path)
        else:
            result = analyze_video_openai(path)
        all_results.append(result)

    output_path = os.path.join(RESULT_DIR, "analysis_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ 모든 미디어 분석 완료 → {output_path}")
    return all_results

def normalize_openai_analysis(openai_results: list, user_prompt: str) -> dict:
    """OpenAI Vision 분석 리스트를 LangChain 파이프라인용 dict로 변환"""
    videos, images, audio = [], [], []

    for item in openai_results:
        if item.get("type") == "video":
            videos.append({
                "filename": item["filename"],
                "description": item.get("description", "")
            })
        elif item.get("type") == "image":
            images.append({
                "filename": item["filename"],
                "description": item.get("description", "")
            })
        elif item.get("type") == "audio":
            audio.append({
                "filename": item["filename"],
                "description": item.get("description", "")
            })

    return {
        "videos": videos,
        "images": images,
        "audio": audio,
        "user_prompt": user_prompt
    }



# ==============================
# 4. LangChain + MoviePy 통합 파이프라인
# ==============================
def main():
    print("🚀 OpenAI Vision 기반 통합 파이프라인 실행 시작!")
    combined_analysis = analyze_all_media()
    
    prompt = "내가 지금 한국 폴리텍 대학 AI융합소프트웨어과에 대한 학과 소개 영상을 만들고 싶어. 학과의 특징과 장점에 대해서 알려주는 영상을 만들어줘"
    print("\n🧠 LangChain 파이프라인 실행 중...")
        
        
        # ✅ analysis_result.json에서 분석 결과 불러오기
    with open(os.path.join(RESULT_DIR, "analysis_result.json"), "r", encoding="utf-8") as f:
        combined_analysis = json.load(f)

    
    
    combined_analysis = normalize_openai_analysis(combined_analysis, prompt)
    result = run_pipeline(combined_analysis, duration=30, user_prompt = prompt)

    # Pydantic-safe JSON 변환
    def make_json_serializable(obj):
        result = run_pipeline(combined_analysis, duration=30, user_prompt=prompt)

# ✅ LangChain Pydantic 결과를 JSON/dict로 변환
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif isinstance(result, dict):
        # 내부 Pydantic 객체까지 안전 변환
        def recursive_dump(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif isinstance(obj, dict):
                return {k: recursive_dump(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_dump(v) for v in obj]
            else:
                return obj
        result = recursive_dump(result)

    print("\n🎬 MoviePy 영상 렌더링 중...")
    render_shorts_from_timeline(result, output_path=os.path.join(RESULT_DIR, "final_shorts.mp4"))

    
    
    
    print("✅ 최종 영상 생성 완료!")



# ==============================
# 5. 실행 진입점
# ==============================
if __name__ == "__main__":
    main()
