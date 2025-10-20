import json
from langchain_story import run_openai_pipeline, ScenesOutput, StoryIdeaOutput, TimelineOutput

# ============================================================
# 1️⃣ 안전 실행 유틸리티
# ============================================================
def safe_run(chain_func, *args, **kwargs):
    """LLM 호출 중 오류 발생 시 빈 결과 반환"""
    try:
        return chain_func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        return {"error": str(e)}


# ============================================================
# 2️⃣ OpenAI 기반 파이프라인 실행
# ============================================================
def run_pipeline(analysis_json: dict, duration: int, user_prompt: str):
    """
    OpenAI Vision 분석 결과(JSON dict)를 LangChain 파이프라인에 전달하여
    scenes → story → timeline 결과를 생성.
    """

    print("🧠 LangChain 파이프라인 실행 중...")

    try:
        result = run_openai_pipeline(analysis_json, duration, user_prompt)
    except Exception as e:
        print(f"⚠️ LangChain 실행 중 오류 발생: {e}")
        return {
            "scenes": ScenesOutput(scenes=[]),
            "story": StoryIdeaOutput(
                tone="", opening="", development="", closing="", key_message="",
                opening_sec=0, development_sec=0, closing_sec=0
            ),
            "timeline": TimelineOutput(story_summary="", timeline=[])
        }

    # 모델이 아닌 dict를 반환할 경우, 강제 변환
    if isinstance(result, dict):
        scenes = result.get("scenes", ScenesOutput(scenes=[]))
        story = result.get("story", StoryIdeaOutput(
            tone="", opening="", development="", closing="", key_message="",
            opening_sec=0, development_sec=0, closing_sec=0
        ))
        timeline = result.get("timeline", TimelineOutput(story_summary="", timeline=[]))
    else:
        # LLM 결과가 객체일 경우 그대로 유지
        scenes, story, timeline = result.scenes, result.story, result.timeline

    print("✅ LangChain 파이프라인 완료!")
    return {"scenes": scenes, "story": story, "timeline": timeline}


# ============================================================
# 3️⃣ 예시 실행 (직접 테스트용)
# ============================================================
if __name__ == "__main__":
    # OpenAI Vision 결과 예시 (테스트용)
    sample_json = {
        "videos": [
            {"filename": "ai_class.mp4", "description": "AI융합소프트웨어과 수업 중 학생들이 토론 중인 장면."}
        ],
        "images": [
            {"filename": "logo.png", "description": "폴리텍 대학 로고"}
        ],
        "audio": [],
        "user_prompt": "폴리텍 AI융합소프트웨어과 소개 영상을 만들어줘"
    }

    result = run_pipeline(sample_json, duration=30, user_prompt=sample_json["user_prompt"])

    print("\n=== Scenes ===")
    print(json.dumps(result["scenes"].model_dump(), ensure_ascii=False, indent=2))

    print("\n=== Story ===")
    print(json.dumps(result["story"].model_dump(), ensure_ascii=False, indent=2))

    print("\n=== Timeline ===")
    print(json.dumps(result["timeline"].model_dump(), ensure_ascii=False, indent=2))
