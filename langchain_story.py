import json, math, os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnableSequence


# ============================================================
# 1️⃣ Pydantic Schemas
# ============================================================
class SceneItem(BaseModel):
    scene_id: int
    summary: str
    highlight: str


class ScenesOutput(BaseModel):
    scenes: List[SceneItem] = Field(default_factory=list)


class StoryIdeaOutput(BaseModel):
    tone: str
    opening: str
    development: str
    closing: str
    key_message: str
    opening_sec: int
    development_sec: int
    closing_sec: int


class TimelineItem(BaseModel):
    type: str
    filename: Optional[str] = None
    text: Optional[str] = None
    start: float
    end: float


class TimelineOutput(BaseModel):
    story_summary: str
    timeline: List[TimelineItem]

class EmotionOutput(BaseModel):
    emotion_story: str

class HookOutput(BaseModel):
    hook_line: str


# ============================================================
# 2️⃣ LangChain 모델 초기화
# ============================================================
base_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="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")
scene_llm = base_llm.with_structured_output(ScenesOutput)
story_llm = base_llm.with_structured_output(StoryIdeaOutput)
emotion_llm = base_llm.with_structured_output(EmotionOutput)
hook_llm = base_llm.with_structured_output(HookOutput)
timeline_llm = base_llm.with_structured_output(TimelineOutput)


# ============================================================
# 3️⃣ Helper Functions
# ============================================================
def scenes_to_json(obj: ScenesOutput) -> str:
    return json.dumps([s.model_dump() for s in obj.scenes], ensure_ascii=False, indent=2)


def story_to_json(obj: StoryIdeaOutput) -> str:
    return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2)


def split_duration(total: int) -> dict:
    """총 길이를 도입, 전개, 결말로 분할"""
    opening = int(total * 0.3)
    development = int(total * 0.5)
    closing = total - opening - development
    return {
        "opening_sec": opening,
        "development_sec": development,
        "closing_sec": closing
    }


# ============================================================
#  Debuging Functions
# ============================================================
def save_debug_timeline(timeline_data, prefix="timeline_debug", folder="results"):
    """
    LangChain이 생성한 타임라인 데이터를 별도로 JSON으로 저장합니다.
    파일명 예시: results/timeline_debug_2025-10-18_1735.json
    """
    os.makedirs(folder, exist_ok=True)
    debug_path = os.path.join(folder, f"{prefix}.json")

    def make_serializable(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj
        
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(make_serializable(timeline_data), f, ensure_ascii=False, indent=2)
        print(f"🐞 [DEBUG] 타임라인 JSON 저장 완료 → {debug_path}")
    except Exception as e:
        print(f"⚠️ [DEBUG 저장 실패] {e}")



# ============================================================
# 4️⃣ Prompts (OpenAI Vision 기반)
# ============================================================

# 🎬 Scene 분석
scene_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 영상 분석가입니다. JSON 형식만 출력하세요."),
    ("user", """
아래는 OpenAI Vision 모델로 분석한 미디어 데이터입니다.

{analysis_json}

사용자 요청:
"{user_prompt}"

각 영상(videos)의 description을 참고하여 장면을 3~8개로 나누고,
각 장면마다 아래 항목을 작성하세요:
- scene_id (1부터 시작)
- summary (한 문장 요약)
- highlight (핵심 키워드)

출력은 반드시 JSON 형식만 허용됩니다.
예:
{{"scenes":[{{"scene_id":1,"summary":"...","highlight":"..."}}]}}
""")
])

# 🧠 Story 구성
story_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 스토리텔러입니다. JSON 형식만 출력하세요."),
    ("user", """
다음은 장면(scene) 요약입니다:

{scenes_json}

사용자 요청:
"{user_prompt}"

이 장면들을 기반으로 아래 항목을 구성하세요:
- tone: 전체 영상 분위기 (예: 감동적, 유쾌함, 정보전달형 등)
- opening: 도입부 주요 내용
- development: 전개부 주요 내용
- closing: 마무리 내용
- key_message: 핵심 메시지

또한 전체 duration({duration}초)을 기준으로
opening_sec, development_sec, closing_sec을 분배하세요.

출력은 반드시 JSON 형식이어야 합니다.
""")
])

# 🎭 감정형 스토리텔링
emotion_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 감정 중심의 스토리텔러입니다. JSON 형식만 출력하세요."),
    ("user", """
아래 스토리 아이디어를 바탕으로 시청자의 감정을 자극하는 스토리를 만드세요.

입력:
{story_idea_json}

요구사항:
- 인물의 감정과 상황을 구체적으로 묘사
- 감동, 유머, 긴장 중 하나 이상 포함
- 너무 설명적이지 말고 자연스럽게 풀어내세요.

출력 형식(JSON):
{{ "emotion_story": "..." }}
""")
])

# ⚡ 3초 후킹 문장 생성
hook_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 영상 쇼츠 기획자입니다. JSON 형식만 출력하세요."),
    ("user", """
아래 스토리를 보고 시청자의 스크롤을 멈추게 할 한 문장을 만드세요.

입력 스토리:
{story_idea_json}

조건:
- 12자 이내, 강렬하거나 의문형
- 단조로운 설명 금지
- 감정/반전/놀라움 중 하나 포함

출력 형식(JSON):
{{ "hook_line": "..." }}
""")
])

# 🧩 Timeline 생성
timeline_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 영상 편집자입니다. JSON 형식만 출력하세요."),
    ("user", """
아래는 영상과 이미지 분석 결과입니다:

{analysis_json}

다음은 구성된 스토리입니다:
{story_idea_json}
     
- 감정 서사: {emotion_json}
- 후킹 문장: {hook_json}

총 영상 길이는 {duration}초입니다.

각 장면을 기반으로 타임라인을 구성하세요.
     
타임라인은 다음의 세 요소를 조화롭게 결합해야 합니다:
1. 스토리 개요와 메시지 (story_idea_json)
2. 감정 흐름 (emotion_json)
3. 도입 후킹 (hook_json)

세 요소가 자연스럽게 연결되도록 각 장면의 자막과 전환을 조정하세요.

요구사항:
1) opening, development, closing 구간을 시간 비율에 맞게 배치할 것
2) 첫 3초에 hook_line 기반의 영상 장면 추가
3) 중간부는 emotion_story 감정선이 느껴지게 구성, 전체 스토리의 흐름가 일치할 것
4) 전체 길이는 {duration}초 이내로 구성
5) 각 구간에는 반드시 1개 이상의 video 또는 image 아이템이 포함되어야 함
6) 모든 video, image 뒤에는 반드시 subtitle 아이템을 추가할 것  
   - subtitle.text에는 해당 장면의 대사나 요약 문장을 자연스럽게 작성할 것  
   - 한글로 1줄(15~25자 이내)로 작성  
   - 장면 내용과 자연스럽게 이어지게 할 것  
7) 필요할 경우 subtitle은 영상 중간에도 여러 번 나올 수 있음
8) 오디오(audio)는 전체 영상에 걸쳐 1개만 포함
9) 각 컷은 3~7초 사이로 구성
10) JSON은 다음 형식이어야 함:

```json
{{
  "story_summary": "...",
  "timeline": [
    {{"type": "video", "filename": "scene1.mp4", "text": "장면 설명", "start": 0.0, "end": 5.0}},
    {{"type": "subtitle", "text": "자막 문장", "start": 0.0, "end": 5.0}},
    {{"type": "image", "filename": "scene2.jpg", "text": "이미지 설명", "start": 5.0, "end": 10.0}},
    {{"type": "subtitle", "text": "이미지 자막", "start": 5.0, "end": 10.0}}
  ]
}}
     """)
])


# ============================================================
# 5️⃣ Helper to ensure timeline validity
# ============================================================
def ensure_timeline_constraints(tl: TimelineOutput, analysis_json: dict, total: int):
    imgs = analysis_json.get("images", [])
    auds = analysis_json.get("audio", [])

    # 필수 컷 최소 길이 보정
    for item in tl.timeline:
        d = item.end - item.start
        if d < 3.0:
            item.end = item.start + 3.0
        elif d > 7.0:
            item.end = item.start + 7.0

    # 이미지, 오디오 필수 컷 자동 추가
    if not any(i.type == "image" for i in tl.timeline) and imgs:
        tl.timeline.append(TimelineItem(
            type="image", filename=imgs[0]["filename"], start=0.0, end=3.0
        ))

    if not any(i.type == "audio" for i in tl.timeline) and auds:
        tl.timeline.append(TimelineItem(
            type="audio", filename=auds[0]["filename"], start=0.0, end=float(total)
        ))

    return tl


# ============================================================
# 6️⃣ LangChain 구성 요소 (병렬 실행)
# ============================================================

def build_pipeline():
    scene_chain = scene_prompt | scene_llm
    story_chain = story_prompt | story_llm
    emotion_chain = emotion_prompt | emotion_llm
    hook_chain = hook_prompt | hook_llm
    timeline_chain = timeline_prompt | timeline_llm

    # RunnableParallel 구성
    story_inputs = RunnableParallel(
        scenes_json=scene_chain | RunnableLambda(lambda out: scenes_to_json(out)),
        duration=RunnableLambda(lambda inp: int(inp.get("duration", 30))),
        user_prompt=RunnableLambda(lambda inp: inp.get("user_prompt", "")),
        analysis_json=RunnableLambda(lambda inp: json.dumps(inp["analysis_json"], ensure_ascii=False))
    )

    timeline_inputs = RunnableParallel(
        analysis_json=RunnableLambda(lambda inp: json.dumps(inp["analysis_json"], ensure_ascii=False)),
        story_idea_json=story_chain | RunnableLambda(lambda out: story_to_json(out)),
        emotion_json=emotion_chain | RunnableLambda(lambda out: json.dumps(out, ensure_ascii=False)),
        hook_json=hook_chain | RunnableLambda(lambda out: json.dumps(out, ensure_ascii=False)),
        duration=RunnableLambda(lambda inp: int(inp.get("duration", 30))),
    )


    return {
        "scene_chain": scene_chain,
        "story_chain": story_chain,
        "emotion_chain": emotion_chain, 
        "hook_chain": hook_chain,
        "timeline_chain": timeline_chain,
        "story_inputs": story_inputs,
        "timeline_inputs": timeline_inputs
    }


# ============================================================
# 7️⃣ 최종 실행 함수
# ============================================================

def run_openai_pipeline(analysis_json: dict, duration: int, user_prompt: str):
    chains = build_pipeline()
    split = split_duration(duration)

    # 1. 장면 분석
    scenes = chains["scene_chain"].invoke({
        "analysis_json": json.dumps(analysis_json, ensure_ascii=False),
        "user_prompt": user_prompt
    })

    # 2. 스토리 생성
    story = chains["story_chain"].invoke({
        "scenes_json": scenes_to_json(scenes),
        "duration": duration,
        "user_prompt": user_prompt,
        **split
    })

    # 2.5 감정 및 후킹 생성
    emotion = chains["emotion_chain"].invoke({
        "story_idea_json": story_to_json(story)
    })
    
    hook = chains["hook_chain"].invoke({
        "story_idea_json": story_to_json(story)
    })

    # 3. 타임라인 생성
    timeline = chains["timeline_chain"].invoke({
        "analysis_json": json.dumps(analysis_json, ensure_ascii=False),
        "story_idea_json": story_to_json(story),
        "emotion_json": json.dumps(emotion.model_dump(), ensure_ascii=False),
        "hook_json": json.dumps(hook.model_dump(), ensure_ascii=False),
        "duration": duration,
    })

    
    # Debug용 타임라인 저장
    save_debug_timeline(timeline)

    # 4. 타임라인 제약 보정
    timeline = ensure_timeline_constraints(timeline, analysis_json, duration)

    return {
        "scenes": scenes,
        "story": story,
        "timeline": timeline
    }
