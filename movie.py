import os, re, json, unicodedata

# ImageMagick 경로 설정 (Windows 전용)
#os마다 다를 수 있으니 필요시 수정
os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, TextClip,
    CompositeVideoClip, CompositeAudioClip
)



# =============================
# 전역 설정 & 유틸
# =============================
DEBUG = True
MEDIA_DIR  = "./media"
RESULT_DIR = "./results"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def dbg(*args):
    if DEBUG:
        print("[DBG]", *args)

def safe_path(filename: str) -> str:
    if not filename:
        return ""
    for base in [MEDIA_DIR, RESULT_DIR, "./temp", "."]:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    dbg(f"safe_path: 파일을 찾지 못함 → {filename}")
    return filename

def _normalize_str(s: str) -> str:
    """스마트따옴표, 제어문자, 비정규 공백 제거"""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201C", "\"").replace("\u201D", "\"")
    s = s.replace("\u00A0", " ")  # non-breaking space
    return s


# =============================
# Timeline 문자열 파서 (강화 + 디버깅)
# =============================
def parse_timeline_from_string(timeline_json):
    """
    LangChain에서 나온 'story_summary=... timeline=[TimelineItem(...), ...]' 문자열,
    혹은 이미 list/dict로 온 케이스를 모두 dict list로 변환.
    """
    dbg("parse_timeline_from_string: type =", type(timeline_json))

    # 이미 list
    if isinstance(timeline_json, list):
        dbg("timeline_json is list → length:", len(timeline_json))
        return timeline_json

    # dict 일 때: timeline 필드 꺼내 재귀
    if isinstance(timeline_json, dict):
        keys = list(timeline_json.keys())
        dbg("timeline_json is dict → keys:", keys)
        if "timeline" in timeline_json:
            inner = timeline_json["timeline"]
            dbg("dict['timeline'] type:", type(inner))
            return parse_timeline_from_string(inner)
        else:
            # 서버에서 {"story_summary": ..., "timeline":[...]} 형태로 올 수도 있음
            if "story_summary" in timeline_json and isinstance(timeline_json.get("timeline"), list):
                return timeline_json["timeline"]
            dbg("dict 형태지만 'timeline' 키가 없음 → 빈 리스트 반환")
            return []

    # 문자열이면 파싱 시도
    if isinstance(timeline_json, str):
        raw = _normalize_str(timeline_json).strip()
        dbg("timeline_json is str → length:", len(raw))
        # 1) "timeline=" 뒤만 자르기
        sub = raw
        if "timeline=" in raw:
            start_idx = raw.find("timeline=") + len("timeline=")
            sub = raw[start_idx:].strip()
            # 끝 ']' 보정
            if not sub.endswith("]") and "]" in sub:
                sub = sub[: sub.rfind("]") + 1]

        # 2) 혹시 JSON 배열로 직렬화되어 있으면 먼저 시도
        if sub.startswith("[") and sub.endswith("]"):
            try:
                arr = json.loads(sub)
                if isinstance(arr, list):
                    dbg("JSON 배열로 파싱 성공 → items:", len(arr))
                    return arr
            except Exception as e:
                dbg("JSON 파싱 실패 (무시):", e)

        # 3) TimelineItem(...) 패턴 파싱
        #    - attr=value 쌍을 더 관대하게 잡음 (문자열, 숫자, None)
        items = []
        matches = re.findall(r"TimelineItem\((.*?)\)", sub, flags=re.DOTALL)
        dbg("Regex matches (TimelineItem):", len(matches))

        for idx, m in enumerate(matches):
            # key=value를 폭넓게 잡기 (문자열 따옴표/공백/None/숫자)
            kv_pairs = re.findall(r"(\w+)\s*=\s*('.*?'|\".*?\"|None|[0-9.]+)", m, flags=re.DOTALL)
            d = {}
            for k, v in kv_pairs:
                v = v.strip()
                # 따옴표 제거
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                if v == "None":
                    d[k] = None
                elif k in ("start", "end"):
                    try:
                        d[k] = float(v)
                    except:
                        d[k] = 0.0
                else:
                    d[k] = _normalize_str(v)
            items.append(d)
            if idx < 2:  # 앞 2개만 샘플 로그
                dbg(f"parsed item[{idx}] →", d)

        dbg("총 파싱 결과 items:", len(items))
        return items

    dbg("알 수 없는 타입의 timeline_json → 빈 리스트")
    return []


# =============================
# 렌더링 메인 함수 (진단 로그 포함)
# =============================
def render_shorts_from_timeline(
    timeline_json,
    output_path="results/final_shorts.mp4",
    resolution=(1080, 1920),
    fps=30
):
    print("🎬 Step 3. MoviePy 렌더링 시작...")

    if hasattr(timeline_json, "model_dump"):
        timeline_json = timeline_json.model_dump()

    # 1) 파싱
    timeline_data = parse_timeline_from_string(timeline_json)
    if not isinstance(timeline_data, list):
        dbg("timeline_data가 list가 아님 → type:", type(timeline_data))

    dbg("timeline_data length:", len(timeline_data))
    if timeline_data[:1]:
        dbg("timeline_data sample[0]:", timeline_data[0])

    # 2) 타입/필수키 검증 + drop 사유 로깅
    renderable = []
    drop_reasons = {"no_type":0, "bad_type":0, "no_time":0, "video_missing":0, "image_missing":0, "subtitle_empty":0}
    for i, item in enumerate(timeline_data):
        if not isinstance(item, dict):
            dbg(f"[DROP#{i}] dict가 아님 → {type(item)}")
            continue
        t = str(item.get("type", "")).strip().lower().replace("'", "").replace('"', '')
        start = item.get("start", None)
        end   = item.get("end", None)

        if not t:
            drop_reasons["no_type"] += 1
            dbg(f"[DROP#{i}] type 없음 → item:", item)
            continue
        if t not in {"video", "image", "subtitle", "audio"}:
            drop_reasons["bad_type"] += 1
            dbg(f"[DROP#{i}] 지원하지 않는 type='{t}' → item:", item)
            continue
        if start is None or end is None:
            drop_reasons["no_time"] += 1
            dbg(f"[DROP#{i}] start/end 없음 → item:", item)
            continue

        filename = item.get("filename")
        if t == "video":
            path = safe_path(filename)
            if not filename or not os.path.exists(path):
                drop_reasons["video_missing"] += 1
                dbg(f"[DROP#{i}] 비디오 파일 없음 → filename={filename} / path={path}")
                continue
        if t == "image":
            path = safe_path(filename)
            if not filename or not os.path.exists(path):
                drop_reasons["image_missing"] += 1
                dbg(f"[DROP#{i}] 이미지 파일 없음 → filename={filename} / path={path}")
                continue
        if t == "subtitle":
            if not str(item.get("text", "")).strip():
                drop_reasons["subtitle_empty"] += 1
                dbg(f"[DROP#{i}] 자막 text 없음 → item:", item)
                continue

        # 통과
        norm = dict(item)
        norm["type"] = t
        try:
            norm["start"] = float(start)
            norm["end"]   = float(end)
        except:
            drop_reasons["no_time"] += 1
            dbg(f"[DROP#{i}] start/end float 변환 실패 → item:", item)
            continue

        renderable.append(norm)

    dbg("drop_reasons:", drop_reasons)
    print(f"✅ 렌더링 대상 {len(renderable)}개 항목 로드 완료")

    if not renderable:
        print("⚠️ 타임라인에 렌더링 가능한 video/image/subtitle이 없습니다.")
        print(json.dumps(timeline_json, ensure_ascii=False, indent=2, default=str))
        return

    # 3) 정렬 & 렌더링
    renderable.sort(key=lambda x: x["start"])

    clips, audio_tracks = [], []
    for item in renderable:
        t        = item["type"]
        start    = item["start"]
        end      = item["end"]
        duration = max(0.1, end - start)
        filename = item.get("filename")
        filepath = safe_path(filename)

        # 🎞️ 동영상
        if t == "video":
            try:
                clip = VideoFileClip(filepath).subclip(0, duration).resize(resolution)
                clips.append(clip.set_start(start).crossfadein(0.2))
                print(f"🎞️ 비디오 추가: {os.path.basename(filepath)} ({start}-{end}s)")
            except Exception as e:
                print(f"⚠️ 비디오 로드 실패: {filepath} ({e})")

        # 🖼️ 이미지
        elif t == "image":
            try:
                img = ImageClip(filepath, duration=duration).resize(resolution)
                clips.append(img.set_start(start).crossfadein(0.3))
                print(f"🖼️ 이미지 추가: {os.path.basename(filepath)} ({start}-{end}s)")
            except Exception as e:
                print(f"⚠️ 이미지 로드 실패: {filepath} ({e})")
        
        # 💬 자막
        elif t == "subtitle":
            try:
                txt = TextClip(
                    item["text"],
                    fontsize=60,
                    color="white",
                    font="C:/Windows/Fonts/malgun.ttf",  # Windows 한글 폰트
                    method="caption",
                    stroke_color="black",
                    stroke_width=2,
                    size=(1080, None)
                ).set_position(("center", resolution[1] - 150)).set_start(start).set_duration(duration)
                clips.append(txt)
                print(f"💬 자막 추가: '{item['text']}' ({start}-{end}s)")
            except Exception as e:
                print(f"⚠️ 자막 렌더링 실패: {item.get('text')} ({e})")
        
        # 🎵 오디오
        elif t == "audio":
            try:
                if not filename:
                    # 기본 BGM 파일명을 원하면 여기에 지정
                    # filename = "default_bgm.mp3"
                    pass
                if filename:
                    path = safe_path(filename)
                    if os.path.exists(path):
                        aud = AudioFileClip(path).subclip(0, duration)
                        audio_tracks.append(aud.set_start(start))
                        print(f"🎵 오디오 추가: {os.path.basename(path)} ({start}-{end}s)")
            except Exception as e:
                print(f"⚠️ 오디오 로드 실패: {filename} ({e})")

    if not clips:
        print("⚠️ 렌더링 가능한 클립이 없습니다. (필터는 통과했지만 clip 생성 실패)")
        return

    video = CompositeVideoClip(clips, size=resolution)
    if audio_tracks:
        final_audio = CompositeAudioClip(audio_tracks)
        video = video.set_audio(final_audio)

    print(f"\n📦 렌더링 시작 → {output_path}")
    video.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=fps, preset="fast", threads=4
    )
    print(f"✅ 최종 영상 생성 완료: {output_path}")
