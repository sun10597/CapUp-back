# 🚀 CapUp AI Reels Generator — 서버 실행 가이드

## 📦 프로젝트 개요
이 서버는 **AI 기반 자동 유튜브 쇼츠 생성기 (CapUp)** 백엔드입니다.  
사용자가 업로드한 영상·이미지·오디오를 분석하여 LangChain을 통해 **스토리라인과 타임라인을 생성**,  
MoviePy로 최종 영상을 자동 렌더링합니다.

---

## 🧩 구성 파일 요약
| 파일 | 역할 |
|------|------|
| `server.py` | FastAPI 메인 서버 (업로드, 렌더링, 다운로드 API) |
| `main.py` | OpenAI Vision/Video Intelligence 분석 통합 |
| `local_langchain.py` | LangChain 기반 스토리/타임라인 파이프라인 |
| `langchain_story.py` | Pydantic 스키마 정의 (SceneItem, StoryIdeaOutput 등) |
| `movie.py` | MoviePy 렌더링 및 타임라인 파서 |
| `results/` | 최종 렌더링된 mp4 저장 경로 |
| `media/` | 업로드된 원본 영상/이미지 저장 경로 |

---

## 🧰 1. 필수 환경 세팅

### 🐍 Python 환경
```bash
conda create -n capup python=3.12
conda activate capup
pip install -r requirements.txt
```

**requirements.txt 예시**
```txt
fastapi
uvicorn
pydantic
moviepy
langchain
langchain-openai
python-multipart
openai
tqdm
```

---

## 🧙‍♂️ 2. ImageMagick 설치 (자막 렌더링 필수)

MoviePy의 `TextClip`은 ImageMagick을 내부적으로 사용합니다.  
Windows 사용 시 다음 경로가 정확히 일치해야 합니다:

```
C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe
```

환경 변수에 추가하거나,  
`movie.py` 상단의 다음 라인이 현재 경로에 맞게 수정되어야 합니다:

```python
os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
```

✅ 설치 확인:
```bash
magick -version
```

---

## 📁 3. 폴더 구조
```
project_root/
├─ media/              # 업로드된 원본 파일
├─ results/            # 렌더링된 결과(mp4)
├─ movie.py
├─ server.py
├─ main.py
├─ local_langchain.py
├─ langchain_story.py
└─ requirements.txt
```

---

## ⚙️ 4. 서버 실행

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

정상 실행 시:
```
INFO:     Started server process [12345]
INFO:     Application startup complete.
```

---

## 🌐 5. API 엔드포인트

| 메서드 | 경로 | 설명 |
|---------|------|------|
| `POST` | `/api/upload` | 파일 업로드 + AI 분석 + 자동 렌더링 |
| `GET`  | `/api/export` | `results` 폴더 내 최신 mp4 다운로드 |
| `GET`  | `/` | 서버 상태 확인 (`✅ FastAPI 서버 작동 중!`) |

---

## 🪄 6. 클라이언트 (React) 연동 요약
프론트엔드에서 업로드 시:
```tsx
const formData = new FormData();
files.forEach(f => formData.append("files", f));
formData.append("clipDuration", 30);
formData.append("aiPrompt", "한국 폴리텍 AI융합소프트웨어과 소개 영상");

await fetch("http://localhost:8000/api/upload", {
  method: "POST",
  body: formData,
});
```

렌더링 완료 후 영상 다운로드 시:
```tsx
const res = await fetch("http://localhost:8000/api/export");
const blob = await res.blob();
const url = window.URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "final_shorts.mp4";
a.click();
```

---

## 🧩 7. 자주 발생하는 문제

| 문제 | 원인 / 해결 |
|------|--------------|
| ⚠️ `MoviePy Error: creation of None failed` | ImageMagick 설치 안 됨 or 경로 불일치 |
| ⚠️ `TypeError: Failed to fetch` | React ↔ FastAPI CORS 설정 누락 |
| ⚠️ `타임라인에 렌더링 가능한 항목이 없습니다` | LangChain 결과에 `video/image/subtitle` 누락 |
| ⚠️ `FileNotFoundError` | media 폴더 또는 results 폴더 누락 |
| ⚠️ 폰트 깨짐 | `malgun.ttf` 경로가 잘못됨 → Windows 기본 폰트 지정 필요 |

---

## 🧾 8. 결과물 저장 규칙
- 모든 결과 영상은 `results/final_shorts.mp4` 형태로 저장  
- 이후 `/api/export` 요청 시 가장 최근 수정된 mp4 자동 반환  
- 과거 영상은 `results/` 폴더에 그대로 보존됨
