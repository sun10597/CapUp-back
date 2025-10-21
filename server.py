from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, json, traceback
from fastapi.responses import FileResponse, JSONResponse


# 기존 모듈 가져오기
from main import analyze_all_media, normalize_openai_analysis
from local_langchain import run_pipeline
from movie import render_shorts_from_timeline

# ---------------------------------
# 기본 설정
# ---------------------------------
app = FastAPI(title="AI Reels Generator CapUp")
RESULT_DIR = "./results"


MEDIA_DIR = "media"
RESULT_DIR = "results"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# CORS (React 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 결과 파일을 static으로 서비스
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.get("/")
def root():
    return {"message": "✅ FastAPI 서버 작동 중!", "media_dir": MEDIA_DIR, "result_dir": RESULT_DIR}


# ---------------------------------
# 업로드 & LangChain 파이프라인 실행
# ---------------------------------
@app.post("/api/upload")
async def upload_media(
    files: list[UploadFile],
    clipDuration: int = Form(...),
    aiPrompt: str = Form(...)
):
    """
    1. 업로드된 영상/이미지 저장
    2. OpenAI Vision 기반 분석
    3. LangChain 파이프라인 실행
    4. MoviePy 렌더링 후 결과 반환
    """
    try:
        # 1️⃣ 파일 저장
        saved_files = []
        for file in files:
            save_path = os.path.join(MEDIA_DIR, file.filename)
            with open(save_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(file.filename)

        print(f"📂 업로드 완료: {saved_files}")

        # 2️⃣ 미디어 분석
        print("🧠 Step 1. OpenAI Vision 분석 중...")
        analysis_results = analyze_all_media()  # media 폴더 전체 분석
        print(f"✅ 분석 완료: {len(analysis_results)}개 항목")

        # 3️⃣ LangChain 파이프라인 실행
        print("🧩 Step 2. LangChain 파이프라인 실행...")
        normalized = normalize_openai_analysis(analysis_results, aiPrompt)
        result = run_pipeline(normalized, duration=clipDuration, user_prompt=aiPrompt)

        # Pydantic to dict 변환
        if hasattr(result.get("timeline"), "model_dump"):
            result["timeline"] = result["timeline"].model_dump()

        # 4️⃣ MoviePy 렌더링
        print("🎬 Step 3. MoviePy 렌더링 시작...")
        output_path = os.path.join(RESULT_DIR, "final_shorts.mp4")
        render_shorts_from_timeline(result, output_path=output_path)

        print(f"✅ 최종 영상 생성 완료 → {output_path}")

        # 5️⃣ 결과 반환
        return {
            "message": "✅ 영상 생성 완료!",
            "result_path": f"/results/final_shorts.mp4",
            "files": saved_files,
        }

    except Exception as e:
        print("❌ 오류 발생:", traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/api/export")
async def download_latest_video():
    """
    results 폴더에서 가장 최근 생성된 MP4 파일을 찾아서 자동 다운로드
    """
    try:
        if not os.path.exists(RESULT_DIR):
            return JSONResponse({"error": "결과 폴더가 없습니다."}, status_code=404)

        mp4_files = [
            f for f in os.listdir(RESULT_DIR)
            if f.lower().endswith(".mp4")
        ]
        if not mp4_files:
            return JSONResponse({"error": "저장된 mp4 파일이 없습니다."}, status_code=404)

        # 수정시간(최근순)으로 정렬
        mp4_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(RESULT_DIR, f)),
            reverse=True
        )
        latest_file = mp4_files[0]
        filepath = os.path.join(RESULT_DIR, latest_file)

        print(f"🎬 최신 파일 다운로드 요청: {latest_file}")
        return FileResponse(
            filepath,
            media_type="video/mp4",
            filename=latest_file
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------------------------
# 로컬 테스트 실행용
# ---------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
