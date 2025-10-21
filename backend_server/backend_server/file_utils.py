import os
import subprocess

# 확장자 판별
def check_file_type(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".mp4", ".mov", ".avi"]:
        return "video"
    elif ext in [".jpg", ".jpeg", ".png"]:
        return "image"
    else:
        return "unknown"
    
def extract_audio(file_path: str, audio_path: str = "temp.wav") -> str:
    file_type = check_file_type(file_path)

    # 음성 추출
    if file_type == "video":
        print("영상입니다.")
        command = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-vn",                  
            "-acodec", "pcm_s16le", 
            "-ar", "16000",         
            "-ac", "1",             
            audio_path
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"오디오 추출 실패: {e.stderr.decode('utf-8')}")
    elif file_type == "image":
        print("이미지입니다.")
        return None
    else:
        raise ValueError("지원하지 않는 파일 형식")