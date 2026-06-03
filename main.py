"""
RTSP CCTV VLM Analyzer (Web)
- FastAPI + native WebSocket (no flask/flask-socketio)
- MJPEG video stream via StreamingResponse
"""

import json
import os
import threading
import time
import queue
from collections import deque
import base64
import asyncio
import requests
import cv2
from PIL import Image
import io
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, StreamingResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL = "http://localhost:1111/v1"
VLLM_MODEL = "/media/ds/DATA/models/Qwen2.5-VL-3B"
CAPTURE_FPS = 1                     # 1fps → 10초에 10장
FRAME_INTERVAL = 1.0 / CAPTURE_FPS
FRAME_BUFFER_SIZE = 10              # 링버퍼 크기 (= image limit per prompt)

# FFMPEG low-latency options — must be set before any VideoCapture is created
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

# ─── State ────────────────────────────────────────────────────────────────────
# 10초 링버퍼: 1fps × 10장. VLM 요청 시 전부 전송
frame_buffer      = deque(maxlen=FRAME_BUFFER_SIZE)
frame_buffer_lock = threading.Lock()

anno_queue  = queue.Queue(maxsize=1)   # raw frames → YOLO thread
output_frame = None
output_lock  = threading.Lock()

# Pre-encoded JPEG cache (annotated by YOLO) served by /frame
latest_jpeg  = None
jpeg_lock    = threading.Lock()

running = False

# ─── YOLO ─────────────────────────────────────────────────────────────────────
_yolo_model  = None
_yolo_lock   = threading.Lock()


def _load_yolo(model_path: str) -> object | None:
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        try:
            os.environ.setdefault("YOLO_AUTOINSTALL", "False")
            os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

            # Jetson: TensorRT/torch CUDA 바인딩 경로 추가
            # 1) 시스템 dist-packages  (레포 yolo_runtime.py 동일 로직)
            # 2) vLLM venv site-packages  (~/vllm-jetson-v092-torch27)
            import sys, glob
            from pathlib import Path
            major, minor = sys.version_info.major, sys.version_info.minor
            candidate_paths = [
                Path(f"/usr/lib/python{major}.{minor}/dist-packages"),
                Path("/usr/lib/python3/dist-packages"),
                Path("/usr/local/lib/python3/dist-packages"),
            ]
            # vLLM venv (Jetson CUDA torch + TensorRT 포함)
            vllm_venv = Path.home() / "vllm-jetson-v092-torch27"
            for sp in glob.glob(str(vllm_venv / "lib" / "python*" / "site-packages")):
                candidate_paths.append(Path(sp))
            for p in candidate_paths:
                if p.exists() and str(p) not in sys.path:
                    sys.path.append(str(p))

            # ultralytics 자동설치 방지 (임포트 후에도 재적용)
            try:
                import ultralytics.utils as _u; _u.AUTOINSTALL = False
            except Exception:
                pass

            from ultralytics import YOLO
            import numpy as np

            m = YOLO(model_path, task="detect")
            m(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # warmup
            _yolo_model = m
            print(f"[YOLO] model ready: {model_path}  classes={m.names}")
        except Exception as exc:
            print(f"[YOLO] load failed ({model_path}): {exc}")
    return _yolo_model


def yolo_annotate_worker():
    global latest_jpeg
    with settings_lock:
        model_path = settings.get("yolo_model_path", "/media/ds/DATA/yolo_final/0507_best.engine")
        confidence = float(settings.get("yolo_confidence", 0.5))

    model = _load_yolo(model_path)
    if model is None:
        print("[YOLO] worker: model unavailable — streaming raw frames")

    while running:
        try:
            frame = anno_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if model is not None:
            try:
                res        = model(frame, conf=confidence, verbose=False)[0]
                # result.plot() — 레포 detector.py 와 동일한 방식
                annotated  = res.plot()
                n = len(res.boxes)
                if n:
                    print(f"[YOLO] {n} detection(s): "
                          + ", ".join(f"{res.names[int(b.cls[0])]} {float(b.conf[0]):.2f}"
                                      for b in res.boxes))
            except Exception as exc:
                print(f"[YOLO] inference error: {exc}")
                annotated = frame
        else:
            annotated = frame

        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()

SETTINGS_FILE = "settings.json"

settings = {
    "cam_ip": "192.168.0.15",
    "cam_port": "554",
    "cam_user": "admin",
    "cam_pw": "@Ekthf5081",
    "cam_path": "/stream1",
    "vllm_url": VLLM_URL,
    "model_name": VLLM_MODEL,
    "api_key": "test-key",
    "prompt": "이 영상에서 위험한 행동이나 안전 문제를 분석하고 보고해 주세요.",
    "yolo_model_path": "0507_best.engine",
    "yolo_confidence": 0.5,
}
settings_lock = threading.Lock()


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        with settings_lock:
            settings.update({k: v for k, v in saved.items() if k in settings})
    except Exception:
        pass


def save_settings():
    with settings_lock:
        data = dict(settings)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ─── WebSocket Manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            if ws in self.active:
                self.active.remove(ws)

    async def broadcast(self, msg: dict):
        with self._lock:
            conns = list(self.active)
        dead = []
        for ws in conns:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def emit(self, msg: dict):
        """Thread-safe broadcast from background workers."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self._loop)


manager = ConnectionManager()


@app.on_event("startup")
async def startup():
    manager._loop = asyncio.get_event_loop()
    load_settings()
    # Preload YOLO in background so it's warm when stream starts
    with settings_lock:
        mp = settings.get("yolo_model_path", "0507_best.engine")
    threading.Thread(target=_load_yolo, args=(mp,), daemon=True).start()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def build_rtsp_url(s):
    user = s["cam_user"]
    pw   = s["cam_pw"]
    ip   = s["cam_ip"]
    port = s["cam_port"]
    path = s["cam_path"]
    if not path.startswith("/"):
        path = "/" + path
    cred = f"{user}:{pw}@" if user else ""
    return f"rtsp://{cred}{ip}:{port}{path}"


# 10장을 max_model_len=2048 안에 넣기 위해 프레임당 픽셀 축소
# Qwen2.5-VL: tokens ≈ (H/28)×(W/28), 320×320 → ~130 tokens/frame × 10 = 1300 tokens
MAX_VLM_PIXELS_PER_FRAME = 320 * 320


def encode_frame_base64(frame_bgr, max_pixels: int = MAX_VLM_PIXELS_PER_FRAME) -> str:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    w, h = pil_img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _stamp_frame(frame, idx: int, total: int):
    """각 프레임 좌상단에 시간 라벨을 삽입해 VLM이 순서를 파악하게 한다."""
    out = frame.copy()
    age = total - idx - 1          # 0 = 현재, total-1 = 가장 오래된 프레임
    time_str = "now" if age == 0 else f"{age}s ago"
    label = f"Frame {idx + 1}/{total}  ({time_str})"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(out, (4, 4), (tw + 14, th + 14), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, label, (9, th + 9), font, scale, (0, 185, 118), thick, cv2.LINE_AA)
    return out


def call_vllm_stream(frames: list, prompt: str, vllm_url: str,
                     model_name: str, api_key: str = "test-key"):
    """링버퍼 프레임 전부를 시간 컨텍스트와 함께 한 요청에 넣어 스트리밍."""
    n = len(frames)

    # ① 각 이미지에 프레임 번호/시간 라벨 삽입
    content = []
    for i, f in enumerate(frames):
        labeled = _stamp_frame(f, i, n)
        b64 = encode_frame_base64(labeled)
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # ② 프롬프트 앞에 시간 흐름 설명 추가 → VLM이 이동 방향 등을 이해
    temporal_ctx = (
        f"The {n} images above are consecutive frames captured 1 second apart "
        f"(Frame 1 = {n-1}s ago → Frame {n} = now). "
        "Use the temporal sequence to understand motion direction, speed, and changes over time.\n\n"
    )
    content.append({"type": "text", "text": temporal_ctx + prompt})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": True,
    }
    try:
        with requests.post(
            f"{vllm_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    except requests.exceptions.ConnectionError:
        yield "[오류] vLLM 서버에 연결할 수 없습니다."
    except requests.exceptions.Timeout:
        yield "[오류] 요청 시간 초과 (60초)"
    except Exception as e:
        yield f"[오류] {str(e)}"


# ─── Worker Threads ───────────────────────────────────────────────────────────
def _open_cap(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def rtsp_capture_worker(rtsp_url):
    global running, output_frame
    cap = _open_cap(rtsp_url)
    last_analysis_time = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            cap.release()
            cap = _open_cap(rtsp_url)
            continue

        # ── Display: pass to YOLO annotator (non-blocking, drop stale) ───────
        with output_lock:
            output_frame = frame
        if anno_queue.full():
            try:
                anno_queue.get_nowait()
            except queue.Empty:
                pass
        anno_queue.put(frame)   # no copy — YOLO worker reads it promptly

        # ── 링버퍼: 1fps로 최신 10장 유지 ──────────────────────────────────
        now = time.time()
        if now - last_analysis_time >= FRAME_INTERVAL:
            with frame_buffer_lock:
                frame_buffer.append(frame.copy())
            last_analysis_time = now

    cap.release()


def _run_vlm_once(frames: list, prompt: str, vllm_url: str,
                  model: str, api_key: str):
    """전송 버튼 클릭 시 1회 실행되는 VLM 분석 스레드."""
    ts = datetime.now().strftime("%H:%M:%S")
    manager.emit({"type": "status",  "text": f"[{ts}] 분석 중… ({len(frames)}장)"})
    manager.emit({"type": "result_start", "time": ts})
    for chunk in call_vllm_stream(frames, prompt, vllm_url, model, api_key):
        manager.emit({"type": "chunk", "text": chunk})
    manager.emit({"type": "result_end"})
    ts2 = datetime.now().strftime("%H:%M:%S")
    manager.emit({"type": "status", "text": f"● 분석 완료 [{ts2}]"})


# ─── MJPEG Stream ─────────────────────────────────────────────────────────────
def mjpeg_generator():
    while True:
        with jpeg_lock:
            data = latest_jpeg
        if data is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(1 / 30)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    with settings_lock:
        s = dict(settings)
    return templates.TemplateResponse(request, "index.html", {"settings": s})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/frame")
def get_frame():
    """Returns the latest pre-encoded JPEG frame (zero encoding latency)."""
    with jpeg_lock:
        data = latest_jpeg
    if data is None:
        return Response(status_code=204)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global running
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "start_stream":
                with settings_lock:
                    for key in ("cam_ip", "cam_port", "cam_user", "cam_pw", "cam_path",
                                "vllm_url", "model_name", "api_key", "prompt",
                                "yolo_model_path", "yolo_confidence"):
                        if key in data:
                            settings[key] = data[key]
                    rtsp_url = build_rtsp_url(settings)
                save_settings()

                if not running:
                    running = True
                    threading.Thread(target=rtsp_capture_worker,  args=(rtsp_url,), daemon=True).start()
                    threading.Thread(target=yolo_annotate_worker, daemon=True).start()

                display = rtsp_url.split("@")[-1]
                await ws.send_json({"type": "status", "text": f"● 스트리밍 중 — {display}"})

            elif action == "stop_stream":
                running = False
                await ws.send_json({"type": "status", "text": "● 중지됨"})

            elif action == "analyze":
                prompt = data.get("prompt", "").strip()
                if prompt:
                    with settings_lock:
                        settings["prompt"] = prompt
                    save_settings()
                else:
                    with settings_lock:
                        prompt = settings["prompt"]

                with frame_buffer_lock:
                    frames = list(frame_buffer)

                if not frames:
                    await ws.send_json({"type": "status",
                                        "text": "⚠ 프레임 없음 — 스트림을 먼저 시작하세요."})
                else:
                    with settings_lock:
                        vllm_url = settings["vllm_url"]
                        model    = settings["model_name"]
                        api_key  = settings["api_key"]
                    threading.Thread(
                        target=_run_vlm_once,
                        args=(frames, prompt, vllm_url, model, api_key),
                        daemon=True,
                    ).start()

            elif action == "update_prompt":
                with settings_lock:
                    settings["prompt"] = data.get("prompt", settings["prompt"])
                save_settings()
                await ws.send_json({"type": "status", "text": "● 프롬프트 저장됨"})

            elif action == "save_settings":
                with settings_lock:
                    for key in ("cam_ip", "cam_port", "cam_user", "cam_pw", "cam_path",
                                "vllm_url", "model_name", "api_key", "prompt",
                                "yolo_model_path", "yolo_confidence"):
                        if key in data:
                            settings[key] = data[key]
                save_settings()
                await ws.send_json({"type": "status", "text": "● 설정 저장됨"})

    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    print("서버 시작: http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
