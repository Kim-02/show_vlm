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
CAPTURE_FPS = 2
FRAME_INTERVAL = 1.0 / CAPTURE_FPS

# FFMPEG low-latency options — must be set before any VideoCapture is created
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

# ─── State ────────────────────────────────────────────────────────────────────
frame_queue = queue.Queue(maxsize=2)
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
            from ultralytics import YOLO
            import numpy as np

            # TensorRT .engine 파일은 task 명시 필요 (레포 detector.py 동일)
            m = YOLO(model_path, task="detect")
            # warmup — inference 첫 실행 지연 제거
            m(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
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


MAX_VLM_PIXELS = 448 * 448  # Qwen2.5-VL: 28px/patch*4token, max_model_len=2048

def encode_frame_base64(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    # 토큰 한계 초과 방지: 픽셀 수가 MAX_VLM_PIXELS를 넘으면 비율 유지하며 축소
    w, h = pil_img.size
    if w * h > MAX_VLM_PIXELS:
        scale = (MAX_VLM_PIXELS / (w * h)) ** 0.5
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vllm_stream(frame_bgr, prompt, vllm_url, model_name, api_key="test-key"):
    """Yields text chunks from the vLLM streaming API."""
    b64 = encode_frame_base64(frame_bgr)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
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

        # ── VLM analysis queue: rate-limited to CAPTURE_FPS ─────────────────
        now = time.time()
        if now - last_analysis_time >= FRAME_INTERVAL:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame.copy())
            last_analysis_time = now

    cap.release()


def vlm_analysis_worker():
    global running
    while running:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        with settings_lock:
            prompt   = settings["prompt"]
            vllm_url = settings["vllm_url"]
            model    = settings["model_name"]
            api_key  = settings["api_key"]

        if not prompt:
            continue

        ts = datetime.now().strftime("%H:%M:%S")
        manager.emit({"type": "status", "text": f"[{ts}] 분석 중..."})
        manager.emit({"type": "result_start", "time": ts})

        for chunk in call_vllm_stream(frame, prompt, vllm_url, model, api_key):
            manager.emit({"type": "chunk", "text": chunk})

        manager.emit({"type": "result_end"})
        ts2 = datetime.now().strftime("%H:%M:%S")
        manager.emit({"type": "status", "text": f"● 스트리밍 중 [{ts2}]"})


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
                    threading.Thread(target=vlm_analysis_worker,  daemon=True).start()

                display = rtsp_url.split("@")[-1]
                await ws.send_json({"type": "status", "text": f"● 스트리밍 중 — {display}"})

            elif action == "stop_stream":
                running = False
                await ws.send_json({"type": "status", "text": "● 중지됨"})

            elif action == "update_prompt":
                with settings_lock:
                    settings["prompt"] = data.get("prompt", settings["prompt"])
                save_settings()
                await ws.send_json({"type": "status", "text": "● 프롬프트 업데이트됨"})

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
