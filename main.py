"""
RTSP CCTV VLM Analyzer (Web)
- FastAPI + native WebSocket (no flask/flask-socketio)
- MJPEG video stream via StreamingResponse
"""

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
from fastapi.responses import StreamingResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL = "http://localhost:1111/v1"
VLLM_MODEL = "/media/ds/DATA/models/Qwen2.5-VL-3B"
CAPTURE_FPS = 2
FRAME_INTERVAL = 1.0 / CAPTURE_FPS

# ─── State ────────────────────────────────────────────────────────────────────
frame_queue = queue.Queue(maxsize=2)
output_frame = None
output_lock = threading.Lock()
running = False

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
}
settings_lock = threading.Lock()

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


def call_vllm(frame_bgr, prompt, vllm_url, model_name, api_key="test-key"):
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
    }
    try:
        resp = requests.post(
            f"{vllm_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[오류] vLLM 서버에 연결할 수 없습니다. 서버 URL을 확인하세요."
    except requests.exceptions.Timeout:
        return "[오류] 요청 시간 초과 (30초)"
    except Exception as e:
        return f"[오류] {str(e)}"


# ─── Worker Threads ───────────────────────────────────────────────────────────
def rtsp_capture_worker(rtsp_url):
    global running, output_frame
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    last_time = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        now = time.time()
        if now - last_time >= FRAME_INTERVAL:
            with output_lock:
                output_frame = frame.copy()
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            else:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put(frame.copy())
            last_time = now

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

        result = call_vllm(frame, prompt, vllm_url, model, api_key)
        ts2 = datetime.now().strftime("%H:%M:%S")
        manager.emit({"type": "result", "text": f"[{ts2}]\n{result}"})


# ─── MJPEG Stream ─────────────────────────────────────────────────────────────
def mjpeg_generator():
    global output_frame
    while True:
        with output_lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.1)
            continue
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
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
                                "vllm_url", "model_name", "api_key", "prompt"):
                        if key in data:
                            settings[key] = data[key]
                    rtsp_url = build_rtsp_url(settings)

                if not running:
                    running = True
                    threading.Thread(target=rtsp_capture_worker, args=(rtsp_url,), daemon=True).start()
                    threading.Thread(target=vlm_analysis_worker, daemon=True).start()

                display = rtsp_url.split("@")[-1]
                await ws.send_json({"type": "status", "text": f"● 스트리밍 중 — {display}"})

            elif action == "stop_stream":
                running = False
                await ws.send_json({"type": "status", "text": "● 중지됨"})

            elif action == "update_prompt":
                with settings_lock:
                    settings["prompt"] = data.get("prompt", settings["prompt"])
                await ws.send_json({"type": "status", "text": "● 프롬프트 업데이트됨"})

    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    print("서버 시작: http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
