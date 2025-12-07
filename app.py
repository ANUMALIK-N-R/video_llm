import streamlit as st
import requests
import time
import base64
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
from PIL import Image

# ---------------------------------------------------------------------
# CUSTOM PROFESSIONAL WHITE/BLACK WEBSITE THEME
# ---------------------------------------------------------------------

st.markdown("""
<style>

/* GLOBAL */
body, .stApp {
    background-color: white !important;
    color: black !important;
    font-family: 'Inter', sans-serif;
}

/* HEADERS */
h1, h2, h3 {
    font-weight: 600 !important;
    color: black !important;
}

/* UPLOAD & INPUT PANELS */
section, .css-1v0mbdj, .css-1t32rxk, .css-1dp5vir {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    padding: 20px !important;
    border-radius: 8px !important;
}

/* BUTTON */
.stButton button {
    background-color: black !important;
    color: white !important;
    padding: 10px 18px;
    border-radius: 6px;
    border: none;
    font-size: 16px;
}
.stButton button:hover {
    background-color: #333333 !important;
}

/* TEXT INPUT */
input[type="text"] {
    border-radius: 6px !important;
    border: 1px solid #cfcfcf !important;
}

/* CODE BLOCKS */
.stCodeBlock {
    background: #f7f7f7 !important;
    color: black !important;
    border-radius: 6px;
    border: 1px solid #dcdcdc !important;
}

/* SIDEBAR */
.css-1d391kg, .css-1lcbmhc {
    background-color: white !important;
    border-right: 1px solid #e5e5e5 !important;
}

/* VIDEO PLAYER BLOCK */
.stVideo {
    border-radius: 6px;
    border: 1px solid #e5e5e5 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------- ORIGINAL LOGIC BELOW -------------------------------

API_KEY = "AIzaSyB_tg6FH673c3MIKgj7rGM4y9Li2xDUOEw"
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-preview-09-2025:generateContent?key=" + API_KEY
)
MAX_FRAMES_FOR_VLM = 8
VLM_SAMPLE_RATE = 2
TARGET_FPS = 1
MAX_RETRIES = 5

if 'object_memory' not in st.session_state:
    st.session_state.object_memory = {}
if 'video_duration' not in st.session_state:
    st.session_state.video_duration = 0.0


# ---------------- MOCK TRACKER (same as before) ----------------
def track_objects(video_path: str, video_duration: float):
    st.session_state.object_memory = {}

    object_memory = {}

    id_1 = 'T-1'
    object_memory[id_1] = {'id': id_1,
                           'class': 'person', 'color': 'Green', 'history': []}
    for t in np.arange(0, video_duration, TARGET_FPS):
        t = round(t, 2)
        x = int((t * 50) % 200)
        object_memory[id_1]['history'].append(
            {'time': t, 'bbox': {'x': x, 'y': 50, 'w': 30, 'h': 80}})

    id_2 = 'T-2'
    start_time_car = video_duration / 2
    object_memory[id_2] = {'id': id_2,
                           'class': 'car', 'color': 'Red', 'history': []}
    for t in np.arange(start_time_car, video_duration, TARGET_FPS):
        t = round(t, 2)
        object_memory[id_2]['history'].append(
            {'time': t, 'bbox': {'x': 300, 'y': 150, 'w': 100, 'h': 50}})

    st.session_state.object_memory = object_memory
    return object_memory


def get_action_context(start_time: float, end_time: float) -> str:
    duration = end_time - start_time

    if duration >= 4 and start_time < 2:
        return "Sustained movement over early timeline."
    elif start_time < 5 and duration <= 3:
        return "Fast short-duration action."
    elif start_time > 5 and duration <= 2:
        return "Isolated short action."

    return "Moderate activity detected."


# --------------- MEMORY FUNCTIONS -----------------

def get_memory_summary() -> str:
    memory = st.session_state.object_memory

    if not memory:
        return "No memory loaded yet."

    summary = "--- Object Tracks ---\n"
    for track_id, track in memory.items():
        if track['history']:
            last = track['history'][-1]
            summary += (
                f"ID: {track['id']} | Class: {track['class']} | "
                f"Color: {track['color']} | Last Seen: {last['time']:.1f}s\n"
            )
    return summary


def find_tracks_by_attributes(cls: str, color: str):
    results = []
    for track in st.session_state.object_memory.values():
        ok = True
        if cls and track['class'].lower() != cls.lower():
            ok = False
        if color and track['color'].lower() != color.lower():
            ok = False
        if ok:
            results.append(track)
    return results


# ---------------- VIDEO UTILITIES -------------------

def parse_time_and_attributes(question: str, video_duration: float) -> dict:
    q_lower = question.lower()
    start_time, end_time = 0.0, video_duration
    clean_q = question
    import re

    m_to = re.search(r"from\s*(\d+\.?\d*)\s*s?\s*to\s*(\d+\.?\d*)", q_lower)
    m_at = re.search(r"at\s*(\d+\.?\d*)", q_lower)

    if m_to:
        start, end = float(m_to.group(1)), float(m_to.group(2))
        start_time, end_time = min(start, end), max(start, end)
        clean_q = re.sub(r"from.*?to.*?", "", question,
                         flags=re.IGNORECASE).strip()

    elif m_at:
        t = float(m_at.group(1))
        start_time, end_time = t, t + 0.5
        clean_q = re.sub(r"at.*?", "", question, flags=re.IGNORECASE).strip()

    classes = ['person', 'car', 'truck', 'dog']
    colors = ['red', 'green', 'blue', 'yellow',
              'purple', 'brown', 'white', 'black']

    target_class = next((c for c in classes if c in q_lower), None)
    target_color = next((c for c in colors if c in q_lower), None)

    start_time = max(0, start_time)
    end_time = min(video_duration, end_time)

    return dict(
        start_time=start_time,
        end_time=end_time,
        clean_question=clean_q,
        target_class=target_class,
        target_color=target_color
    )


def get_frames_base64(video_path: str, start_time: float, end_time: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = end_time - start_time
    total_samples = min(MAX_FRAMES_FOR_VLM, max(
        1, int(duration * VLM_SAMPLE_RATE)))
    step_time = duration / total_samples

    frames = []
    for i in range(total_samples):
        time_seek = start_time + i * step_time
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_seek * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = BytesIO()
        pil_img.save(buf, "JPEG")
        data = base64.b64encode(buf.getvalue()).decode()

        frames.append({"inlineData": {"mimeType": "image/jpeg", "data": data}})

    cap.release()
    return frames


# ---------------- GEMINI API -------------------

def fetch_with_exponential_backoff(url, options, retries=0):
    try:
        r = requests.post(
            url, headers=options['headers'], json=options['json'])
        r.raise_for_status()
        return r
    except Exception as e:
        if retries < MAX_RETRIES:
            time.sleep(2 ** retries)
            return fetch_with_exponential_backoff(url, options, retries + 1)
        raise e


def answer_generative(frames, prompt: str):
    payload = {
        "contents": [
            {"role": "user", "parts": frames + [{"text": prompt}]}
        ],
        "systemInstruction": {
            "parts": [{"text": "You analyze surveillance frames. Respond factually."}]
        }
    }

    opt = {'headers': {'Content-Type': 'application/json'}, 'json': payload}
    res = fetch_with_exponential_backoff(API_URL, opt).json()

    return (
        res.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "No answer returned.")
    )


# ---------------- ORCHESTRATION -------------------

def handle_query_orchestration(uploaded_file, question):

    if not uploaded_file:
        st.error("Upload a video first.")
        return

    if not question.strip():
        st.error("Enter a query.")
        return

    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(uploaded_file.read())
        temp_video = f.name

    parsed = parse_time_and_attributes(
        question, st.session_state.video_duration)

    q_lower = question.lower()

    if ("how many" in q_lower) or ("count" in q_lower):
        count = len(st.session_state.object_memory)
        st.session_state.output_answer = (
            f"Count: {count} objects detected."
        )
        os.unlink(temp_video)
        return

    target_class = parsed['target_class']
    target_color = parsed['target_color']

    query_type = "Temporal + Generative"
    temporal_ctx = get_action_context(parsed['start_time'], parsed['end_time'])

    final_prompt = (
        f"Context: {temporal_ctx}. "
        f"Question: {parsed['clean_question']}"
    )

    if target_class or target_color:
        tracks = find_tracks_by_attributes(target_class, target_color)
        if tracks:
            query_type = "Memory + Temporal + Generative"
            summary = ""
            for t in tracks:
                last = t['history'][-1]
                summary += f"Track {t['id']} last seen at {last['time']}s. "
            final_prompt = (
                f"Memory: {summary} "
                f"Context: {temporal_ctx}. "
                f"Question: {parsed['clean_question']}"
            )
        else:
            st.session_state.output_answer = (
                "No object matches the given description."
            )
            os.unlink(temp_video)
            return

    frames = get_frames_base64(
        temp_video, parsed['start_time'], parsed['end_time']
    )

    answer = answer_generative(frames, final_prompt)

    st.session_state.output_answer = (
        f"Query Type: {query_type}\n\n"
        f"Temporal Context: {temporal_ctx}\n\n"
        f"Answer:\n{answer}"
    )

    os.unlink(temp_video)


# ---------------------- UI -----------------------

st.title("Video Intelligence System")
st.write("A professional interface for video understanding and query-based analysis.")

left, right = st.columns([1.2, 1.6])

with left:
    st.subheader("1. Upload Video")

    uploaded_file = st.file_uploader(
        "Upload CCTV or camera footage",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        st.video(uploaded_file)

        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(uploaded_file.read())
            temp_path = f.name

        cap = cv2.VideoCapture(temp_path)
        if cap.isOpened():
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.session_state.video_duration = frames / fps
            st.success(f"Video Duration: {st.session_state.video_duration:.1f} seconds")
            cap.release()

            if not st.session_state.object_memory:
                track_objects(temp_path, st.session_state.video_duration)
                st.info("Object memory initialized.")

        os.unlink(temp_path)

with right:
    st.subheader("2. Ask a Query")
    q = st.text_input("", placeholder="Example: What is the green person doing from 2s to 5s?")

    if st.button("Analyze"):
        if uploaded_file:
            handle_query_orchestration(uploaded_file, q)
        else:
            st.error("Upload a video first.")

st.subheader("Analysis Result")
if 'output_answer' in st.session_state:
    st.code(st.session_state.output_answer)
else:
    st.info("Results will appear here.")

st.subheader("Object Memory")
st.code(get_memory_summary())
