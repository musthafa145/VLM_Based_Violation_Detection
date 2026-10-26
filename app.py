import json
import os
import tempfile
import time
from io import BytesIO

import cv2
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(
    page_title="VLM Video Safety Assessment",
    page_icon="🎥",
    layout="wide",
)

st.title("VLM-Based Video Violation Detection")
st.caption(
    "Video safety assessment using a vision-language model. "
    "No YOLO or separate object detector is used."
)
st.warning(
    "Prototype only. Results may be incorrect or incomplete. "
    "Human review is required before enforcement decisions."
)

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Check your .env file.")
    st.stop()

st.subheader("1. Configure the assessment")
rule = st.selectbox(
    "Safety rule",
    [
        "Workers must wear protective safety helmets.",
        "Workers must wear high-visibility safety vests.",
        "Workers must wear both safety helmets and high-visibility vests.",
    ],
)

interval = st.select_slider(
    "Sample one frame every",
    options=[1, 2, 3, 5],
    value=1,
    format_func=lambda x: f"{x} second{'s' if x != 1 else ''}",
)

max_frames = st.selectbox(
    "Maximum frames to assess",
    [10, 20, 30],
    index=1,
)

st.subheader("2. Upload video footage")
uploaded_video = st.file_uploader(
    "Choose a construction-site video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is not None:
    st.video(uploaded_video)
    st.caption(
        f"File: {uploaded_video.name} · "
        f"Size: {uploaded_video.size / (1024 * 1024):.1f} MB"
    )


def timestamp_text(seconds):
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def assess_frame(
    client,
    image_bytes,
    timestamp,
    safety_rule,
    previous_image_bytes=None,
    previous_timestamp=None,
):
    """Assess the current frame, optionally comparing it with the prior sample."""
    prompt = f"""
You are a cautious construction-site safety auditor analysing video frames.

SAFETY RULE: {safety_rule}
CURRENT TIMESTAMP: {timestamp}

You will receive the CURRENT frame and, if available, the PREVIOUS sampled
frame from the same video. Use the previous frame as context, but make the
current frame's visible evidence the basis of the current decision.

ASSESS WORKERS INDIVIDUALLY. Do not assume everyone is compliant because
some workers wear helmets. Do not transfer one worker's PPE status to another.
Compare the same worker across frames using clothing, position, and visible
appearance. Do not claim identity or an action if the comparison is unclear.

HEAD VISIBILITY:
Before deciding that a worker is missing a helmet, the entire head from top
to chin, including both sides, must be clearly visible and unobstructed.
The head must be inside the image with visible background space around it.
Prefer at least a 5% image-width margin from left/right edges and a 5% image-
height margin from top/bottom edges, where applicable. If the head is cropped,
obscured, too small, blurry, or too close to an edge, do not claim a violation
for that worker. A helmet that cannot be seen is not proof that it is absent.

TEMPORAL CONTEXT:
If the same worker clearly removes a helmet between the previous and current
frames and is still clearly bareheaded in the current frame, classify the
current frame as VIOLATION. If a worker remains clearly bareheaded in later
frames, continue to report VIOLATION while the full head is visible. If the
removal, worker match, or current head visibility is unclear, use UNCERTAIN.
Do not infer that a worker put a helmet back on unless it is visible.

DECISIONS:
- VIOLATION: Clear evidence that the required PPE is absent, with the
  relevant body area fully visible and assessable.
- COMPLIANT: The required PPE is clearly visible on the worker being assessed.
- UNCERTAIN: Visibility or PPE status is insufficient to decide reliably.

For a frame with multiple workers, describe the relevant worker(s) and do
not generalise from background workers. If a violation is clearly visible
for at least one assessable worker, report it even if other workers are
compliant. If no violation can be established and the relevant evidence is
unclear, use UNCERTAIN rather than guessing.

Return only valid JSON with exactly these fields:
{{
  "decision": "VIOLATION, COMPLIANT, or UNCERTAIN",
  "head_fully_visible": true,
  "evidence": "Describe the specific visible worker and evidence",
  "reasoning": "Explain the decision and any clear change from the previous frame",
  "recommended_action": "Suggest a proportionate next step"
}}

For head_fully_visible, return true only when the relevant worker's full head
is clearly visible and satisfies the visibility requirements; otherwise false.
If the selected rule is about vests rather than helmets, assess visibility of
the relevant body area and set head_fully_visible to true only if the head
also meets the head visibility requirements. This field is a model judgment.
"""

    contents = [prompt]
    if previous_image_bytes is not None:
        contents.extend(
            [
                f"PREVIOUS SAMPLED FRAME at {previous_timestamp}:",
                types.Part.from_bytes(
                    data=previous_image_bytes,
                    mime_type="image/jpeg",
                ),
            ]
        )
    contents.extend(
        [
            f"CURRENT FRAME at {timestamp}:",
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ]
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result = json.loads(response.text or "{}")

    allowed = {"VIOLATION", "COMPLIANT", "UNCERTAIN"}
    decision = str(result.get("decision", "UNCERTAIN")).upper().strip()
    head_fully_visible = result.get("head_fully_visible")
    if decision not in allowed:
        decision = "UNCERTAIN"

    # Safety gate: do not accept a violation unless the model explicitly
    # reports the relevant head is fully visible. This is not independent
    # verification; the visibility judgment still comes from the VLM.
    if (
        "helmet" in safety_rule.lower()
        and decision == "VIOLATION"
        and head_fully_visible is not True
    ):
        decision = "UNCERTAIN"

    return {
        "decision": decision,
        "head_fully_visible": head_fully_visible,
        "evidence": str(result.get("evidence", "No evidence supplied.")),
        "reasoning": str(result.get("reasoning", "")),
        "recommended_action": str(
            result.get("recommended_action", "Review this frame manually.")
        ),
    }


if uploaded_video is not None and st.button("Analyse Video", type="primary"):
    temp_path = None
    cap = None
    try:
        video_bytes = uploaded_video.getvalue()
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(uploaded_video.name)[1],
            delete=False,
        ) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Could not open this video. Try a standard MP4 file.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fps or fps <= 0 or frame_count <= 0:
            st.error("The video duration or frame rate could not be read.")
            st.stop()

        duration = frame_count / fps
        st.write(f"**Video duration:** {timestamp_text(duration)}")
        st.write(f"**Model:** `{MODEL_NAME}`")
        st.write(f"**Sampling interval:** {interval} second(s)")

        sample_times = list(range(0, int(duration) + 1, interval))[:max_frames]
        if not sample_times:
            sample_times = [0]

        client = genai.Client(api_key=API_KEY)
        results = []
        previous_image_bytes = None
        previous_timestamp = None
        progress = st.progress(0)
        status = st.empty()

        for index, seconds in enumerate(sample_times):
            frame_number = min(
                int(seconds * fps),
                max(0, int(frame_count) - 1),
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
            timestamp = timestamp_text(seconds)

            if not success:
                results.append(
                    {
                        "timestamp": timestamp,
                        "decision": "UNCERTAIN",
                        "head_fully_visible": None,
                        "evidence": "Could not read this video frame.",
                        "reasoning": "Frame extraction failed.",
                        "recommended_action": "Check the video file.",
                        "image": None,
                        "error": "Frame extraction failed.",
                    }
                )
                progress.progress((index + 1) / len(sample_times))
                continue

            height, width = frame.shape[:2]
            max_dimension = 1024
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()

            status.write(
                f"Assessing frame {index + 1}/{len(sample_times)} at {timestamp}..."
            )
            error_message = None
            try:
                assessment = assess_frame(
                    client=client,
                    image_bytes=image_bytes,
                    timestamp=timestamp,
                    safety_rule=rule,
                    previous_image_bytes=previous_image_bytes,
                    previous_timestamp=previous_timestamp,
                )
            except Exception as exc:
                assessment = {
                    "decision": "UNCERTAIN",
                    "head_fully_visible": None,
                    "evidence": "The model assessment did not complete.",
                    "reasoning": "An API or response-processing error occurred.",
                    "recommended_action": "Retry this frame or review it manually.",
                }
                error_message = str(exc)

            results.append(
                {
                    "timestamp": timestamp,
                    **assessment,
                    "image": pil_image,
                    "error": error_message,
                }
            )

            # Pass this sampled frame as context to the next sampled frame.
            previous_image_bytes = image_bytes
            previous_timestamp = timestamp
            progress.progress((index + 1) / len(sample_times))
            if index < len(sample_times) - 1:
                time.sleep(1)

        cap.release()
        cap = None
        status.empty()
        progress.empty()
        st.session_state["video_results"] = results
        st.session_state["video_rule"] = rule
        st.success("Video assessment finished.")

    except Exception as exc:
        st.error(f"Video processing failed: {exc}")
    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if "video_results" in st.session_state:
    results = st.session_state["video_results"]
    st.divider()
    st.subheader("3. Assessment report")

    violations = sum(item["decision"] == "VIOLATION" for item in results)
    compliant = sum(item["decision"] == "COMPLIANT" for item in results)
    uncertain = sum(item["decision"] == "UNCERTAIN" for item in results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Potential violations", violations)
    col2.metric("Compliant frames", compliant)
    col3.metric("Uncertain frames", uncertain)
    st.caption("Counts represent sampled frames, not unique violations or people.")

    for item in results:
        with st.expander(
            f"{item['timestamp']} — {item['decision']}",
            expanded=(item["decision"] == "VIOLATION"),
        ):
            left, right = st.columns([1, 2])
            with left:
                if item.get("image") is not None:
                    st.image(item["image"], use_container_width=True)
            with right:
                st.write(f"**Rule:** {st.session_state['video_rule']}")
                st.write(f"**Visible evidence:** {item['evidence']}")
                st.write(f"**Reasoning:** {item['reasoning']}")
                st.write(f"**Recommended action:** {item['recommended_action']}")
                if item.get("error"):
                    st.warning(f"Technical error: {item['error']}")

    report = [
        {
            "timestamp": item["timestamp"],
            "decision": item["decision"],
            "head_fully_visible": item.get("head_fully_visible"),
            "rule": st.session_state["video_rule"],
            "evidence": item["evidence"],
            "reasoning": item["reasoning"],
            "recommended_action": item["recommended_action"],
            "error": item.get("error"),
        }
        for item in results
    ]
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name="video_safety_report.json",
        mime="application/json",
    )
    st.info(
        "Review flagged and uncertain frames manually. This prototype does not "
        "guarantee detection of every violation. Previous-frame comparison adds "
        "context but does not guarantee worker identity tracking."
    )
