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

assessment_mode = st.selectbox(
    "Assessment type",
    [
        "PPE compliance",
        "Fire and smoke detection",
        "Restricted-area entry detection",
    ],
)

if assessment_mode == "PPE compliance":
    rule = st.selectbox(
        "Safety rule",
        [
            "Workers must wear protective safety helmets.",
            "Workers must wear high-visibility safety vests.",
            "Workers must wear both safety helmets and high-visibility vests.",
        ],
    )
elif assessment_mode == "Fire and smoke detection":
    rule = st.selectbox(
        "Hazard to assess",
        [
            "Detect fire and smoke",
            "Detect fire only",
            "Detect smoke only",
        ],
    )

else:
    rule = "Detect entry into an area marked RESTRICTED AREA."
    st.info(
        "The model will assess visible entry into a sign-marked "
        "restricted area. It cannot determine authorization status."
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
    """Assess PPE compliance or fire/smoke hazards in a video frame."""

    if safety_rule.startswith("Detect entry into an area"):
        prompt = f"""
You are a cautious CCTV safety auditor.

CURRENT TIMESTAMP: {timestamp}
RULE: {safety_rule}

You receive the current CCTV frame and, if available, the previous
sampled frame from the same video.

Assess only visible evidence of a person entering an area marked by
a sign that clearly reads "RESTRICTED AREA".

Look for:
- A visible sign reading "RESTRICTED AREA".
- The area that the sign appears to refer to.
- A person visibly crossing into that area.
- Previous-frame context, if available, to help determine whether
  the person moved from outside to inside the indicated area.

DECISIONS:
- ENTRY_OBSERVED: A person is visibly entering or has clearly crossed
  into the sign-marked area, supported by the available frame evidence.
- NO_ENTRY_OBSERVED: The sign and relevant area are sufficiently visible,
  but no entry is observed in the current evidence.
- UNCERTAIN: The sign, relevant area, person's movement, or boundary
  cannot be determined reliably.

IMPORTANT:
- Do not infer authorization status from clothing, appearance, or PPE.
- Do not assume a person is unauthorized merely because they lack PPE.
- Do not claim entry if the person is only standing near the sign.
- If the sign or the relevant area is unclear, use UNCERTAIN.
- A single frame may not establish whether entry occurred. Use the
  previous frame as context when available.
- Describe only what is visible.

Return only valid JSON with exactly these fields:
{{
  "decision": "ENTRY_OBSERVED, NO_ENTRY_OBSERVED, or UNCERTAIN",
  "evidence": "Describe the visible sign, person, and observed movement",
  "reasoning": "Explain the decision using visible evidence",
  "recommended_action": "Suggest proportionate human verification"
}}
"""
    elif safety_rule.startswith("Detect"):
        if safety_rule == "Detect fire and smoke":
            hazard_scope = "Assess both fire and smoke."
        elif safety_rule == "Detect fire only":
            hazard_scope = "Assess fire only. Do not report smoke alone as a hazard."
        else:
            hazard_scope = "Assess smoke only. Do not report fire alone as a hazard."

        prompt = f"""
You are a cautious construction-site fire and smoke safety auditor.

ASSESSMENT SCOPE: {hazard_scope}
CURRENT TIMESTAMP: {timestamp}

You receive the current video frame and, if available, the previous
sampled frame from the same video. Use the previous frame only as context.
Base the current decision on visible evidence in the current frame.

DEFINITIONS:
- FIRE: Visible flames or a clearly identifiable active fire.
- SMOKE: A visible smoke plume or smoke-like emission. Do not automatically
  label dust, steam, fog, clouds, or ordinary exhaust as smoke.
- NO_HAZARD: No visible evidence of the selected hazard in this frame.
- UNCERTAIN: The image is ambiguous, obscured, blurry, or insufficient
  to make a reliable decision.

RULES:
- Do not infer a fire from orange lighting, sparks, or hot-looking objects alone.
- Do not infer smoke from dust, steam, fog, clouds, or exhaust without
  sufficient visual evidence.
- Do not invent the source, cause, location, or severity of a hazard.
- If a hazard is only suspected, classify it as UNCERTAIN.
- If both fire and smoke are clearly visible and both are in scope, report BOTH.
- Describe only what is actually visible.
- A sampled frame cannot establish that no hazard exists in the full video.

Return only valid JSON with exactly these fields:
{{
  "decision": "FIRE, SMOKE, BOTH, NO_HAZARD, or UNCERTAIN",
  "evidence": "Specific visible evidence in the frame",
  "reasoning": "Brief explanation of the decision",
  "recommended_action": "Proportionate safety action"
}}

If the selected scope is fire only, return FIRE, NO_HAZARD, or UNCERTAIN.
If the selected scope is smoke only, return SMOKE, NO_HAZARD, or UNCERTAIN.
For ambiguous evidence, prefer UNCERTAIN rather than guessing.
"""
    else:
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

    evidence = str(result.get("evidence", "No evidence supplied."))
    reasoning = str(result.get("reasoning", ""))
    recommended_action = str(
        result.get("recommended_action", "Review this frame manually.")
    )

    if safety_rule.startswith("Detect entry into an area"):
        allowed = {
            "ENTRY_OBSERVED",
            "NO_ENTRY_OBSERVED",
            "UNCERTAIN",
        }
        decision = str(result.get("decision", "UNCERTAIN")).upper().strip()

        if decision not in allowed:
            decision = "UNCERTAIN"

        return {
            "decision": decision,
            "head_fully_visible": None,
            "evidence": evidence,
            "reasoning": reasoning,
            "recommended_action": recommended_action,
        }

    if safety_rule.startswith("Detect"):
        allowed = {"FIRE", "SMOKE", "BOTH", "NO_HAZARD", "UNCERTAIN"}
        decision = str(result.get("decision", "UNCERTAIN")).upper().strip()

        if decision not in allowed:
            decision = "UNCERTAIN"

        if safety_rule == "Detect fire only" and decision in {"SMOKE", "BOTH"}:
            decision = "UNCERTAIN"
        elif safety_rule == "Detect smoke only" and decision in {"FIRE", "BOTH"}:
            decision = "UNCERTAIN"

        return {
            "decision": decision,
            "head_fully_visible": None,
            "evidence": evidence,
            "reasoning": reasoning,
            "recommended_action": recommended_action,
        }

    allowed = {"VIOLATION", "COMPLIANT", "UNCERTAIN"}
    decision = str(result.get("decision", "UNCERTAIN")).upper().strip()
    head_fully_visible = result.get("head_fully_visible")

    if decision not in allowed:
        decision = "UNCERTAIN"

    # A violation requires an explicit positive visibility judgment.
    if (
        "helmet" in safety_rule.lower()
        and decision == "VIOLATION"
        and head_fully_visible is not True
    ):
        decision = "UNCERTAIN"

    return {
        "decision": decision,
        "head_fully_visible": head_fully_visible,
        "evidence": evidence,
        "reasoning": reasoning,
        "recommended_action": recommended_action,
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

    current_rule = st.session_state["video_rule"]

    if current_rule.startswith("Detect entry into an area"):
        entry_observed = sum(
            item["decision"] == "ENTRY_OBSERVED" for item in results
        )
        no_entry = sum(
            item["decision"] == "NO_ENTRY_OBSERVED" for item in results
        )
        uncertain = sum(
            item["decision"] == "UNCERTAIN" for item in results
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Entry observed", entry_observed)
        col2.metric("No entry observed", no_entry)
        col3.metric("Uncertain frames", uncertain)

        st.caption(
            "Counts represent sampled frames, not unique events. "
            "Entry alerts require human verification."
        )

    elif current_rule.startswith("Detect"):
        fire_alerts = sum(
            item["decision"] in {"FIRE", "BOTH"} for item in results
        )
        smoke_alerts = sum(
            item["decision"] in {"SMOKE", "BOTH"} for item in results
        )
        no_hazard = sum(
            item["decision"] == "NO_HAZARD" for item in results
        )
        uncertain = sum(
            item["decision"] == "UNCERTAIN" for item in results
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fire alert frames", fire_alerts)
        col2.metric("Smoke alert frames", smoke_alerts)
        col3.metric("No hazard detected", no_hazard)
        col4.metric("Uncertain frames", uncertain)

        st.caption(
            "Counts represent sampled frames, not unique incidents. "
            "Alerts require human verification."
        )
    else:
        violations = sum(
            item["decision"] == "VIOLATION" for item in results
        )
        compliant = sum(
            item["decision"] == "COMPLIANT" for item in results
        )
        uncertain = sum(
            item["decision"] == "UNCERTAIN" for item in results
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Potential violations", violations)
        col2.metric("Compliant frames", compliant)
        col3.metric("Uncertain frames", uncertain)

        st.caption(
            "Counts represent sampled frames, not unique violations or people."
        )


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
            "assessment_mode": (
                "Restricted-area entry detection"
                if st.session_state["video_rule"].startswith(
                    "Detect entry into an area"
                )
                else (
                    "Fire and smoke detection"
                    if st.session_state["video_rule"].startswith("Detect")
                    else "PPE compliance"
                )
            ),
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

    current_rule = st.session_state["video_rule"]

    if current_rule.startswith("Detect entry into an area"):
        report_filename = "restricted_area_entry_report.json"
    elif current_rule.startswith("Detect"):
        report_filename = "fire_smoke_assessment_report.json"
    else:
        report_filename = "ppe_compliance_report.json"

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name=report_filename,
        mime="application/json",
    )

    st.info(
        "Review flagged and uncertain frames manually. This prototype does not "
        "guarantee detection of every violation. Previous-frame comparison adds "
        "context but does not guarantee worker identity tracking."
    )
