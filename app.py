import json
import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

_prompt_path = Path(__file__).with_name("prompt.py")
_prompt_spec = importlib.util.spec_from_file_location("snapstudy_prompt", _prompt_path)
if _prompt_spec is None or _prompt_spec.loader is None:
    raise ImportError(f"Could not load prompt module from {_prompt_path}")
_prompt_module = importlib.util.module_from_spec(_prompt_spec)
_prompt_spec.loader.exec_module(_prompt_module)
SYSTEM_PROMPT = _prompt_module.SYSTEM_PROMPT


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"].strip()

GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

DB_PATH = "snapstudy.db"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Snap & Study AI",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS study_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            subject TEXT,
            summary TEXT,
            result_json TEXT,
            created_at TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def save_history(result):

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        INSERT INTO study_history
        (
            title,
            subject,
            summary,
            result_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            result.get("title", ""),
            result.get("subject", ""),
            result.get("summary", ""),
            json.dumps(
                result,
                ensure_ascii=False
            ),
            datetime.now().isoformat(
                timespec="seconds"
            ),
        ),
    )

    connection.commit()
    connection.close()


def get_history():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    rows = connection.execute(
        """
        SELECT *
        FROM study_history
        ORDER BY id DESC
        LIMIT 20
        """
    ).fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def clear_history():

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        "DELETE FROM study_history"
    )

    connection.commit()
    connection.close()


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is missing.\n\n"
            "Add your Gemini API key to "
            ".streamlit/secrets.toml"
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# JSON PARSER
# ============================================================

def clean_json(text):

    text = text.strip()

    if text.startswith("```"):

        if text.startswith("```json"):
            text = text[7:]

        else:
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

    text = text.strip()

    return json.loads(text)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_bytes,
    mime_type,
    student_instruction
):

    client = get_gemini_client()

    instruction = (
        student_instruction.strip()
    )

    if not instruction:

        instruction = (
            "Analyze this image and teach me "
            "the content clearly."
        )

    prompt = f"""
STUDENT REQUEST:

{instruction}
"""

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],

        config=types.GenerateContentConfig(

            temperature=0.2,

            response_mime_type="application/json",

            system_instruction=SYSTEM_PROMPT,
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    result = clean_json(
        response.text
    )

    return result


# ============================================================
# FOLLOW-UP CHAT
# ============================================================

def ask_followup(
    result,
    question
):

    client = get_gemini_client()

    previous_result = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are continuing a Snap & Study session.

Here is the previous analysis:

{previous_result}

The student asks:

{question}

Answer the student clearly.

Rules:

- Stay focused on the uploaded study material.
- Do not invent information.
- If the previous analysis does not contain enough information,
  say that clearly.
- Explain technical concepts step by step.
- Use simple language when possible.
"""

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt,

        config=types.GenerateContentConfig(
            temperature=0.3
        ),
    )

    return response.text or (
        "I could not generate a response."
    )


# ============================================================
# DOWNLOAD TEXT
# ============================================================

def create_download_text(result):

    text = ""

    text += "SNAP & STUDY AI\n"
    text += "=" * 50
    text += "\n\n"

    text += (
        f"Title: {result.get('title', '')}\n"
    )

    text += (
        f"Subject: {result.get('subject', '')}\n"
    )

    text += (
        f"Concept: {result.get('concept', '')}\n\n"
    )

    text += "SUMMARY\n"
    text += "-" * 50
    text += "\n"

    text += result.get(
        "summary",
        ""
    )

    text += "\n\n"

    text += "KEY POINTS\n"
    text += "-" * 50
    text += "\n"

    for point in result.get(
        "key_points",
        []
    ):

        text += f"- {point}\n"

    text += "\n"

    text += "STEPS\n"
    text += "-" * 50
    text += "\n"

    for index, step in enumerate(
        result.get("steps", []),
        start=1
    ):

        text += (
            f"{index}. {step}\n"
        )

    text += "\n"

    text += "ANSWER\n"
    text += "-" * 50
    text += "\n"

    text += result.get(
        "answer",
        "No final answer provided."
    )

    text += "\n\n"

    text += "IMPORTANT TERMS\n"
    text += "-" * 50
    text += "\n"

    for term in result.get(
        "important_terms",
        []
    ):

        text += f"- {term}\n"

    text += "\n"

    text += "CONFIDENCE\n"
    text += "-" * 50
    text += "\n"

    text += result.get(
        "confidence",
        "unknown"
    )

    text += "\n"

    return text


# ============================================================
# INITIALIZE
# ============================================================

initialize_database()


if "analysis" not in st.session_state:

    st.session_state.analysis = None


if "chat_messages" not in st.session_state:

    st.session_state.chat_messages = []


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #f8fafc,
            #eef2ff
        );
    }

    .hero {

        padding: 28px;

        border-radius: 20px;

        background:
        linear-gradient(
            135deg,
            #111827,
            #2563eb
        );

        color: white;

        margin-bottom: 25px;

        box-shadow:
        0 15px 40px
        rgba(37,99,235,0.18);
    }

    .hero-title {

        font-size: 36px;

        font-weight: 800;

        margin-bottom: 6px;
    }

    .hero-subtitle {

        color: #dbeafe;

        font-size: 16px;
    }

    .result-card {

        background: white;

        padding: 22px;

        border-radius: 18px;

        border:
        1px solid #e2e8f0;

        box-shadow:
        0 10px 30px
        rgba(15,23,42,0.06);

        margin-bottom: 18px;
    }

    .section-title {

        font-size: 18px;

        font-weight: 700;

        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            📚 Snap & Study AI
        </div>

        <div class="hero-subtitle">
            Snap a question → Understand it → Learn it
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Snap & Study")

    st.write(
        "Turn a photo of study material "
        "into an understandable explanation."
    )

    st.divider()

    st.subheader("Workflow")

    st.write("📷 1. Upload")
    st.write("🧠 2. Analyze")
    st.write("📖 3. Understand")
    st.write("💬 4. Ask")
    st.write("💾 5. Save")

    st.divider()

    if st.button(
        "🗑️ Clear Study History",
        use_container_width=True
    ):

        clear_history()

        st.session_state.analysis = None

        st.session_state.chat_messages = []

        st.success(
            "Study history cleared."
        )

        st.rerun()


# ============================================================
# MAIN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE
# ============================================================

with left_column:

    st.subheader(
        "📷 Upload Study Material"
    )

    uploaded_file = st.file_uploader(

        "Choose an image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],

        help=(
            "Upload a clear image of "
            "your question, notes, diagram, "
            "or textbook page."
        ),
    )

    student_instruction = st.text_area(

        "What do you want me to do?",

        placeholder=(
            "Examples:\n"
            "• Explain this like a beginner.\n"
            "• Solve this step by step.\n"
            "• Explain this circuit diagram.\n"
            "• Give me important exam points."
        ),

        height=130,
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded study material",
            use_container_width=True,
        )

    analyze_button = st.button(

        "🧠 Analyze with Gemini",

        type="primary",

        use_container_width=True,

        disabled=uploaded_file is None,
    )

    if analyze_button and uploaded_file:

        try:

            with st.spinner(
                "Reading and understanding the image..."
            ):

                result = analyze_image(

                    image_bytes=
                    uploaded_file.getvalue(),

                    mime_type=
                    uploaded_file.type
                    or "image/jpeg",

                    student_instruction=
                    student_instruction,
                )

            st.session_state.analysis = result

            st.session_state.chat_messages = []

            save_history(result)

            st.success(
                "Study material analyzed successfully."
            )

        except json.JSONDecodeError:

            st.error(
                "Gemini returned an invalid JSON response. "
                "Please try the image again."
            )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


# ============================================================
# RIGHT SIDE
# ============================================================

with right_column:

    st.subheader(
        "📖 Study Explanation"
    )

    result = st.session_state.analysis

    if result is None:

        st.info(
            "Upload a study image and click "
            "'Analyze with Gemini'."
        )

    else:

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"## {result.get('title', 'Study Topic')}"
        )

        st.write(
            f"**Subject:** "
            f"{result.get('subject', 'Unknown')}"
        )

        st.write(
            f"**Concept:** "
            f"{result.get('concept', 'Unknown')}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 🧠 Simple Explanation"
        )

        st.write(
            result.get(
                "summary",
                "No summary available."
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # KEY POINTS
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 📌 Key Points"
        )

        key_points = result.get(
            "key_points",
            []
        )

        for point in key_points:

            st.markdown(
                f"- {point}"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # STEPS
        # ----------------------------------------------------

        steps = result.get(
            "steps",
            []
        )

        if steps:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 🔢 Step-by-Step"
            )

            for index, step in enumerate(
                steps,
                start=1
            ):

                st.markdown(
                    f"**Step {index}:** {step}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### ✅ Answer"
        )

        st.write(
            result.get(
                "answer",
                "No final answer available."
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # IMPORTANT TERMS
        # ----------------------------------------------------

        terms = result.get(
            "important_terms",
            []
        )

        if terms:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 📚 Important Terms"
            )

            for term in terms:

                st.markdown(
                    f"- `{term}`"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = result.get(
            "confidence",
            "unknown"
        )

        st.info(
            f"Analysis confidence: "
            f"**{confidence}**"
        )

        # ----------------------------------------------------
        # WARNINGS
        # ----------------------------------------------------

        warnings = result.get(
            "warnings",
            []
        )

        for warning in warnings:

            st.warning(
                warning
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        study_text = create_download_text(
            result
        )

        st.download_button(

            "⬇️ Download Study Notes",

            data=study_text,

            file_name="snap-study-notes.txt",

            mime="text/plain",

            use_container_width=True,
        )


# ============================================================
# FOLLOW-UP CHAT
# ============================================================

if st.session_state.analysis:

    st.divider()

    st.subheader(
        "💬 Ask Follow-up Questions"
    )

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    question = st.chat_input(
        "Ask something about this study material..."
    )

    if question:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    answer = ask_followup(

                        st.session_state.analysis,

                        question,
                    )

                    st.write(answer)

                except Exception as error:

                    answer = (
                        f"Error: {error}"
                    )

                    st.error(answer)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.subheader(
    "🗂️ Previous Study Sessions"
)

history = get_history()

if not history:

    st.caption(
        "No study sessions saved yet."
    )

else:

    for item in history:

        title = (
            item["title"]
            or "Untitled"
        )

        subject = (
            item["subject"]
            or "Unknown subject"
        )

        created = item["created_at"]

        with st.expander(
            f"{title} • {subject} • {created}"
        ):

            st.write(
                item["summary"]
            )

            try:

                old_result = json.loads(
                    item["result_json"]
                )

                st.json(
                    old_result
                )

            except Exception:

                st.write(
                    item["result_json"]
                )