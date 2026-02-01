"""Simple Streamlit app for audio transcription."""

import tempfile
from pathlib import Path

import streamlit as st

from transcribe.audio import convert_to_whisper_format, is_whisper_compatible
from transcribe.config import MODEL_DIR, SUPPORTED_MODELS
from transcribe.transcriber import Transcriber, DependencyError

st.set_page_config(page_title="Transcribe", page_icon="🎙️")

st.title("🎙️ Audio Transcription")


@st.cache_resource
def get_transcriber(model: str):
    """Cache the transcriber instance."""
    return Transcriber(model=model)


# Check for model
def get_available_models():
    """Return list of models that are downloaded."""
    available = []
    for model in SUPPORTED_MODELS:
        model_path = MODEL_DIR / f"ggml-{model}.bin"
        if model_path.exists():
            available.append(model)
    return available


available_models = get_available_models()

if not available_models:
    st.error("No whisper models found. Run `transcribe --bootstrap` first.")
    st.code("transcribe --bootstrap")
    st.stop()

# Model selection
model = st.selectbox("Model", available_models, index=0)

# File upload
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav", "m4a", "flac", "ogg", "aac", "wma", "opus"],
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe", type="primary"):
        with st.spinner("Transcribing..."):
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)

            try:
                # Get transcriber
                transcriber = get_transcriber(model)

                # Convert if needed
                if not is_whisper_compatible(tmp_path):
                    wav_path = convert_to_whisper_format(tmp_path)
                    cleanup_wav = True
                else:
                    wav_path = tmp_path
                    cleanup_wav = False

                # Transcribe
                output_path = tmp_path.with_suffix(".txt")
                result = transcriber.transcribe(wav_path, output_path, "txt")

                if result.success:
                    # Read transcript
                    transcript = output_path.read_text()
                    st.session_state.transcript = transcript
                    st.session_state.filename = Path(uploaded_file.name).stem + ".txt"

                    # Cleanup
                    output_path.unlink(missing_ok=True)
                else:
                    st.error(f"Transcription failed: {result.error}")

                # Cleanup temp files
                if cleanup_wav:
                    wav_path.unlink(missing_ok=True)
                tmp_path.unlink(missing_ok=True)

            except DependencyError as e:
                st.error(str(e))

# Show transcript if available
if "transcript" in st.session_state:
    st.subheader("Transcript")
    st.text_area(
        "transcript_text",
        st.session_state.transcript,
        height=300,
        label_visibility="hidden",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download",
            st.session_state.transcript,
            file_name=st.session_state.filename,
            mime="text/plain",
        )
    with col2:
        # Copy button using JS
        st.markdown(
            """
            <button onclick="navigator.clipboard.writeText(document.querySelector('textarea').value); this.textContent='Copied!'">
                Copy to clipboard
            </button>
            """,
            unsafe_allow_html=True,
        )
