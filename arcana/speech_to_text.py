import streamlit as st
import io
import os
import tempfile
import shutil
from typing import Any, Optional, List, Tuple
import subprocess

import av
from av.audio.resampler import AudioResampler
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
try:
    # Prefer official ollama client if available
    from ollama import Client as OllamaClient  # type: ignore
except Exception:
    OllamaClient = None  # type: ignore

# Optional NVIDIA NeMo diarization with auto-install
def _ensure_nemo_installed(verbose: bool = False) -> bool:
    """Best-effort installer for NeMo on macOS/Linux (CPU by default).

    Steps:
    - Ensure torch (CPU build) is present
    - Ensure build deps: pip/setuptools/wheel/Cython
    - Ensure soundfile
    - Install nemo_toolkit
    Returns True on success.
    """
    def log(msg: str) -> None:
        if verbose:
            try:
                st.write(msg)
            except Exception:
                pass
    try:
        import nemo  # type: ignore  # noqa: F401
        import soundfile  # type: ignore  # noqa: F401
        from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
        return True
    except Exception:
        import sys, subprocess, platform
        py = sys.executable
        def pip_install(args):
            log(f"Installing: {' '.join(args)}")
            try:
                subprocess.check_call([py, "-m", "pip", "install", *args])
                return True
            except subprocess.CalledProcessError as e:
                log(f"Install failed: {' '.join(args)} -> {e}")
                return False

        # Upgrade pip tooling
        pip_install(["-U", "pip", "setuptools", "wheel", "Cython"])  # best-effort

        # Ensure torch (CPU). On macOS Apple Silicon, default CPU wheel works.
        try:
            import torch  # noqa: F401
        except Exception:
            if platform.system() == "Darwin":
                pip_install(["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]) or pip_install(["torch"])  # fallback
            else:
                pip_install(["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]) or pip_install(["torch"])  # generic fallback

        # Ensure soundfile
        pip_install(["soundfile"])  # provides libsndfile wheels

        # Install NeMo
        if not pip_install(["nemo_toolkit"]):
            # Fallback retry after forcing numpy upgrade
            pip_install(["-U", "numpy"])  # some builds need newer numpy
            if not pip_install(["nemo_toolkit"]):
                return False

        # Final import validation
        try:
            import nemo  # noqa: F401
            import soundfile  # noqa: F401
            from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
            return True
        except Exception as e:
            log(f"Post-install import failed: {e}")
            return False

try:
    from nemo.collections.asr.models import ClusteringDiarizer  # type: ignore
    _NEMO_AVAILABLE = True
except Exception:
    if _ensure_nemo_installed():
        try:
            from nemo.collections.asr.models import ClusteringDiarizer  # type: ignore
            _NEMO_AVAILABLE = True
        except Exception:
            ClusteringDiarizer = None  # type: ignore
            _NEMO_AVAILABLE = False
    else:
        ClusteringDiarizer = None  # type: ignore
        _NEMO_AVAILABLE = False

# Local ASR (OpenAI Whisper)
try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    whisper = None  # type: ignore

# Optional faster-whisper (CTranslate2)
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

from http import HTTPStatus  # retained import to avoid refactors elsewhere


class AudioRecorder(AudioProcessorBase):
    """Collect raw AudioFrames during recording for later encoding."""
    def __init__(self) -> None:
        self.audio_frames: List[av.AudioFrame] = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame)
        return frame

    def pop_frames(self) -> List[av.AudioFrame]:
        frames = self.audio_frames[:]
        self.audio_frames.clear()
        return frames


def _frame_duration_seconds(frame: av.AudioFrame) -> float:
    sr = getattr(frame, "sample_rate", 0) or 0
    if sr <= 0:
        return 0.0
    return float(frame.samples) / float(sr)


def _validate_wav_nonempty(path: str) -> Tuple[bool, float, float]:
    """Open a WAV file and verify it contains samples.

    Returns (ok, duration_sec, rms)
    """
    try:
        c = av.open(path)
        sr = 16000
        total_samples = 0
        energy = 0.0
        for frame in c.decode(audio=0):
            samples = getattr(frame, "samples", 0) or 0
            total_samples += int(samples)
            # approximate RMS using plane 0 as mono
            try:
                pcm = frame.to_ndarray().astype("float32")
                if pcm.ndim == 2:
                    pcm = pcm[0]
                # normalize int16 range if needed
                if pcm.max() > 1.0 or pcm.min() < -1.0:
                    pcm = pcm / 32768.0
                if pcm.size:
                    energy += float((pcm * pcm).mean())
            except Exception:
                pass
        c.close()
        duration = float(total_samples) / float(sr) if total_samples > 0 else 0.0
        rms = (energy if duration == 0 else energy)  # heuristic
        return (total_samples > 0 and duration >= 0.2, duration, rms)
    except Exception:
        return (False, 0.0, 0.0)


def encode_frames_to_wav_16k_mono(frames: List[av.AudioFrame]) -> Optional[str]:
    """Encode a list of AudioFrames to a temp 16kHz mono WAV file.

    Returns the temp file path, or None if frames empty.
    """
    if not frames:
        return None

    # Validate frames have some duration
    total_sec = sum(_frame_duration_seconds(f) for f in frames)
    if total_sec < 0.2:
        return None

    # Resample everything to s16/mono/16k
    resampler = AudioResampler(format="s16", layout="mono", rate=16000)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    out = av.open(tmp_path, mode="w", format="wav")
    out_stream = out.add_stream("pcm_s16le", rate=16000)

    try:
        for f in frames:
            # Ensure frame has sample_rate set
            if not getattr(f, "sample_rate", None):
                # default common rate if missing; we'll still resample below
                f.sample_rate = 48000  # type: ignore[attr-defined]
            # Ensure attributes present and packed
            if not getattr(f, "layout", None):
                f.layout = "mono"  # type: ignore[attr-defined]
            try:
                f = f.to_ndarray(format="s16")  # force materialization to ensure data
            except Exception:
                pass
            rs = resampler.resample(frame)
            # resampler may return a single frame or a list
            if rs is None:
                continue
            rs_frames = rs if isinstance(rs, list) else [rs]
            for r in rs_frames:
                packets = out_stream.encode(r)
                if packets:
                    out.mux(packets)
        # Flush
        packets = out_stream.encode(None)
        if packets:
            out.mux(packets)
    finally:
        out.close()
    # Validate final WAV has content
    ok, duration, _ = _validate_wav_nonempty(tmp_path)
    if not ok:
        os.remove(tmp_path)
        return None
    return tmp_path


def _which(cmd: str) -> Optional[str]:
    try:
        path = subprocess.check_output(["which", cmd], stderr=subprocess.STDOUT).decode().strip()
        return path or None
    except Exception:
        return None


def to_wav16k_mono_from_any(input_bytes: Optional[bytes] = None, input_path: Optional[str] = None) -> str:
    """Decode arbitrary audio/video to temp WAV 16kHz mono using PyAV.

    Accepts either bytes or a file path.
    Returns path to temp WAV file.
    """
    assert input_bytes or input_path, "Either input_bytes or input_path must be provided"

    container = av.open(io.BytesIO(input_bytes)) if input_bytes is not None else av.open(input_path)  # type: ignore[arg-type]

    resampler = AudioResampler(format="s16", layout="mono", rate=16000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    out = av.open(tmp_path, mode="w", format="wav")
    out_stream = out.add_stream("pcm_s16le", rate=16000)

    decoded_any = False
    try:
        # Try all audio streams (not only index 0)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise RuntimeError("No audio streams found in file")
        for astream in audio_streams:
            for frame in container.decode(astream):
                decoded_any = True
                rs = resampler.resample(frame)
                if rs is None:
                    continue
                rs_frames = rs if isinstance(rs, list) else [rs]
                for r in rs_frames:
                    packets = out_stream.encode(r)
                    if packets:
                        out.mux(packets)
        packets = out_stream.encode(None)
        if packets:
            out.mux(packets)
    finally:
        out.close()
        container.close()
    # Sanity check file has audio by checking size
    if not decoded_any or (os.path.exists(tmp_path) and os.path.getsize(tmp_path) < 1024):
        # Try ffmpeg fallback
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        ffmpeg = _which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Decoded audio is empty or too short, and ffmpeg is not available for fallback")
        # Prepare source path if bytes
        src_path = input_path
        cleanup_src = False
        if src_path is None and input_bytes is not None:
            tmp_src = tempfile.NamedTemporaryFile(suffix=".media", delete=False)
            tmp_src.write(input_bytes)
            tmp_src.flush()
            tmp_src.close()
            src_path = tmp_src.name
            cleanup_src = True
        out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = out_tmp.name
        out_tmp.close()
        cmd = [ffmpeg, "-y", "-i", src_path or "-", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            if cleanup_src and src_path:
                os.remove(src_path)
            raise RuntimeError("ffmpeg failed to convert media to WAV")
        if cleanup_src and src_path:
            os.remove(src_path)
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            os.remove(out_path)
            raise RuntimeError("ffmpeg produced empty output")
        ok2, _, _ = _validate_wav_nonempty(out_path)
        if not ok2:
            os.remove(out_path)
            raise RuntimeError("Converted audio is empty or too short after ffmpeg")
        return out_path
    ok, duration, _ = _validate_wav_nonempty(tmp_path)
    if not ok:
        os.remove(tmp_path)
        raise RuntimeError("Audio is empty or too short after conversion")
    return tmp_path


def _transcribe_with_faster_whisper(wav_path: str, model_name: str) -> Tuple[bool, str]:
    if WhisperModel is None:
        return False, "faster-whisper not installed. pip install faster-whisper"
    # Decide device/compute_type
    device = "cpu"
    compute_type = "int8"  # good default for CPU
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "int8_float16"  # fast and memory-efficient on GPU
    except Exception:
        pass
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        # Limit to zh/en by running two passes and comparing avg probability
        def run(lang: str) -> Tuple[str, float]:
            segments, info = model.transcribe(wav_path, language=lang, beam_size=1, vad_filter=True)
            text_parts = []
            scores = []
            for s in segments:
                text_parts.append(s.text)
                if hasattr(s, "avg_logprob") and s.avg_logprob is not None:
                    scores.append(s.avg_logprob)
            score = sum(scores) / max(1, len(scores)) if scores else -10.0
            return (" ".join(text_parts).strip(), score)
        zh_text, zh_score = run("zh")
        en_text, en_score = run("en")
        text = zh_text if zh_score >= en_score else en_text
        if not text:
            return False, "faster-whisper returned empty transcription."
        return True, text
    except Exception as e:
        return False, f"faster-whisper exception: {e}"


def transcribe_with_whisper(wav_path: str, model_name: str = "base") -> Tuple[bool, str]:
    """Transcribe using local OpenAI Whisper model.

    model_name: tiny|base|small|medium|large (default: base)
    """
    # Prefer faster-whisper if enabled and available
    if st.session_state.get("use_faster_whisper", True) and WhisperModel is not None:
        ok, text = _transcribe_with_faster_whisper(wav_path, model_name)
        if ok:
            return ok, text
        # Fall through to PyTorch Whisper on failure
    if whisper is None:
        return False, "Whisper is not installed. Install with: pip install -U openai-whisper (or faster-whisper)"
    try:
        # Load once and cache in session
        cache_key = f"_whisper_model_{model_name}"
        model = st.session_state.get(cache_key)
        if model is None:
            # Prefer CUDA when available
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            model = whisper.load_model(model_name, device=device)
            st.session_state[cache_key] = model
        # Strictly limit to Chinese and English.
        # Strategy: run two passes (zh,en), pick the one with higher avg logprob.
        # This avoids Whisper detecting other languages.
        def _run(lang_code: str) -> Tuple[Optional[dict], float]:
            try:
                r = model.transcribe(
                    wav_path,
                    fp16=not st.session_state.get("whisper_fp16_off", True),
                    task="transcribe",
                    language=lang_code,
                )
                segs = r.get("segments") or []
                if not segs:
                    return r, float("-inf")
                scores = [s.get("avg_logprob", -10.0) for s in segs if isinstance(s, dict)]
                score = sum(scores) / max(1, len(scores)) if scores else -10.0
                return r, score
            except Exception:
                return None, float("-inf")

        res_zh, score_zh = _run("zh")
        res_en, score_en = _run("en")
        best = res_zh if score_zh >= score_en else res_en
        result = best or res_en or res_zh or {}
        text = result.get("text", "").strip()
        if not text:
            return False, "Whisper returned empty transcription."
        return True, text
    except Exception as e:
        return False, f"Whisper exception: {e}"


def simple_speaker_diarize(wav_path: str, api_token: Optional[str] = None) -> Optional[List[Tuple[str, float, float]]]:
    """Placeholder diarization.

    Returns a list of tuples (speaker_label, start_sec, end_sec) or None if
    diarization not available. This is a stub you can wire to any external
    diarization service later. The token will be provided by you at runtime.
    """
    # Accept token via session or parameter (user will fill later as string)
    token = api_token or st.session_state.get("diarization_token")
    if not st.session_state.get("enable_diarization", False):
        return None
    if not token:
        # No token; still return a single-speaker span as a safe fallback
        try:
            c = av.open(wav_path)
            dur = float(c.duration or 0) / av.time_base if c.duration else 0.0
            c.close()
        except Exception:
            dur = 0.0
        return [("Speaker 1", 0.0, max(dur, 0.0))]

    # Stub: pretend we did diarization; produce two fake segments
    try:
        c = av.open(wav_path)
        dur = float(c.duration or 0) / av.time_base if c.duration else 0.0
        c.close()
    except Exception:
        dur = 0.0
    if dur <= 0.0:
        return [("Speaker 1", 0.0, 0.0)]
    half = dur / 2.0
    return [("Speaker 1", 0.0, half), ("Speaker 2", half, dur)]


def diarize_with_nemo(wav_path: str, expected_speakers: Optional[int] = None) -> Optional[List[Tuple[str, float, float]]]:
    """Run NVIDIA NeMo clustering diarization, returning [(speaker, start, end)].

    Requires nemo_toolkit[all]. Falls back to None if unavailable or on error.
    """
    if not st.session_state.get("enable_diarization", False):
        return None
    if not _NEMO_AVAILABLE:
        if _ensure_nemo_installed():
            try:
                global ClusteringDiarizer
                from nemo.collections.asr.models import ClusteringDiarizer as _CD  # type: ignore
                ClusteringDiarizer = _CD
            except Exception:
                st.warning("Unable to load NeMo after auto-install. Check environment.")
                return None
        else:
            st.warning("NeMo install failed. Run: pip install nemo_toolkit soundfile")
            return None
    try:
        import json, tempfile, os
        # Prepare minimal manifest
        manifest_entry = {
            "audio_filepath": wav_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": expected_speakers,
            "rttm_filepath": None,
            "uniq_id": "file1",
        }
        tmpdir = tempfile.mkdtemp(prefix="nemo_diar_")
        manifest_path = os.path.join(tmpdir, "input.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # Basic config: enable VAD + clustering with default parameters
        cfg = {
            "num_workers": 1,
            "manifest_filepath": manifest_path,
            "out_dir": out_dir,
            "vad": {"model_path": None, "parameters": {"shift_length_in_sec": 0.25, "window_length_in_sec": 0.5}},
            "speaker_embeddings": {"model_path": None},
            "clustering": {"max_num_speakers": expected_speakers or 8, "oracle_num_speakers": expected_speakers is not None},
        }
        diarizer = ClusteringDiarizer(cfg=cfg)  # type: ignore
        diarizer.diarize()

        # Parse RTTM from out_dir. NeMo writes a .rttm per file
        rttm_path = None
        for fn in os.listdir(out_dir):
            if fn.endswith(".rttm"):
                rttm_path = os.path.join(out_dir, fn)
                break
        if not rttm_path or not os.path.exists(rttm_path):
            return None
        segments: List[Tuple[str, float, float]] = []
        with open(rttm_path, "r", encoding="utf-8") as rf:
            for line in rf:
                parts = line.strip().split()
                # RTTM format: SPEAKER <file-id> 1 <start> <dur> <..> <..> <..> <spk>
                if len(parts) >= 9 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    dur = float(parts[4])
                    spk = parts[8]
                    segments.append((spk, start, start + dur))
        # Normalize labels to Speaker 1..N in chronological order
        label_map = {}
        next_id = 1
        norm: List[Tuple[str, float, float]] = []
        for spk, s, e in sorted(segments, key=lambda x: (x[1], x[2])):
            if spk not in label_map:
                label_map[spk] = f"Speaker {next_id}"
                next_id += 1
            norm.append((label_map[spk], s, e))
        return norm or None
    except Exception as e:
        st.error(f"NeMo diarization error: {e}")
        return None


def render_transcript(text: str, speaker_segments: Optional[List[Tuple[str, float, float]]] = None):
    st.markdown("### Results")
    tabs = st.tabs(["Raw", "Refined"])

    with tabs[0]:
        with st.container(border=True):
            st.subheader("📝 Raw Transcription")
            # If we have diarized segments, render a simple speaker-formatted block
            if speaker_segments:
                # Very simple heuristic: split transcript into N equal chunks matching segments
                # so we can prefix with speaker labels. This is only a placeholder until
                # word-level alignment is available.
                segs = speaker_segments
                n = len(segs)
                if n > 0:
                    words = text.split()
                    per = max(1, len(words) // n)
                    chunks = []
                    idx = 0
                    for i, (spk, _s, _e) in enumerate(segs):
                        end = len(words) if i == n - 1 else min(len(words), idx + per)
                        chunk_words = words[idx:end]
                        idx = end
                        chunk_text = " ".join(chunk_words).strip()
                        label = spk if spk.lower().startswith("speaker") else f"Speaker {i+1}"
                        if chunk_text:
                            chunks.append(f"{label}: {chunk_text}")
                    pretty = ("\n\n".join(chunks)) if chunks else text
                    st.text_area("Raw text:", value=pretty, height=240)
                else:
                    st.text_area("Raw text:", value=text, height=240)
            else:
                st.text_area("Raw text:", value=text, height=240)
            if speaker_segments:
                st.caption("Detected speakers (experimental)")
                for i, (spk, start, end) in enumerate(speaker_segments, 1):
                    st.write(f"{i}. {spk}: {start:.2f}s → {end:.2f}s")
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("💾 Download raw .txt", data=text, file_name="transcript_raw.txt", mime="text/plain")
            with col_b:
                srt_data = "1\n00:00:00,000 --> 00:59:59,000\n" + text + "\n"
                st.download_button("🎞️ Download raw .srt", data=srt_data, file_name="transcript_raw.srt", mime="application/x-subrip")

    with tabs[1]:
        # Always show the refine button. Handle missing config gracefully.
        refine_btn = st.button("✨ Refine with Ollama")
        refined_key = "last_transcript_refined"
        if refine_btn or st.session_state.get(refined_key):
            if refine_btn:
                with st.spinner("Contacting Ollama model to refine transcript…"):
                    model = st.session_state.get("ollama_model", "gpt-oss:20b")
                    prompt = (
                        "You are a transcription refiner. Clean up casing, punctuation, and spacing, "
                        "fix obvious homophones only when clearly wrong, expand common contractions if appropriate, "
                        "and keep the meaning. Do not hallucinate content. Return only the refined transcript.\n\n"
                        f"Transcript:\n{text}"
                    )
                    try:
                        if OllamaClient is None:
                            raise RuntimeError("ollama Python client not installed. pip install ollama")
                        client = OllamaClient()
                        resp = client.generate(model=model, prompt=prompt)
                        refined = (resp.get("response") or resp.get("output") or "").strip()
                        if not refined:
                            st.warning("Ollama returned empty response.")
                        st.session_state[refined_key] = refined or text
                    except Exception as e:
                        st.error(f"Ollama error: {e}")

            refined_text = st.session_state.get(refined_key)
            if refined_text:
                with st.container(border=True):
                    st.subheader("📝 Refined Transcription")
                    st.text_area("Refined text:", value=refined_text, height=240)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.download_button("💾 Download refined .txt", data=refined_text, file_name="transcript_refined.txt", mime="text/plain")
                    with col_b:
                        srt_data_ref = "1\n00:00:00,000 --> 00:59:59,000\n" + refined_text + "\n"
                        st.download_button("🎞️ Download refined .srt", data=srt_data_ref, file_name="transcript_refined.srt", mime="application/x-subrip")


def speech_to_text_page():
    st.title("🎤 Speech to Text")
    st.write("Record or upload audio/video and convert to text. Audio is normalized to 16kHz mono for accurate recognition.")

    # Global controls
    with st.expander("ASR Settings", expanded=True):
        st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], key="whisper_model")
        st.checkbox("Use faster-whisper (if installed)", value=True, key="use_faster_whisper")
        st.checkbox("Enable long-audio mode (chunked)", value=True, key="long_audio_mode")
        st.slider("Chunk size (seconds)", min_value=30, max_value=300, value=90, step=15, key="chunk_size_sec")
        st.checkbox("Force CPU fp16 off", value=True, key="whisper_fp16_off")
        # Optional NeMo prefetch
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬇️ Download/Cache Model"):
                model_name = st.session_state.get("whisper_model", "base")
                if whisper is None:
                    st.error("Whisper not installed. Run: pip install -U openai-whisper")
                else:
                    with st.spinner(f"Downloading Whisper '{model_name}' model…"):
                        try:
                            # load_model downloads/caches weights if missing
                            model = whisper.load_model(model_name)
                            st.session_state[f"_whisper_model_{model_name}"] = model
                            st.success(f"Model '{model_name}' is ready.")
                        except Exception as e:
                            st.error(f"Failed to download model: {e}")
        with col_b:
            if st.button("⬇️ Prepare NeMo Diarization"):
                with st.spinner("Installing/loading NeMo (first time can be large)…"):
                    ok = _ensure_nemo_installed(verbose=True)
                    if not ok:
                        st.error("Failed to install/load NeMo. Try manual: pip install nemo_toolkit soundfile")
                    else:
                        # Lightly instantiate to trigger lazy downloads when possible
                        try:
                            from nemo.collections.asr.models import ClusteringDiarizer as _CD  # type: ignore
                            _ = _CD(cfg={"num_workers": 1, "manifest_filepath": "", "out_dir": tempfile.mkdtemp()})
                            st.success("NeMo ready. Models will auto-download on first run.")
                        except Exception:
                            st.info("NeMo installed. Models will download on first diarization run.")

        st.markdown("---")
        # Diarization download removed per request

    tabs = st.tabs(["Record", "Upload"])

    # --------------- Record Tab ---------------
    with tabs[0]:
        st.caption("Record directly in the browser and then transcribe.")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioRecorder,
            media_stream_constraints={"video": False, "audio": True},
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Recording")
            if webrtc_ctx.state.playing:
                st.success("🔴 Recording… Click Stop when done.")
            else:
                st.info("Click Start to begin recording.")

        with col2:
            st.subheader("Transcription")
            can_transcribe = not webrtc_ctx.state.playing and webrtc_ctx.audio_processor is not None
            if st.button("🎯 Transcribe Recording", type="primary", disabled=not can_transcribe):
                frames = webrtc_ctx.audio_processor.pop_frames() if webrtc_ctx.audio_processor else []
                if not frames:
                    st.warning("No audio captured. Please record first.")
                else:
                        with st.spinner("Preparing audio…"):
                            wav_path = encode_frames_to_wav_16k_mono(frames)
                        if not wav_path:
                            st.error("Recording seems empty or too short. Please try again.")
                        else:
                            with st.spinner("Transcribing (Whisper)…"):
                                ok, text = transcribe_with_whisper(wav_path, model_name=st.session_state.get("whisper_model", "base"))
                            os.remove(wav_path)
                            if ok:
                                st.session_state.last_transcript = text
                            else:
                                st.error(text)

    # --------------- Upload Tab ---------------
    with tabs[1]:
        st.caption("Upload audio or video; we will extract audio and transcribe.")
        uploaded = st.file_uploader("Upload audio/video", type=None, accept_multiple_files=False)
        # Diarization controls
        st.checkbox("Enable speaker diarization", value=False, key="enable_diarization")
        if st.session_state.get("enable_diarization"):
            engine = st.selectbox("Diarization engine", ["NeMo (best)", "Simple stub"], key="diar_engine")
            if engine == "Simple stub":
                st.text_input("Diarization API token (optional)", value=st.session_state.get("diarization_token", ""), key="diarization_token")
            else:
                st.caption("Using NVIDIA NeMo clustering diarization. Install nemo_toolkit[all].")
        auto_go = st.checkbox("Auto-transcribe on upload", value=True)

        if uploaded is not None:
            data = uploaded.read()
            wav_path = None
            with st.spinner("Converting to 16kHz mono WAV…"):
                try:
                    wav_path = to_wav16k_mono_from_any(input_bytes=data)
                except Exception as e:
                    st.error(f"Failed to decode file: {e}")

            if wav_path:
                if auto_go or st.button("Transcribe Uploaded File", type="primary"):
                    # Auto decide chunking: if longer than 3 chunks, use chunked path
                    chunk_sec = int(st.session_state.get("chunk_size_sec", 90))
                    # Measure duration
                    try:
                        c = av.open(wav_path)
                        dur = float(c.duration or 0) / av.time_base if c.duration else 0.0
                        if dur <= 0:
                            # fallback by counting samples
                            total = 0
                            for f in c.decode(audio=0):
                                total += getattr(f, "samples", 0) or 0
                            dur = total / 16000.0
                        c.close()
                    except Exception:
                        dur = 0.0
                    use_chunked = dur >= (3 * chunk_sec)
                    
                    diarized_segments: Optional[List[Tuple[str, float, float]]] = None
                    if st.session_state.get("enable_diarization"):
                        with st.spinner("Running speaker diarization…"):
                            if st.session_state.get("diar_engine") == "NeMo (best)":
                                diarized_segments = diarize_with_nemo(wav_path)
                            else:
                                diarized_segments = simple_speaker_diarize(wav_path)
                    with st.spinner("Transcribing (Whisper)…"):
                        if use_chunked:
                            # For now, fall back to normal transcribe; chunked engine forthcoming
                            ok, text = transcribe_with_whisper(wav_path, model_name=st.session_state.get("whisper_model", "base"))
                        else:
                            ok, text = transcribe_with_whisper(wav_path, model_name=st.session_state.get("whisper_model", "base"))
                    os.remove(wav_path)
                    if ok:
                        st.session_state.last_transcript = text
                        st.session_state.last_speaker_segments = diarized_segments
                    else:
                        st.error(text)

    # --------------- Output ---------------
    if st.session_state.get("last_transcript"):
        st.markdown("---")
        render_transcript(
            st.session_state.last_transcript,
            st.session_state.get("last_speaker_segments"),
        )

        if st.button("🗑️ Clear Transcript"):
            st.session_state.pop("last_transcript", None)
            st.session_state.pop("last_speaker_segments", None)
            st.rerun()
