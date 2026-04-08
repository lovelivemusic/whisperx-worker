import base64
import os
import sys
import shutil
import logging
import tempfile

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, whoami
import torch
import numpy as np
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
from speaker_profiles import load_embeddings, relabel
from panns_classify import classify_sed
from speaker_processing import (
    process_diarized_output, enroll_profiles, identify_speakers_on_segments,
    load_known_speakers_from_samples, identify_speaker, relabel_speakers_by_avg_similarity,
    spk_embed,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if hf_token and not hf_token.startswith("hf_"):
    logger.warning("HF_TOKEN does not start with 'hf_' prefix - token may be malformed")

if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")




MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
def run(job):
    job_id     = job["id"]
    job_input  = job["input"]

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # ------------- route by task -----------------------------------
    task = job_input.get("task", "transcribe")

    # ------------- 1) resolve audio input (URL or base64) -----------
    audio_input = job_input["audio_file"]
    try:
        if "://" in audio_input:
            # Standard URL — download as before
            audio_file_path = download_files_from_urls(job_id, [audio_input])[0]
            logger.debug(f"Audio downloaded → {audio_file_path}")
        else:
            # Treat as base64-encoded audio data
            # Strip optional data-URI prefix (e.g. "data:audio/wav;base64,")
            if "," in audio_input:
                audio_input = audio_input.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_input)
            os.makedirs(f"/jobs/{job_id}", exist_ok=True)
            audio_file_path = f"/jobs/{job_id}/audio_input"
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)
            logger.debug(f"Audio decoded from base64 → {audio_file_path} ({len(audio_bytes)} bytes)")
    except Exception as e:
        logger.error("Audio input failed", exc_info=True)
        return {"error": f"audio input: {e}"}

    # ------------- PANNs classification task -------------------------
    if task == "classify":
        try:
            result = classify_sed(
                audio_file_path,
                frame_size=job_input.get("frame_size", 0.5),
                merge_threshold=job_input.get("merge_threshold", 0.5),
                refine_boundaries=job_input.get("refine_boundaries", True),
            )
            cleanup_job_files(job_id)
            return result
        except Exception as e:
            logger.error("PANNs classification failed", exc_info=True)
            cleanup_job_files(job_id)
            return {"error": f"classification: {e}"}

    # ------------- 2) download speaker profiles (optional) ----------
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=hf_token  # or job_input.get("huggingface_access_token")
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            embeddings = {}  # graceful degradation: proceed without profiles

    # ------------- 3) call WhisperX / VAD / diarization -------------
    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 64),
        "temperature"              : job_input.get("temperature", 0),
        "vad_onset"                : job_input.get("vad_onset", 0.50),
        "vad_offset"               : job_input.get("vad_offset", 0.363),
        "align_output"             : job_input.get("align_output", False),
        "diarization"              : job_input.get("diarization", False),
        "huggingface_access_token" : job_input.get("huggingface_access_token") or hf_token,
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)             # <-- heavy job
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    # GPU and worker info
    gpu_name = None
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    output_dict = {
        "segments"         : result.segments,
        "detected_language": result.detected_language,
        "gpu_name"         : gpu_name,
        "worker_version"   : "11",
    }

    # Include overlap regions if requested and available
    if job_input.get("include_overlaps", False) and result.overlaps:
        output_dict["overlaps"] = result.overlaps
        logger.info(f"Overlap regions: {len(result.overlaps)}")

    # Extract per-speaker embeddings if requested (192-D ECAPA via SpeechBrain)
    if job_input.get("include_embeddings", False) and job_input.get("diarization", False):
        try:
            import librosa
            unique_speakers = set()
            for seg in output_dict["segments"]:
                spk = seg.get("speaker")
                if spk:
                    unique_speakers.add(spk)

            speaker_embeddings = {}
            for spk in unique_speakers:
                # Find longest segment for this speaker
                spk_segs = [s for s in output_dict["segments"] if s.get("speaker") == spk]
                longest = max(spk_segs, key=lambda s: s["end"] - s["start"])
                # Cap at 15 seconds
                start = longest["start"]
                duration = min(longest["end"] - start, 15.0)
                wav, sr = librosa.load(audio_file_path, sr=16000, mono=True,
                                       offset=start, duration=duration)
                if len(wav) > 0:
                    emb = spk_embed(wav)
                    speaker_embeddings[spk] = emb.flatten().tolist()

            if speaker_embeddings:
                output_dict["speaker_embeddings"] = speaker_embeddings
                logger.info(f"Speaker embeddings: {len(speaker_embeddings)} speakers, 192-D each")
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}", exc_info=True)
            output_dict["embedding_error"] = str(e)

    # 4) speaker verification (optional)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1  # Adjust threshold as needed
            )
            #output_dict["segments"] = segments_with_speakers
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed successfully.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings available; skipping speaker identification.")

    # 4-Cleanup and return output_dict normally
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict

runpod.serverless.start({"handler": run})