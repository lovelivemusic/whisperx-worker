"""
PANNs CNN14 Audio Classification — GPU-accelerated SED for the RunPod worker.

Provides the same classification as the local PANNs service but runs on GPU
for ~50-100x faster inference. Used when task='classify' in the RunPod handler.
"""

import os
import time
import logging

import numpy as np
import torch
import librosa
from panns_inference import AudioTagging

logger = logging.getLogger("panns_classify")

# Global model (loaded once, reused across jobs)
_MODEL = None
_LABELS = {}

SPEECH_LABELS = {'Speech', 'Male speech, man speaking', 'Female speech, woman speaking',
                 'Child speech, kid speaking', 'Conversation', 'Narration, monologue'}
MUSIC_LABELS = {'Music', 'Musical instrument', 'Singing', 'Song', 'Plucked string instrument',
                'Keyboard (musical)', 'Drum', 'Guitar', 'Piano'}
NOISE_LABELS = {'Noise', 'White noise', 'Pink noise', 'Static', 'Hum', 'Buzz'}


def _get_model():
    """Load and cache PANNs CNN14 model on GPU."""
    global _MODEL, _LABELS
    if _MODEL is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading PANNs CNN14 model on {device}...")
        start = time.time()
        _MODEL = AudioTagging(checkpoint_path=None, device=device)

        # Load labels
        labels_path = os.path.expanduser("~/panns_data/class_labels_indices.csv")
        if os.path.exists(labels_path):
            import csv
            with open(labels_path) as f:
                reader = csv.reader(f)
                next(reader)
                _LABELS = {int(row[0]): row[2] for row in reader}
        logger.info(f"PANNs model loaded in {time.time() - start:.2f}s")
    return _MODEL, _LABELS


def _category_score(scores, label_set, labels):
    """Get max score across labels in a category."""
    return float(max((scores[idx] for idx, l in labels.items() if l in label_set), default=0))


def classify_sed(audio_path, frame_size=0.5, merge_threshold=0.5, refine_boundaries=True):
    """
    Run SED classification on an audio file using PANNs CNN14 on GPU.

    Args:
        audio_path: Path to audio file
        frame_size: Frame size in seconds (default 0.5)
        merge_threshold: Minimum segment duration to keep (default 0.5s)
        refine_boundaries: Snap edges to speech onset/offset via RMS (default True)

    Returns:
        dict with segments, frame_count, total_duration_s, inference_time_s
    """
    model, labels = _get_model()
    sr = 32000

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio = audio.astype(np.float32)
    total_duration = len(audio) / sr

    # Frame-level classification
    frame_samples = int(frame_size * sr)
    frames = []

    start = time.time()
    for i in range(0, len(audio) - frame_samples, frame_samples):
        frame_audio = audio[i:i + frame_samples]
        frame_start = i / sr
        frame_end = (i + frame_samples) / sr

        clipwise_output, _ = model.inference(frame_audio[None, :])
        scores = clipwise_output[0]

        speech_score = _category_score(scores, SPEECH_LABELS, labels)
        music_score = _category_score(scores, MUSIC_LABELS, labels)
        noise_score = _category_score(scores, NOISE_LABELS, labels)

        max_score = max(speech_score, music_score, noise_score)
        if speech_score == max_score:
            category = 'speech'
        elif music_score == max_score:
            category = 'music'
        else:
            category = 'noise'

        frames.append({
            'start': round(frame_start, 3),
            'end': round(frame_end, 3),
            'category': category,
        })

    inference_time = time.time() - start

    # Merge adjacent frames with same category
    merged = []
    for frame in frames:
        if merged and merged[-1]['category'] == frame['category']:
            merged[-1]['end'] = frame['end']
        else:
            merged.append({
                'start': frame['start'],
                'end': frame['end'],
                'category': frame['category']
            })

    # Filter short segments
    filtered = [seg for seg in merged if (seg['end'] - seg['start']) >= merge_threshold]

    # Fill gaps
    if filtered:
        final_segments = [filtered[0]]
        for seg in filtered[1:]:
            if final_segments[-1]['end'] < seg['start']:
                gap_mid = (final_segments[-1]['end'] + seg['start']) / 2
                final_segments[-1]['end'] = gap_mid
                seg['start'] = gap_mid
            final_segments.append(seg)
        filtered = final_segments

    # Energy-based boundary refinement
    if refine_boundaries and filtered:
        rms_hop = 512
        rms = librosa.feature.rms(y=audio, hop_length=rms_hop)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=rms_hop)
        rms_threshold = float(np.median(rms) * 0.1)

        for seg in filtered:
            if seg['category'] != 'speech':
                continue
            # Snap start
            search_start = max(0, seg['start'] - 0.2)
            search_end = min(seg['start'] + 0.2, seg['end'])
            mask = (rms_times >= search_start) & (rms_times <= search_end)
            active = np.where(mask & (rms > rms_threshold))[0]
            if len(active) > 0:
                seg['start'] = round(float(rms_times[active[0]]), 3)
            # Snap end
            search_start = max(seg['start'], seg['end'] - 0.2)
            search_end = min(total_duration, seg['end'] + 0.2)
            mask = (rms_times >= search_start) & (rms_times <= search_end)
            active = np.where(mask & (rms > rms_threshold))[0]
            if len(active) > 0:
                seg['end'] = round(float(rms_times[active[-1]]), 3)

    logger.info(f"SED complete: {len(frames)} frames -> {len(filtered)} segments in {inference_time:.2f}s")

    # Include GPU name for cost tracking
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        'segments': filtered,
        'frame_count': len(frames),
        'total_duration_s': round(total_duration, 2),
        'frame_size': frame_size,
        'inference_time_s': round(inference_time, 2),
        'gpu_name': gpu_name,
    }
