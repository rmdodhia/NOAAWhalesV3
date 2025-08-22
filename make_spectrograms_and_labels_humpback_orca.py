# -----------------------------------------------------------------------------
# Script: make_spectrograms_and_labels_humpback_orca.py
# -----------------------------------------------------------------------------
# This script processes audio files and their annotations to generate spectrogram
# segments and corresponding labels for machine learning tasks.
#
# How it works:
# 1. For each audio file referenced in the annotation CSV, the script loads the audio
#    and computes its MEL spectrogram.
# 2. The spectrogram is split into overlapping segments of fixed duration (SDUR)
#    and overlap (OVERLAP).
# 3. For each segment, the script checks if it overlaps with any annotation by at least
#    MIN_OVERLAP_FOR_POSITIVE seconds. If so, the segment is labeled 1 (positive), else 0 (negative).
# 4. Each segment is saved as a .pt file (PyTorch tensor) containing the dB-scaled spectrogram
#    slice for that segment.
# 5. A CSV file is generated for each location, listing the .pt filename, label, audiofile,
#    annotationfile(s), location, full path, and species.
# 6. All location CSVs are combined into a single CSV for the species.
#
# Key configuration parameters:
#   - SDUR: Segment duration in seconds
#   - OVERLAP: Overlap between segments in seconds
#   - MIN_OVERLAP_FOR_POSITIVE: Minimum annotation overlap required for label=1
#   - N_MELS: MEL spectrogram parameter
#
# Output:
#   - .pt files: Spectrogram segments for each audio file
#   - CSV files: Labels and metadata for each segment
# -----------------------------------------------------------------------------
# make_spectrograms_and_labels.py

import os
import glob
import math
import time
import torch
import torchaudio
import logging
import datetime
import pytz
import numpy as np
import pandas as pd
from torchaudio.transforms import MelSpectrogram

# ─── CONFIG ────────────────────────────────────────────────────────────────
SDUR      = 2.0        # segment duration (s)
OVERLAP   = 0.4        # succeeding segment overlap (s)
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
# MEL spectrogram parameters
N_MELS    = 128        # number of mel bins for mel spectrogram
N_FFT     = 2048       # FFT window size
HOP_LEN   = 512        # hop length (samples)
# Minimum overlap (seconds) with annotation required for label=1
MIN_OVERLAP_FOR_POSITIVE = 0.3
SPEC_TYPE = "stft"     # Options: "mel" or "stft"

# ─── LOGGING ───────────────────────────────────────────────────────────────
os.makedirs('Logs', exist_ok=True)
today = datetime.datetime.now(pytz.UTC).strftime('%Y-%m-%d')
run_no= len(glob.glob(f"Logs/MakeSpectrograms/make_s_and_l_humpback_orca_{today}_*.log")) + 1
local_time = datetime.datetime.now().strftime('%H%M%S')
logging.basicConfig(
    filename=f"Logs/MakeSpectrograms/make_s_and_l_humpback_orca_{today}_{local_time}.log",
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    force=True
)

# ─── UTILITIES ────────────────────────────────────────────────────────────
def resolve_audio_folder(base):
    for d,_,fs in os.walk(base):
        if any(f.endswith('.wav') for f in fs):
            return d
    raise FileNotFoundError(f"No .wav under {base}")

# ─── LOCATION NAME MAPPING ────────────────────────────────────────────────
# Not needed anymore; directories are already canonical under DataInput_New

# ─── CORE WORK ─────────────────────────────────────────────────────────────

def process_wav_file(
    wav_path,
    dst_folder,
    ann_df,
    segment_duration=2.0,
    overlap=0.4,
    n_pad=5.0,
    k_null=3,
    min_overlap_for_positive=MIN_OVERLAP_FOR_POSITIVE,
    large_file_samples=15_000_000,  # ~600s at 24kHz
    chunk_duration_s=600           # 10 minutes per chunk
):
    import math
    wav_info = torchaudio.info(wav_path)
    num_samples = wav_info.num_frames
    sr = wav_info.sample_rate
    file_duration = num_samples / sr
    audiofile = os.path.basename(wav_path)
    location = os.path.basename(os.path.dirname(wav_path))
    results = []

    def process_chunk(wave, chunk_start_s, ann_df, sr, chunk_index, n_chunks):
        try:
            chunk_samples = wave.shape[-1]
            if DEVICE == 'cuda' and chunk_samples <= large_file_samples:
                try:
                    wave = wave.to(DEVICE).half()
                    device = DEVICE
                    dtype = torch.float16
                    logging.info(f"Processing chunk {chunk_index + 1} of {n_chunks} for location {location} on GPU (float16): {chunk_samples} samples starting at {chunk_start_s:.2f}s")
                except RuntimeError as oom:
                    logging.warning(f"CUDA OOM for chunk {chunk_index + 1} ({chunk_start_s:.2f}s), retrying on CPU: {oom}")
                    wave = wave.cpu().float()
                    device = 'cpu'
                    dtype = torch.float32
            else:
                wave = wave.cpu()
                device = 'cpu'
                dtype = torch.float32
                logging.info(f"Processing chunk {chunk_index} of {n_chunks} for location {location} on CPU: {chunk_samples} samples starting at {chunk_start_s:.2f}s")

            # --- Get all annotations and build padded intervals ---
            sub = ann_df[ann_df["audiofile_path"] == audiofile]
            annotation_records = []  # (start, end, annotationfile)
            padded_intervals = []
            for _, row in sub.iterrows():
                start = row["startseconds"]
                end = start + row["durationseconds"]
                ann_file = row.get("annotationfile", row.get("source_annotation_file", ""))
                annotation_records.append((start, end, ann_file))
                # Use exact interval without padding
                padded_intervals.append((start, end))
            # extract pure time intervals for region checks
            annotation_intervals = [(a, b) for (a, b, _) in annotation_records]

            # --- Merge overlapping padded intervals into non-overlapping regions ---
            def merge_intervals(intervals):
                if not intervals:
                    return []
                sorted_intervals = sorted(intervals, key=lambda x: x[0])
                merged = [sorted_intervals[0]]
                for current in sorted_intervals[1:]:
                    last = merged[-1]
                    if current[0] <= last[1]:
                        merged[-1] = (last[0], max(last[1], current[1]))
                    else:
                        merged.append(current)
                return merged

            merged_padded = merge_intervals(padded_intervals)

            def in_any_region(t0, t1, regions):
                return any(max(0, min(b, t1) - max(a, t0)) > 0 for a, b in regions)

            def overlaps_any_anno(t0, t1, intervals, min_overlap):
                return any(max(0, min(b, t1) - max(a, t0)) >= min_overlap for a, b in intervals)

            try:
                    # --- Compute spectrogram once ---
                    if SPEC_TYPE == "mel":
                        from torchaudio.transforms import MelSpectrogram
                        transform = MelSpectrogram(
                            sample_rate=sr,
                            n_fft=N_FFT,
                            hop_length=HOP_LEN,
                            n_mels=N_MELS,
                            win_length=N_FFT,
                            power=2.0,
                            normalized=False
                        ).to(device=device, dtype=dtype)
                        spec = transform(wave)
                    elif SPEC_TYPE == "stft":
                        from torchaudio.transforms import Spectrogram
                        transform = Spectrogram(
                            n_fft=N_FFT,
                            hop_length=HOP_LEN,
                            win_length=N_FFT,
                            power=2.0,
                            normalized=False
                        ).to(device=device, dtype=dtype)
                        spec = transform(wave)
                    else:
                        raise ValueError(f"Unknown SPEC_TYPE: {SPEC_TYPE}")
                    db = 10 * torch.log10(spec + 1e-6)
                    db = torch.clamp(db, min=-80, max=0).cpu()
            except Exception as e:
                logging.error(f"MEL failed for {audiofile} chunk {chunk_index} ({chunk_start_s:.2f}s): {e}")
                return
            seg_frames = math.ceil((segment_duration * sr) / HOP_LEN)
            step_frames = math.ceil(((segment_duration - overlap) * sr) / HOP_LEN)

            # --- Check if audio is too short for segment extraction ---
            if db.shape[1] < seg_frames:
                logging.warning(f"Skipping {audiofile} chunk {chunk_index} of {n_chunks}: too short for segment extraction (frames: {db.shape[1]}, needed: {seg_frames})")
                return
            try:
                slices = db.unfold(dimension=1, size=seg_frames, step=step_frames)
                slices = slices.permute(1, 0, 2)
            except Exception as e:
                logging.error(f"Spectrogram slicing failed for {audiofile} chunk {chunk_index} ({chunk_start_s:.2f}s): {e}")
                return
            num_segments = slices.shape[0]
            stride = segment_duration - overlap

            # --- Store indices for null segment selection ---
            all_indices = []
            for i in range(num_segments):
                seg_start = i * stride
                seg_end = seg_start + segment_duration
                abs_seg_start = seg_start + chunk_start_s
                abs_seg_end = seg_end + chunk_start_s
                all_indices.append((i, abs_seg_start, abs_seg_end))

            # --- For each segment across the whole chunk, label as 1 if overlaps any annotation, else 0, and record details ---
            for i, abs_seg_start, abs_seg_end in all_indices:
                arr = slices[i].numpy()
                is_pos = overlaps_any_anno(abs_seg_start, abs_seg_end, annotation_intervals, min_overlap_for_positive)
                label = int(is_pos)
                overlapping_ann = []
                annotation_starts = []
                annotation_ends = []
                total_overlap = 0.0
                for a, b, af in annotation_records:
                    ov = max(0, min(b, abs_seg_end) - max(a, abs_seg_start))
                    if ov >= min_overlap_for_positive:
                        overlapping_ann.append(af)
                        annotation_starts.append(a)
                        annotation_ends.append(b)
                        total_overlap += ov
                ann_str = ";".join(overlapping_ann) if overlapping_ann else ""
                annotation_start = min(annotation_starts) if annotation_starts else ''
                annotation_end = max(annotation_ends) if annotation_ends else ''
                overlap_ms = int(round(total_overlap * 1000))
                seg_len = (abs_seg_end - abs_seg_start) if (abs_seg_end - abs_seg_start) > 0 else 1.0
                overlap_pct = 100.0 * (total_overlap / seg_len)
                fname = f"{audiofile.replace('.wav','')}_{int(abs_seg_start*1000)}_{int(abs_seg_end*1000)}.pt"
                fpath = os.path.join(dst_folder, fname)
                try:
                    torch.save(torch.from_numpy(arr), fpath)
                except Exception as e:
                    logging.error(f"Failed to save tensor for {audiofile} segment {abs_seg_start:.2f}-{abs_seg_end:.2f}s: {e}")
                    continue
                results.append((fname, label, audiofile, ann_str, overlap_ms, overlap_pct, annotation_start, annotation_end))
        except Exception as e:
            logging.error(f"process_chunk failed for {audiofile} chunk {chunk_index} ({chunk_start_s:.2f}s): {e}")
            return

    os.makedirs(dst_folder, exist_ok=True)
    if num_samples > large_file_samples:
        logging.info(f"Chunked processing for {audiofile} ({num_samples} samples)")
        chunk_samples = int(chunk_duration_s * sr)
        n_chunks = math.ceil(num_samples / chunk_samples)
        for chunk_idx in range(n_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, num_samples)
            chunk_len = end_sample - start_sample
            if chunk_len <= 0:
                continue
            wave, _ = torchaudio.load(wav_path, frame_offset=start_sample, num_frames=chunk_len)
            process_chunk(wave[0], chunk_start_s=start_sample / sr, ann_df=ann_df, sr=sr, chunk_index=chunk_idx, n_chunks=n_chunks)
    else:
        n_chunks = 1
        wave, _ = torchaudio.load(wav_path)
        process_chunk(wave[0], chunk_start_s=0, ann_df=ann_df, sr=sr, chunk_index=0, n_chunks=n_chunks)
    # Robust to tuple length
    n_pos = sum(1 for _, label, *rest in results if label == 1)
    n_neg = sum(1 for _, label, *rest in results if label == 0)
    logging.info(f"Processed file {audiofile}: {n_pos} positive, {n_neg} negative images generated")
    return results


def generate_spectrograms_and_labels(
    species, locations, ann_csv=None, proportion=1,
    segment_duration=2.0, overlap=0.4, n_pad=5.0, k_null=3, min_overlap_for_positive=MIN_OVERLAP_FOR_POSITIVE
):
    ann_csv = ann_csv or f"./DataInput_New/{species}/Processed/{species}_annotations_processed.csv"
    logging.info(f"Loading annotations from {ann_csv}")
    ann_df = pd.read_csv(ann_csv, low_memory=False)
    logging.info(f"Loaded {len(ann_df)} annotation rows")
    if proportion < 1.0:
        ann_df = ann_df.sample(frac=proportion, random_state=88)
        logging.info(f"Sampled {len(ann_df)} annotation rows (proportion={proportion})")

    # Standardize column names for startseconds and durationSeconds
    if 'startSeconds' in ann_df.columns:
        ann_df = ann_df.rename(columns={'startSeconds': 'startseconds'})
    if 'durationSeconds' in ann_df.columns:
        ann_df = ann_df.rename(columns={'durationSeconds': 'durationseconds'})  # keep as is for code compatibility
    else:
        # Compute durationseconds if not present
        if 'Begin Time (s)' in ann_df.columns and 'End Time (s)' in ann_df.columns:
            ann_df['durationseconds'] = ann_df['End Time (s)'] - ann_df['Begin Time (s)']
            logging.info("Computed durationseconds from End Time (s) - Begin Time (s)")
        else:
            logging.error("Cannot compute durationseconds: missing 'Begin Time (s)' or 'End Time (s)' columns.")
            raise ValueError("Missing columns to compute durationseconds.")

    # Always use basename for matching
    if 'audiofile_path' in ann_df.columns:
        ann_df['audiofile_path'] = ann_df['audiofile_path'].apply(os.path.basename)
        logging.info(f"First 5 audiofile entries after basename extraction: {ann_df['audiofile_path'].head().tolist()}")
    elif 'audiofile' in ann_df.columns:
        ann_df.rename(columns={'audiofile': 'audiofile_path'}, inplace=True)
        ann_df['audiofile_path'] = ann_df['audiofile_path'].apply(os.path.basename)
        logging.info(f"First 5 audiofile entries after basename extraction: {ann_df['audiofile_path'].head().tolist()}")
    else:
        logging.warning("'audiofile_path' column not found in annotation DataFrame!")

    # Standardize location column in annotation DataFrame
    if 'location' not in ann_df.columns:
        logging.warning("'location' column not found in annotation DataFrame!")

    for loc in locations:
        # WAVs are under DataInput_New/{species}/Audio/{loc}
        src = resolve_audio_folder(f"./DataInput_New/{species}/Audio/{loc}")
        dst = f"./DataInput_New/{species}/Processed/SpectrogramsOverlap{int(overlap*1000)}ms/{loc}"

        # Only process WAVs referenced in ann_df for this location
        wavs_in_ann = set(ann_df[ann_df['location'] == loc]['audiofile_path'].unique())
        wavs_to_process = [os.path.join(src, w) for w in wavs_in_ann if os.path.exists(os.path.join(src, w))]

        logging.info(f"Processing location {loc}: src={src}, dst={dst}, {len(wavs_to_process)} wavs")
        all_out = []
        for wav in sorted(wavs_to_process):
            out = process_wav_file(
                wav, dst, ann_df,
                segment_duration=segment_duration,
                overlap=overlap,
                n_pad=n_pad,
                k_null=k_null,
                min_overlap_for_positive=min_overlap_for_positive
            )
            all_out.extend(out)

        # Write label CSV for this location
        df = pd.DataFrame(all_out, columns=['filename', 'label', 'audiofile', 'annotationfile', 'overlap_ms', 'overlap_pct', 'annotation_start', 'annotation_end'])
        df['location'] = loc
        df['fullpath'] = df['filename'].apply(lambda fn: os.path.join(dst, fn))
        df['species'] = species.lower()
        csv_out = f"./DataInput_New/{species}/Processed/LabelsOverlap{int(overlap*1000)}ms/{species}_{loc}_overlap{int(overlap*1000)}ms_spectrogram_labels.csv"
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        df.to_csv(csv_out, index=False)
        logging.info(f"Saved labels to {csv_out}")

    # After all locations processed, combine all label CSVs for this species into one file
    pattern = f"./DataInput_New/{species}/Processed/LabelsOverlap{int(overlap*1000)}ms/{species}_*_overlap{int(overlap*1000)}ms_spectrogram_labels.csv"
    all_label_files = glob.glob(pattern)
    if all_label_files:
        all_df = pd.concat([pd.read_csv(f) for f in all_label_files], ignore_index=True)
        combined_csv = f"./DataInput_New/{species}/Processed/LabelsOverlap{int(overlap*1000)}ms/{species}_labels.csv"
        all_df.to_csv(combined_csv, index=False)
        logging.info(f"Combined all label files into {combined_csv}\n\n")

# ─── RUNNER ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_spectrograms_and_labels(
        species   = "Humpback",
        locations = [
            "Chinitna",    # canonical, not LCI_Chinitna
            "Iniskin",     # canonical, not LCI_Iniskin
            "PtGraham",     # canonical, not LCI_Port_Graham
            "AL16_BS4",
            "AL16_NM1"
        ],
        ann_csv   = None,
        proportion = 1.0
    )
    generate_spectrograms_and_labels(
        species   = "Orca",
        locations = [
            "Chinitna",
            "Iniskin",
            "PtGraham",
            "SWCorner"
        ],
        ann_csv   = None,
        proportion = 1.0
    )

    # # adjust species & overlap_ms as needed
    # species    = "Humpback"
    # overlap_ms = 400
    # base       = f"./DataInput/{species}"
    # pattern    = f"{base}/*_{species.lower()}_overlap{overlap_ms}ms_spectrogram_labels.csv"

    # all_df = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)], ignore_index=True)
    # all_df.to_csv(f"{base}/{species.lower()}_spectrogram_labels_overlap{overlap_ms}ms.csv", index=False)

    # Check if any 'audiofile' entry begins with '604536840' and show row indices
