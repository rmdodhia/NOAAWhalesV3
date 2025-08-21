#!/usr/bin/env python3
"""
Unified spectrogram analysis tool that supports both row-based and time-based analysis.

Usage modes:
1. Row-based analysis (compare pre-generated vs newly generated):
   python unified_spectrogram_analyzer.py --species Beluga --row 42
   
2. Time-based analysis (generate from specific time and check overlaps):
   python unified_spectrogram_analyzer.py --audiofile Iniskin_HB8_191020205232 --start 26148800

Features:
- Generate spectrograms from audio files
- Compare with pre-generated .pt spectrograms
- Check for annotation overlaps
- Check for existing spectrogram overlaps
- Comprehensive visualization with timeline overlays
"""

import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio


def find_audio_file_by_name(audiofile_name):
    """Find the full path to the audio file based on partial name."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to NOAAWhalesV3
    
    # Add .wav extension if not present
    if not audiofile_name.endswith('.wav'):
        audiofile_with_ext = audiofile_name + '.wav'
    else:
        audiofile_with_ext = audiofile_name
    
    # Search in DataInput_New structure for all species
    species_dirs = ['Humpback', 'Beluga', 'Orca']
    
    for species in species_dirs:
        species_audio_dir = os.path.join(base_dir, f"DataInput_New/{species}/Audio")
        if os.path.exists(species_audio_dir):
            # Recursively search for the audio file
            for root, dirs, files in os.walk(species_audio_dir):
                if audiofile_with_ext in files:
                    return os.path.join(root, audiofile_with_ext), species
    
    return None, None


def find_audio_file_by_species_location(species, location, audiofile):
    """Find the audio file path based on species, location, and filename (legacy function)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to NOAAWhalesV2
    
    # Add .wav extension if not present
    if not audiofile.endswith('.wav'):
        audiofile_with_ext = audiofile + '.wav'
    else:
        audiofile_with_ext = audiofile
    
    # Try different possible paths in both DataInput and DataInput_New
    possible_paths = [
        # DataInput_New paths (primary)
        os.path.join(base_dir, f"DataInput_New/{species}/Audio/{location}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput_New/{species.lower()}/Audio/{location}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput_New/{species.capitalize()}/Audio/{location}/{audiofile_with_ext}"),
        # DataInput paths (legacy)
        os.path.join(base_dir, f"DataInput/{species}/{location}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput/{species.lower()}/{location}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput/{species.capitalize()}/{location}/{audiofile_with_ext}"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, also try to find the file recursively
    for data_dir in [f"DataInput_New/{species}", f"DataInput/{species}"]:
        species_dir = os.path.join(base_dir, data_dir)
        if os.path.exists(species_dir):
            for root, dirs, files in os.walk(species_dir):
                if audiofile_with_ext in files:
                    return os.path.join(root, audiofile_with_ext)
    
    return None


def load_pregenerated_spectrogram(pt_path, hop_length=256, sample_rate=None):
    """Load and prepare a pre-generated spectrogram from .pt file."""
    if not os.path.exists(pt_path):
        print(f"Warning: Spectrogram file not found: {pt_path}")
        return None, None, None
    
    # Load spectrogram
    spec = torch.load(pt_path)
    if hasattr(spec, 'cpu'):
        spec = spec.cpu().numpy()
    
    # Calculate time and frequency axes
    n_time_frames = spec.shape[1] if spec.ndim == 2 else spec.shape[-1]
    n_freq_bins = spec.shape[0] if spec.ndim == 2 else spec.shape[-2]
    
    if sample_rate is None:
        # Try to infer from common values
        sample_rate = 24000  # Common for marine audio
    
    # Time axis in seconds
    time_axis = np.arange(n_time_frames) * hop_length / sample_rate
    
    # Frequency axis in Hz (approximate for display)
    freq_axis = np.linspace(0, sample_rate/2, n_freq_bins)
    
    return spec, time_axis, freq_axis


def generate_spectrogram_from_audio(audio_path, start_ms, end_ms, n_fft=1024, hop_length=256):
    """Generate a spectrogram from raw audio file for the specified time segment."""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return None, None, None, None
    
    try:
        # Load audio info first
        info = torchaudio.info(audio_path)
        sample_rate = info.sample_rate
        
        # Convert milliseconds to samples
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)
        num_samples = end_sample - start_sample
        
        if num_samples <= 0:
            print(f"Invalid time range: {start_ms}ms to {end_ms}ms")
            return None, None, None, None
        
        # Load the specific audio segment
        wave_tensor, sr = torchaudio.load(
            audio_path,
            frame_offset=start_sample,
            num_frames=num_samples
        )
        
        # Handle multi-channel audio
        if wave_tensor.shape[0] > 1:
            wave = wave_tensor.mean(dim=0)
        else:
            wave = wave_tensor[0]
        
        # Generate spectrogram using same parameters as original scripts
        spec = torch.stft(
            wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            return_complex=True
        )
        
        # Convert to magnitude and dB
        mag = spec.abs()
        db = 20 * torch.log10(mag + 1e-6)
        db = torch.clamp(db, min=-80, max=0)
        
        # Convert to numpy
        spec_np = db.numpy()
        
        # Calculate time and frequency axes
        n_time_frames = spec_np.shape[1]
        n_freq_bins = spec_np.shape[0]
        
        # Time axis in seconds (relative to segment start)
        time_axis = np.arange(n_time_frames) * hop_length / sample_rate
        
        # Frequency axis in Hz
        freq_axis = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_freq_bins]
        
        return spec_np, time_axis, freq_axis, sample_rate
        
    except Exception as e:
        print(f"Error generating spectrogram from audio: {e}")
        return None, None, None, None


def extract_time_from_filename(filename):
    """Extract start and end milliseconds from filename like 'audio_1234_5678.pt'."""
    try:
        # Remove extension and split by underscore
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        
        # Get the last two parts as start and end times
        if len(parts) >= 2:
            start_ms = int(parts[-2])
            end_ms = int(parts[-1])
            return start_ms, end_ms
        else:
            raise ValueError("Filename doesn't contain timing information")
            
    except (ValueError, IndexError) as e:
        print(f"Error extracting time from filename '{filename}': {e}")
        return None, None


def check_annotation_overlaps(species, audiofile_name, start_ms, end_ms, sample_rate=None):
    """
    Check if the spectrogram time range overlaps with any annotations.
    
    Args:
        species: Species name (Humpback, Beluga, Orca)
        audiofile_name: Audio filename (without extension)
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        sample_rate: Sample rate (used for Beluga calculations)
    
    Returns:
        List of overlapping annotations with details
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try both DataInput_New and legacy DataInput paths
    possible_annotation_files = [
        os.path.join(base_dir, f"DataInput_New/{species}/Processed/{species}_annotations_processed.csv"),
        os.path.join(base_dir, f"DataInput/{species}/{species}_annotations.csv")
    ]
    
    annotations_csv = None
    for path in possible_annotation_files:
        if os.path.exists(path):
            annotations_csv = path
            break
    
    if annotations_csv is None:
        print(f"Warning: No annotations file found for {species}")
        return []
    
    try:
        # Load annotations
        df_ann = pd.read_csv(annotations_csv)
        
        # Convert spectrogram times to seconds
        spec_start_sec = start_ms / 1000.0
        spec_end_sec = end_ms / 1000.0
        
        # Add .wav extension if not present for matching
        audiofile_with_ext = audiofile_name + '.wav' if not audiofile_name.endswith('.wav') else audiofile_name
        
        # Filter annotations for the same audiofile
        if 'audiofile_path' in df_ann.columns:
            # New format (processed annotations)
            mask = df_ann['audiofile_path'].str.contains(audiofile_with_ext, na=False)
        elif 'audiofile' in df_ann.columns:
            # Legacy format
            mask = df_ann['audiofile'] == audiofile_with_ext
        elif 'Begin File' in df_ann.columns:
            # Humpback format
            mask = df_ann['Begin File'] == audiofile_with_ext
        else:
            print(f"Warning: Could not find audiofile column in annotations")
            return []
        
        filtered_annotations = df_ann[mask]
        
        if len(filtered_annotations) == 0:
            print(f"No annotations found for {audiofile_with_ext}")
            return []
        
        print(f"Found {len(filtered_annotations)} total annotations for {audiofile_with_ext}")
        
        overlapping_annotations = []
        
        for _, ann_row in filtered_annotations.iterrows():
            # Determine annotation time format based on available columns
            if 'startSeconds' in ann_row and 'durationSeconds' in ann_row:
                # New processed format
                ann_start = ann_row['startSeconds']
                ann_end = ann_start + ann_row['durationSeconds']
            elif 'Begin Time (s)' in ann_row and 'End Time (s)' in ann_row:
                # Humpback/Orca format
                ann_start = ann_row['Begin Time (s)']
                ann_end = ann_row['End Time (s)']
            elif 'startSeconds' in ann_row and 'duration' in ann_row and sample_rate:
                # Beluga format
                ann_start = ann_row['startSeconds']
                ann_duration_sec = ann_row['duration'] / (ann_row.get('sampleRate', sample_rate))
                ann_end = ann_start + ann_duration_sec
            else:
                continue  # Skip if format not recognized
            
            # Check for overlap: two intervals overlap if max(start1,start2) < min(end1,end2)
            overlap_start = max(spec_start_sec, ann_start)
            overlap_end = min(spec_end_sec, ann_end)
            
            if overlap_start < overlap_end:
                # There is an overlap
                overlap_duration = overlap_end - overlap_start
                overlap_info = {
                    'annotation_start': ann_start,
                    'annotation_end': ann_end,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration,
                    'annotation_row': ann_row
                }
                overlapping_annotations.append(overlap_info)
        
        return overlapping_annotations
        
    except Exception as e:
        print(f"Error checking annotation overlap: {e}")
        return []


def check_spectrogram_overlaps(species, audiofile_name, start_ms, end_ms):
    """
    Check if the spectrogram time range overlaps with existing .pt spectrogram files.
    
    Args:
        species: Species name (Humpback, Beluga, Orca)
        audiofile_name: Audio filename (without extension)
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
    
    Returns:
        List of overlapping spectrogram files with details
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try both DataInput_New and legacy DataInput paths
    possible_label_files = [
        os.path.join(base_dir, f"DataInput_New/{species}/Processed/{species}_labels.csv"),
        os.path.join(base_dir, f"DataInput/{species}/LabelsOverlap400ms/{species}_labels.csv")
    ]
    
    labels_csv = None
    for path in possible_label_files:
        if os.path.exists(path):
            labels_csv = path
            break
    
    if labels_csv is None:
        print(f"Warning: No labels file found for {species}")
        return []
    
    try:
        # Load labels
        df_labels = pd.read_csv(labels_csv)
        
        # Add .wav extension if not present for matching
        audiofile_with_ext = audiofile_name + '.wav' if not audiofile_name.endswith('.wav') else audiofile_name
        
        # Filter labels for the same audiofile
        mask = df_labels['audiofile'] == audiofile_with_ext
        
        filtered_labels = df_labels[mask]
        
        if len(filtered_labels) == 0:
            print(f"No existing spectrograms found for {audiofile_with_ext}")
            return []
        
        print(f"Found {len(filtered_labels)} existing spectrograms for {audiofile_with_ext}")
        
        overlapping_spectrograms = []
        
        for _, label_row in filtered_labels.iterrows():
            # Extract start and end times from filename
            filename = label_row['filename']
            
            # Parse filename like "Iniskin_HB8_191020205232_0_2000.pt"
            try:
                base_name = os.path.splitext(filename)[0]
                parts = base_name.split('_')
                
                # Get the last two parts as start and end times
                if len(parts) >= 2:
                    spec_start_ms = int(parts[-2])
                    spec_end_ms = int(parts[-1])
                else:
                    continue  # Skip if can't parse
                
                # Check for overlap
                overlap_start_ms = max(start_ms, spec_start_ms)
                overlap_end_ms = min(end_ms, spec_end_ms)
                
                if overlap_start_ms < overlap_end_ms:
                    # There is an overlap
                    overlap_duration_ms = overlap_end_ms - overlap_start_ms
                    overlap_info = {
                        'spec_start_ms': spec_start_ms,
                        'spec_end_ms': spec_end_ms,
                        'overlap_start_ms': overlap_start_ms,
                        'overlap_end_ms': overlap_end_ms,
                        'overlap_duration_ms': overlap_duration_ms,
                        'filename': filename,
                        'label': label_row['label'],
                        'label_row': label_row
                    }
                    overlapping_spectrograms.append(overlap_info)
                    
            except (ValueError, IndexError) as e:
                print(f"Error parsing filename '{filename}': {e}")
                continue
        
        return overlapping_spectrograms
        
    except Exception as e:
        print(f"Error checking spectrogram overlap: {e}")
        return []


def find_best_matching_label_row(species, audiofile_name, start_ms, end_ms):
    """
    Find the best matching row in the labels file based on overlap.
    
    Returns the row with the highest overlap duration.
    """
    overlapping_spectrograms = check_spectrogram_overlaps(species, audiofile_name, start_ms, end_ms)
    
    if not overlapping_spectrograms:
        return None
    
    # Sort by overlap duration (descending) and return the best match
    best_match = max(overlapping_spectrograms, key=lambda x: x['overlap_duration_ms'])
    return best_match


def analyze_by_row(species, row_index, n_fft=1024, hop_length=256):
    """Analyze spectrogram by species and row index (legacy mode)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to NOAAWhalesV2
    
    # Try both DataInput_New and legacy DataInput paths
    possible_label_files = [
        os.path.join(base_dir, f"DataInput_New/{species}/Processed/{species}_labels.csv"),
        os.path.join(base_dir, f"DataInput/{species}/LabelsOverlap400ms/{species}_labels.csv")
    ]
    
    labels_csv = None
    for path in possible_label_files:
        if os.path.exists(path):
            labels_csv = path
            break
    
    if labels_csv is None:
        raise FileNotFoundError(f"Labels CSV not found for {species}")
    
    print(f"Loading labels from: {labels_csv}")
    print(f"Processing row index: {row_index}")
    
    # Load labels
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} total samples")
    
    # Check if row index is valid
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"Row index {row_index} is out of range. Valid range: 0 to {len(df)-1}")
    
    # Get the specific row
    row = df.iloc[row_index]
    
    # Extract row information
    row_species = row['species']
    location = row['location']
    label = row['label']
    filename = row['filename']
    
    # Check if audiofile column exists and use it, otherwise extract from filename
    if 'audiofile' in row and pd.notna(row['audiofile']):
        audiofile = row['audiofile']
        # Remove .wav extension if present
        if audiofile.endswith('.wav'):
            audiofile = audiofile[:-4]
    else:
        # Extract audiofile from filename
        filename_base = filename.replace('.pt', '')
        parts = filename_base.split('_')
        if len(parts) >= 3:
            # Find where the timestamps start
            for i in range(len(parts) - 2):
                if parts[i+1].isdigit() and parts[i+2].isdigit():
                    audiofile = '_'.join(parts[:i+1])
                    break
            else:
                raise ValueError(f"Could not parse audiofile from filename: {filename}")
        else:
            raise ValueError(f"Unexpected filename format: {filename}")
    
    print(f"\nProcessing row {row_index}:")
    print(f"  Species: {row_species}")
    print(f"  Location: {location}")
    print(f"  Label: {label}")
    print(f"  Audio file: {audiofile}")
    print(f"  Filename: {filename}")
    
    # Extract start and end times from filename
    start_ms, end_ms = extract_time_from_filename(filename)
    if start_ms is None or end_ms is None:
        raise ValueError("Failed to extract timing information from filename")
    
    print(f"  Time range: {start_ms}ms - {end_ms}ms")
    
    # Get paths
    if 'fullpath' in row and pd.notna(row['fullpath']):
        # Convert relative path to absolute path
        pt_path = row['fullpath']
        if pt_path.startswith('./'):
            pt_path = os.path.join(base_dir, pt_path[2:])  # Remove './' and prepend base_dir
        elif pt_path.startswith('/'):
            pt_path = base_dir + pt_path  # Prepend base_dir to absolute path starting with '/'
        else:
            pt_path = os.path.join(base_dir, pt_path)
    else:
        pt_path = os.path.join(base_dir, row.get('dirpath', '').lstrip('./'), filename)
    
    # Find audio file
    audio_path = find_audio_file_by_species_location(species, location, audiofile)
    if audio_path is None:
        raise FileNotFoundError(f"Audio file not found: {audiofile}")
    
    print(f"  Pre-generated spectrogram: {pt_path}")
    print(f"  Audio file path: {audio_path}")
    
    return {
        'species': row_species,
        'location': location,
        'label': label,
        'filename': filename,
        'audiofile': audiofile,
        'start_ms': start_ms,
        'end_ms': end_ms,
        'pt_path': pt_path,
        'audio_path': audio_path,
        'row_index': row_index
    }


def analyze_by_time(audiofile_name, start_ms, duration_ms=2000):
    """Analyze spectrogram by audiofile name and start time (new mode)."""
    end_ms = start_ms + duration_ms
    
    print(f"Searching for audio file: {audiofile_name}")
    print(f"Time range: {start_ms}ms - {end_ms}ms ({duration_ms/1000} seconds)")
    
    # Find the audio file
    audio_path, species = find_audio_file_by_name(audiofile_name)
    if audio_path is None:
        raise FileNotFoundError(f"Audio file not found: {audiofile_name}")
    
    print(f"Found audio file: {audio_path}")
    print(f"Detected species: {species}")
    
    # Extract location from path
    path_parts = audio_path.split(os.sep)
    if 'Audio' in path_parts:
        audio_idx = path_parts.index('Audio')
        if audio_idx + 1 < len(path_parts):
            location = path_parts[audio_idx + 1]
        else:
            location = "Unknown"
    else:
        location = "Unknown"
    
    print(f"Location: {location}")
    
    return {
        'species': species,
        'location': location,
        'label': None,  # Unknown for time-based analysis
        'filename': f"{audiofile_name}_{start_ms}_{end_ms}.pt",  # Generated filename
        'audiofile': audiofile_name,
        'start_ms': start_ms,
        'end_ms': end_ms,
        'pt_path': None,  # No pre-generated spectrogram
        'audio_path': audio_path,
        'row_index': None
    }


def create_visualization(analysis_info, spec1, spec2, time1, time2, freq1, freq2, 
                        annotation_overlaps, spectrogram_overlaps, sample_rate, output_dir):
    """Create comprehensive visualization with all overlaps."""
    
    start_ms = analysis_info['start_ms']
    end_ms = analysis_info['end_ms']
    species = analysis_info['species']
    location = analysis_info['location']
    audiofile = analysis_info['audiofile']
    label = analysis_info['label']
    row_index = analysis_info['row_index']
    
    # Calculate how many subplots we need
    num_plots = 1 if spec1 is None else 2  # Main spectrograms
    if annotation_overlaps:
        num_plots += 1
    if spectrogram_overlaps:
        num_plots += 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    duration_sec = (end_ms - start_ms) / 1000.0
    
    # Pre-generated spectrogram (if available)
    if spec1 is not None:
        axes[plot_idx].imshow(spec1, aspect='auto', origin='lower', cmap='magma',
                             extent=[0, duration_sec, freq1[0], freq1[-1]])
        axes[plot_idx].set_title(f"Pre-generated Spectrogram\n{analysis_info['filename']}", fontsize=12)
        axes[plot_idx].set_ylabel('Frequency (Hz)')
        if num_plots == 1:
            axes[plot_idx].set_xlabel('Time (seconds)')
        plot_idx += 1
    
    # Newly generated spectrogram
    if spec2 is not None:
        axes[plot_idx].imshow(spec2, aspect='auto', origin='lower', cmap='magma',
                             extent=[0, duration_sec, freq2[0], freq2[-1]])
        title = "Generated from Audio"
        if spec1 is None:
            title = "Generated Spectrogram"
        axes[plot_idx].set_title(f"{title}\n{audiofile} ({start_ms}-{end_ms}ms)", fontsize=12)
        axes[plot_idx].set_ylabel('Frequency (Hz)')
        if plot_idx == len(axes) - 1:
            axes[plot_idx].set_xlabel('Time (seconds)')
        plot_idx += 1
    
    # Annotation timeline if we have overlaps
    if annotation_overlaps and plot_idx < len(axes):
        ax_ann = axes[plot_idx]
        ax_ann.set_xlim(0, duration_sec)
        ax_ann.set_ylim(-0.5, len(annotation_overlaps) - 0.5)
        ax_ann.set_ylabel('Annotations')
        ax_ann.set_title('Annotation Overlaps')
        if plot_idx == len(axes) - 1:
            ax_ann.set_xlabel('Time (seconds)')
        
        spec_start_sec = start_ms / 1000.0
        colors = ['red', 'cyan', 'yellow', 'lime', 'orange', 'magenta']
        
        for i, overlap in enumerate(annotation_overlaps):
            color = colors[i % len(colors)]
            
            ann_start_rel = overlap['annotation_start'] - spec_start_sec
            ann_end_rel = overlap['annotation_end'] - spec_start_sec
            overlap_start_rel = overlap['overlap_start'] - spec_start_sec
            overlap_end_rel = overlap['overlap_end'] - spec_start_sec
            
            # Draw full annotation range
            ax_ann.hlines(i, ann_start_rel, ann_end_rel, 
                         colors=color, linewidth=10, alpha=0.7)
            
            # Mark overlap boundaries
            ax_ann.axvline(overlap_start_rel, color='black', linewidth=4, alpha=0.9)
            ax_ann.axvline(overlap_end_rel, color='black', linewidth=4, alpha=0.9)
            
            # Add duration label
            mid_time = (overlap_start_rel + overlap_end_rel) / 2
            duration_ms = overlap['overlap_duration'] * 1000
            ax_ann.text(mid_time, i, f'{duration_ms:.0f}ms', 
                       ha='center', va='center', fontsize=10, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        ax_ann.set_yticks(range(len(annotation_overlaps)))
        ax_ann.set_yticklabels([f'Ann{i+1}' for i in range(len(annotation_overlaps))])
        ax_ann.grid(True, alpha=0.3, axis='x')
        plot_idx += 1
    
    # Spectrogram overlap timeline
    if spectrogram_overlaps and plot_idx < len(axes):
        ax_spec = axes[plot_idx]
        ax_spec.set_xlim(0, duration_sec)
        ax_spec.set_ylim(-0.5, len(spectrogram_overlaps) - 0.5)
        ax_spec.set_ylabel('Existing Spectrograms')
        ax_spec.set_title('Spectrogram Overlaps')
        ax_spec.set_xlabel('Time (seconds)')
        
        colors = ['blue', 'green', 'purple', 'brown', 'pink', 'gray']
        
        for i, overlap in enumerate(spectrogram_overlaps):
            color = colors[i % len(colors)]
            
            # Convert to seconds relative to our spectrogram start
            spec_start_rel = (overlap['spec_start_ms'] - start_ms) / 1000.0
            spec_end_rel = (overlap['spec_end_ms'] - start_ms) / 1000.0
            overlap_start_rel = (overlap['overlap_start_ms'] - start_ms) / 1000.0
            overlap_end_rel = (overlap['overlap_end_ms'] - start_ms) / 1000.0
            
            # Draw full spectrogram range
            ax_spec.hlines(i, spec_start_rel, spec_end_rel, 
                          colors=color, linewidth=10, alpha=0.7)
            
            # Mark overlap boundaries
            ax_spec.axvline(overlap_start_rel, color='black', linewidth=4, alpha=0.9)
            ax_spec.axvline(overlap_end_rel, color='black', linewidth=4, alpha=0.9)
            
            # Add duration label
            mid_time = (overlap_start_rel + overlap_end_rel) / 2
            duration_ms = overlap['overlap_duration_ms']
            label_text = f"{duration_ms}ms\nL:{overlap['label']}"
            ax_spec.text(mid_time, i, label_text, 
                        ha='center', va='center', fontsize=9, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        ax_spec.set_yticks(range(len(spectrogram_overlaps)))
        ax_spec.set_yticklabels([f'Spec{i+1}' for i in range(len(spectrogram_overlaps))])
        ax_spec.grid(True, alpha=0.3, axis='x')
    
    # Overall title
    overlap_text = ""
    if annotation_overlaps:
        overlap_text += f" | {len(annotation_overlaps)} annotation(s)"
    if spectrogram_overlaps:
        overlap_text += f" | {len(spectrogram_overlaps)} spectrogram(s)"
    if not overlap_text:
        overlap_text = " | No overlaps"
    
    title_parts = [species, location, audiofile, f"{start_ms}-{end_ms}ms"]
    if label is not None:
        title_parts.insert(-1, f"Label:{label}")
    if row_index is not None:
        title_parts.insert(-1, f"Row:{row_index}")
    
    fig.suptitle(" | ".join(title_parts) + overlap_text, fontsize=12)
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    
    mode = "row" if row_index is not None else "time"
    output_filename = f"unified_{mode}_{species}_{location}_{audiofile}_{start_ms}_{end_ms}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Unified spectrogram analysis tool supporting both row-based and time-based analysis."
    )
    
    # Row-based mode arguments
    parser.add_argument(
        "--species", type=str,
        help="Species name for row-based analysis (e.g., Beluga, Humpback, Orca)."
    )
    parser.add_argument(
        "--row", type=int,
        help="Row index (0-based) in the species' labels CSV file."
    )
    
    # Time-based mode arguments
    parser.add_argument(
        "--audiofile", type=str,
        help="Partial audiofile name for time-based analysis (e.g., 'Iniskin_HB8_191020205232')."
    )
    parser.add_argument(
        "--start", type=int,
        help="Start time in milliseconds for time-based analysis."
    )
    parser.add_argument(
        "--duration", type=int, default=2000,
        help="Duration in milliseconds for time-based analysis (default: 2000ms = 2 seconds)."
    )
    
    # Common parameters
    parser.add_argument(
        "--n-fft", type=int, default=1024,
        help="FFT size for spectrogram generation (default: 1024)."
    )
    parser.add_argument(
        "--hop-length", type=int, default=256,
        help="Hop length for spectrogram generation (default: 256)."
    )
    
    args = parser.parse_args()
    
    # Determine mode and validate arguments
    if (args.species is not None and args.row is not None) and (args.audiofile is not None and args.start is not None):
        parser.error("Cannot specify both row-based (--species, --row) and time-based (--audiofile, --start) arguments")
    elif args.species is not None and args.row is not None:
        # Row-based mode
        print("="*60)
        print("ROW-BASED ANALYSIS MODE")
        print("="*60)
        analysis_info = analyze_by_row(args.species, args.row, args.n_fft, args.hop_length)
        mode = "row"
    elif args.audiofile is not None and args.start is not None:
        # Time-based mode
        print("="*60)
        print("TIME-BASED ANALYSIS MODE")
        print("="*60)
        analysis_info = analyze_by_time(args.audiofile, args.start, args.duration)
        mode = "time"
    else:
        parser.error("Must specify either (--species and --row) OR (--audiofile and --start)")
    
    # Extract common variables
    species = analysis_info['species']
    location = analysis_info['location']
    audiofile = analysis_info['audiofile']
    start_ms = analysis_info['start_ms']
    end_ms = analysis_info['end_ms']
    audio_path = analysis_info['audio_path']
    pt_path = analysis_info['pt_path']
    
    # Generate new spectrogram from audio
    print("\nGenerating spectrogram from audio...")
    spec2, time2, freq2, sample_rate = generate_spectrogram_from_audio(
        audio_path, start_ms, end_ms, n_fft=args.n_fft, hop_length=args.hop_length
    )
    
    if spec2 is None:
        print("Failed to generate spectrogram from audio")
        return
    
    print(f"Generated spectrogram: {spec2.shape}")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Load pre-generated spectrogram (if available)
    spec1, time1, freq1 = None, None, None
    if pt_path and os.path.exists(pt_path):
        print(f"\nLoading pre-generated spectrogram from: {pt_path}")
        spec1, time1, freq1 = load_pregenerated_spectrogram(pt_path, hop_length=args.hop_length, sample_rate=sample_rate)
        if spec1 is not None:
            print(f"Loaded pre-generated spectrogram: {spec1.shape}")
        else:
            print("Failed to load pre-generated spectrogram")
    elif mode == "row":
        print(f"Warning: Pre-generated spectrogram not found: {pt_path}")
    
    # Check for annotation overlaps
    print("\n" + "="*60)
    print("CHECKING ANNOTATION OVERLAPS")
    print("="*60)
    annotation_overlaps = check_annotation_overlaps(species, audiofile, start_ms, end_ms, sample_rate)
    
    if annotation_overlaps:
        print(f"\nFound {len(annotation_overlaps)} overlapping annotation(s):")
        for i, overlap in enumerate(annotation_overlaps, 1):
            ann_row = overlap['annotation_row']
            print(f"\n  {i}. Annotation overlap:")
            print(f"     Time: {overlap['annotation_start']:.3f}s - {overlap['annotation_end']:.3f}s")
            print(f"     Overlap: {overlap['overlap_start']:.3f}s - {overlap['overlap_end']:.3f}s")
            print(f"     Duration: {overlap['overlap_duration']:.3f}s")
            
            # Handle different annotation formats
            if 'lowFreq' in ann_row and pd.notna(ann_row['lowFreq']):
                print(f"     Frequency: {ann_row['lowFreq']:.1f}Hz - {ann_row['highFreq']:.1f}Hz")
            elif 'Low Freq (Hz)' in ann_row and pd.notna(ann_row['Low Freq (Hz)']):
                print(f"     Frequency: {ann_row['Low Freq (Hz)']:.1f}Hz - {ann_row['High Freq (Hz)']:.1f}Hz")
            
            if 'location' in ann_row:
                print(f"     Location: {ann_row['location']}")
    else:
        print("No overlapping annotations found.")
    
    # Check for spectrogram overlaps
    print("\n" + "="*60)
    print("CHECKING EXISTING SPECTROGRAM OVERLAPS")
    print("="*60)
    spectrogram_overlaps = check_spectrogram_overlaps(species, audiofile, start_ms, end_ms)
    
    if spectrogram_overlaps:
        print(f"\nFound {len(spectrogram_overlaps)} overlapping spectrogram(s):")
        for i, overlap in enumerate(spectrogram_overlaps, 1):
            print(f"\n  {i}. Spectrogram overlap:")
            print(f"     File: {overlap['filename']}")
            print(f"     Time: {overlap['spec_start_ms']}ms - {overlap['spec_end_ms']}ms")
            print(f"     Overlap: {overlap['overlap_start_ms']}ms - {overlap['overlap_end_ms']}ms")
            print(f"     Duration: {overlap['overlap_duration_ms']}ms")
            print(f"     Label: {overlap['label']}")
    else:
        print("No overlapping spectrograms found.")
    
    # Find best matching label row
    print("\n" + "="*60)
    print("BEST MATCHING LABEL ROW")
    print("="*60)
    best_match = find_best_matching_label_row(species, audiofile, start_ms, end_ms)
    
    if best_match:
        print(f"\nBest matching row:")
        print(f"  Filename: {best_match['filename']}")
        print(f"  Time range: {best_match['spec_start_ms']}ms - {best_match['spec_end_ms']}ms")
        print(f"  Overlap duration: {best_match['overlap_duration_ms']}ms")
        print(f"  Label: {best_match['label']}")
        print(f"  Location: {best_match['label_row']['location']}")
        
        # Show the full row data
        if mode == "time":  # Only show full details in time mode to avoid redundancy
            label_row = best_match['label_row']
            print(f"\nFull row details:")
            for column, value in label_row.items():
                print(f"  {column}: {value}")
    else:
        print("No matching label rows found.")
    
    # Create visualization
    print(f"\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    output_dir = os.path.join(os.path.dirname(__file__), "Comparisons")
    output_path = create_visualization(
        analysis_info, spec1, spec2, time1, time2, freq1, freq2,
        annotation_overlaps, spectrogram_overlaps, sample_rate, output_dir
    )
    
    print(f"\nVisualization saved: {output_path}")
    
    # Summary
    print(f"\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Mode: {mode.upper()}")
    print(f"Species: {species}")
    print(f"Location: {location}")
    print(f"Audio file: {audiofile}")
    print(f"Time range: {start_ms}ms - {end_ms}ms")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Generated spectrogram: {spec2.shape}")
    if spec1 is not None:
        print(f"Pre-generated spectrogram: {spec1.shape}")
    print(f"Annotation overlaps: {len(annotation_overlaps)}")
    print(f"Spectrogram overlaps: {len(spectrogram_overlaps)}")
    if best_match:
        print(f"Best match overlap: {best_match['overlap_duration_ms']}ms")


if __name__ == '__main__':
    main()
