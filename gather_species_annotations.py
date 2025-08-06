#!/usr/bin/env python3
"""
Script to process Humpback and Orca annotation files and create unified datasets.

This script:
1. Reads all CSV/TXT files from DataInput_New/{species}/Annotations/
2. Matches annotations to audio files based on filename patterns
3. Calculates duration in seconds and extracts audio metadata
4. Outputs a processed CSV with standardized columns

Supported species: Humpback, Orca

Audio file matching logic:
- Humpback: match date/time between .txt and .wav files (yyyymmdd_HHMMSS format)
- Orca: match encounter number between .txt and .wav files
"""

import pandas as pd
import numpy as np
import os
import glob
import librosa
import logging
import re
from datetime import datetime
import argparse

# ─── LOCATION NAME MAPPING ────────────────────────────────────────────────
# Maps various directory/file naming conventions to a canonical location name
LOCATION_NAME_MAP = {
    # Humpback - original naming conventions
    "AL16_BS4_humpback_data": "AL16_BS4",
    "AL16_NM1_humpback_data": "AL16_NM1", 
    "LCI_Chinitna_humpback_data": "Chinitna",
    "LCI_Iniskin_humpback_data": "Iniskin",
    "LCI_Port_Graham_humpback_data": "PtGraham",
    "AL16_BS4_humpback_selections": "AL16_BS4",
    "AL16_NM1_humpback_selections": "AL16_NM1",
    "LCI_Chinitna_humpback_selections": "Chinitna", 
    "LCI_Iniskin_humpback_selections": "Iniskin",
    "LCI_Port_Graham_humpback_selections": "PtGraham",
    # Humpback - new directory structure (direct mapping)
    "AL16_BS4": "AL16_BS4",
    "AL16_NM1": "AL16_NM1",
    "Chinitna": "Chinitna",
    "Iniskin": "Iniskin",
    "PtGraham": "PtGraham",
    # Orca
    "Chinitna": "Chinitna",
    "Iniskin": "Iniskin", 
    "PtGraham": "PtGraham",
    "Port Graham": "PtGraham",
    "SWCorner": "SWCorner",
    "SWcorner": "SWCorner",  # handle case difference
}

def setup_logging(species):
    """Set up logging configuration."""
    os.makedirs("Logs", exist_ok=True)
    log_file = f'Logs/gather_{species.lower()}_annotations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def normalize_location_name(raw_location):
    """Normalize location name using the mapping."""
    return LOCATION_NAME_MAP.get(raw_location, raw_location)

def extract_humpback_timestamp(filename):
    """
    Extract timestamp from Humpback filename.
    Expected formats: 
    - AL16: {location}_{yyyymmdd}_{HHMMSS}.{extension}
    - LCI: {location}_HB{#}_{yymmddHHMMSS}.{extension}
    Returns: (location, normalized_timestamp) or (None, None)
    """
    # Pattern 1: AL16 format - location_yyyymmdd_HHMMSS
    pattern1 = r'([^_]+)_(\d{8})_(\d{6})'
    match1 = re.search(pattern1, filename)
    if match1:
        location = match1.group(1)
        date_part = match1.group(2)  # yyyymmdd
        time_part = match1.group(3)  # HHMMSS
        timestamp = f"{date_part}_{time_part}"
        return location, timestamp
    
    # Pattern 2: LCI format - location_HB#_yymmddHHMMSS or location_HB#.#_yymmddHHMMSS
    pattern2 = r'([^_]+)_HB[\d\.]+_(\d{12})'
    match2 = re.search(pattern2, filename)
    if match2:
        location = match2.group(1)
        full_timestamp = match2.group(2)  # yymmddHHMMSS (12 digits)
        
        # Convert yy to yyyy (assume 19xx for these files)
        yy = full_timestamp[:2]
        mm = full_timestamp[2:4]
        dd = full_timestamp[4:6]
        HHMMSS = full_timestamp[6:12]
        
        # Convert 2-digit year to 4-digit (19xx for these files)
        yyyy = f"19{yy}"
        timestamp = f"{yyyy}{mm}{dd}_{HHMMSS}"
        
        return location, timestamp
    
    return None, None

def extract_orca_encounter(filename):
    """
    Extract encounter number from Orca filename.
    Expected format: {location}_encounter_{number}.{extension}
    Returns: (location, encounter_number) or (None, None)
    """
    pattern = r'([^_]+)_encounter_(\d+)'
    match = re.search(pattern, filename)
    if match:
        location = match.group(1)
        encounter_num = match.group(2)
        return location, encounter_num
    return None, None

def load_audio_files_metadata(species):
    """
    Load metadata for all audio files for the given species.
    
    Returns:
        DataFrame with audio file metadata
    """
    audio_files = []
    base_path = f"DataInput_New/{species}/Audio"
    
    if not os.path.exists(base_path):
        logging.error(f"Base path {base_path} does not exist")
        return pd.DataFrame()
    
    logging.info(f"Scanning for {species} audio files in {base_path}")
    
    # Find all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                subfolder = os.path.relpath(root, base_path)
                
                try:
                    # Get audio metadata
                    sample_rate = librosa.get_samplerate(full_path)
                    
                    if species == "Humpback":
                        location, timestamp = extract_humpback_timestamp(file)
                        if location and timestamp:
                            # Normalize location name
                            normalized_location = normalize_location_name(subfolder)
                            
                            audio_files.append({
                                'species': species,
                                'location': normalized_location,
                                'subfolder': subfolder,
                                'filename': file,
                                'timestamp': timestamp,
                                'audiofile_path': full_path,
                                'sample_rate': sample_rate
                            })
                    
                    elif species == "Orca":
                        location, encounter = extract_orca_encounter(file)
                        if location and encounter:
                            # Normalize location name 
                            normalized_location = normalize_location_name(location)
                            
                            audio_files.append({
                                'species': species,
                                'location': normalized_location,
                                'subfolder': subfolder,
                                'filename': file,
                                'encounter': encounter,
                                'audiofile_path': full_path,
                                'sample_rate': sample_rate
                            })
                
                except Exception as e:
                    logging.warning(f"Error processing audio file {full_path}: {e}")
    
    df = pd.DataFrame(audio_files)
    logging.info(f"Loaded {len(df)} {species} audio files")
    return df

def find_matching_audio_file(annotation_file, audio_files_df, species):
    """
    Find the matching audio file for an annotation file.
    
    Args:
        annotation_file: Path to annotation file
        audio_files_df: DataFrame of audio files
        species: "Humpback" or "Orca"
    
    Returns:
        Dictionary with audio file info or None
    """
    filename = os.path.basename(annotation_file)
    
    if species == "Humpback":
        location, timestamp = extract_humpback_timestamp(filename)
        if not location or not timestamp:
            logging.debug(f"Could not extract timestamp from {filename}")
            return None
        
        # Find matching audio file
        matches = audio_files_df[
            (audio_files_df['timestamp'] == timestamp)
        ]
        
        if len(matches) == 0:
            logging.debug(f"No audio file found for timestamp {timestamp}")
            return None
        elif len(matches) > 1:
            logging.warning(f"Multiple audio files found for timestamp {timestamp}, using first")
        
        return matches.iloc[0].to_dict()
    
    elif species == "Orca":
        location, encounter = extract_orca_encounter(filename)
        if not location or not encounter:
            logging.debug(f"Could not extract encounter from {filename}")
            return None
        
        # Normalize location for matching
        normalized_location = normalize_location_name(location)
        
        # Find matching audio file
        matches = audio_files_df[
            (audio_files_df['location'] == normalized_location) &
            (audio_files_df['encounter'] == encounter)
        ]
        
        if len(matches) == 0:
            logging.debug(f"No audio file found for {normalized_location} encounter {encounter}")
            return None
        elif len(matches) > 1:
            logging.warning(f"Multiple audio files found for {normalized_location} encounter {encounter}, using first")
        
        return matches.iloc[0].to_dict()
    
    return None

def process_annotation_file(annotation_file, audio_files_df, species):
    """
    Process a single annotation file.
    
    Args:
        annotation_file: Path to annotation file
        audio_files_df: DataFrame of audio files
        species: "Humpback" or "Orca"
    
    Returns:
        DataFrame with processed annotations
    """
    try:
        logging.info(f"Processing {annotation_file}")
        
        # Read annotation file (could be CSV or TSV)
        if annotation_file.endswith('.csv'):
            df = pd.read_csv(annotation_file, low_memory=False)
        else:
            # Try tab-separated first, then comma-separated
            try:
                df = pd.read_csv(annotation_file, sep='\t', low_memory=False)
            except:
                df = pd.read_csv(annotation_file, sep=',', low_memory=False)
        
        logging.info(f"  Loaded {len(df)} rows")
        
        if df.empty:
            logging.warning(f"  No data in {annotation_file}")
            return pd.DataFrame()
        
        # Remove header duplicates (common in concatenated files)
        if 'Begin Time (s)' in df.columns:
            df = df[df['Begin Time (s)'] != 'Begin Time (s)']
        
        # Check required columns
        required_cols = ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"  Missing required columns in {annotation_file}: {missing_cols}")
            return pd.DataFrame()
        
        # Convert numeric columns
        numeric_cols = ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid numeric data
        initial_count = len(df)
        df = df.dropna(subset=numeric_cols)
        if len(df) < initial_count:
            logging.warning(f"  Dropped {initial_count - len(df)} rows with invalid numeric data")
        
        if df.empty:
            logging.warning(f"  No valid data remaining in {annotation_file}")
            return pd.DataFrame()
        
        # Add basic columns
        df['Species'] = species
        df['source_annotation_file'] = os.path.basename(annotation_file)
        
        # Calculate derived columns
        df['startSeconds'] = df['Begin Time (s)']
        df['durationSeconds'] = df['End Time (s)'] - df['Begin Time (s)']
        df['lowFreq'] = df['Low Freq (Hz)']
        df['highFreq'] = df['High Freq (Hz)']
        
        # Log any missing values for awareness
        missing_start = df['startSeconds'].isna().sum()
        missing_duration = df['durationSeconds'].isna().sum()
        missing_lowfreq = df['lowFreq'].isna().sum()
        missing_highfreq = df['highFreq'].isna().sum()
        
        if missing_start > 0:
            logging.warning(f"  Found {missing_start} rows with missing startSeconds")
        if missing_duration > 0:
            logging.warning(f"  Found {missing_duration} rows with missing durationSeconds")
        if missing_lowfreq > 0:
            logging.warning(f"  Found {missing_lowfreq} rows with missing lowFreq")
        if missing_highfreq > 0:
            logging.warning(f"  Found {missing_highfreq} rows with missing highFreq")
        
        # Find matching audio file
        audio_match = find_matching_audio_file(annotation_file, audio_files_df, species)
        
        if audio_match:
            # Verify the audio file actually exists on disk
            audio_path = audio_match['audiofile_path']
            if os.path.exists(audio_path):
                df['audiofile_path'] = audio_path
                df['sampleRate'] = audio_match['sample_rate']
                df['location'] = audio_match['location']
                logging.info(f"  Matched to audio file: {audio_match['filename']}")
            else:
                df['audiofile_path'] = None
                df['sampleRate'] = None
                # Try to infer location from annotation file path
                path_parts = annotation_file.split(os.sep)
                if len(path_parts) >= 2:
                    subfolder = path_parts[-2]  # parent directory
                    df['location'] = normalize_location_name(subfolder)
                else:
                    df['location'] = 'Unknown'
                logging.error(f"  Matched audio file does not exist: {audio_path}")
        else:
            df['audiofile_path'] = None
            df['sampleRate'] = None
            # Try to infer location from annotation file path
            path_parts = annotation_file.split(os.sep)
            if len(path_parts) >= 2:
                subfolder = path_parts[-2]  # parent directory
                df['location'] = normalize_location_name(subfolder)
            else:
                df['location'] = 'Unknown'
            
            logging.warning(f"  No matching audio file found")
        
        logging.info(f"  Processed {len(df)} annotations")
        return df
        
    except Exception as e:
        logging.error(f"Error processing {annotation_file}: {e}")
        return pd.DataFrame()

def main(species):
    """Main processing function."""
    logging.info(f"Starting {species} annotation processing")
    
    # Load audio file metadata
    logging.info("Loading audio file metadata...")
    audio_files_df = load_audio_files_metadata(species)
    
    if audio_files_df.empty:
        logging.warning(f"No audio files found for {species}")
    
    # Find annotation files
    annotations_path = f"DataInput_New/{species}/Annotations"
    if not os.path.exists(annotations_path):
        logging.error(f"Species directory {annotations_path} does not exist")
        return
    
    # Find all annotation files (CSV and TXT) in location subdirectories
    annotation_files = []
    
    if species == "Humpback":
        # Look in location directories for Humpback (AL16_BS4, AL16_NM1, Chinitna, etc.)
        for root, dirs, files in os.walk(annotations_path):
            # Only go one level deep (location directories)
            level = root.replace(annotations_path, '').count(os.sep)
            if level == 1:  # Location directories like AL16_BS4, Chinitna, etc.
                for ext in ['*.csv', '*.txt']:
                    annotation_files.extend(glob.glob(os.path.join(root, ext)))
    elif species == "Orca":
        # Look in location directories for Orca (avoid subdirectories like Annotations if they exist)
        for root, dirs, files in os.walk(annotations_path):
            # Skip nested subdirectories beyond one level
            level = root.replace(annotations_path, '').count(os.sep)
            if level == 1:  # Only go one level deep (location directories)
                for ext in ['*.csv', '*.txt']:
                    found_files = glob.glob(os.path.join(root, ext))
                    # Filter out any processed annotation files that might exist
                    filtered_files = [f for f in found_files if not f.endswith('_annotations.csv')]
                    annotation_files.extend(filtered_files)
    
    logging.info(f"Found {len(annotation_files)} annotation files")
    
    if not annotation_files:
        logging.error("No annotation files found")
        return
    
    # Process each annotation file
    all_annotations = []
    for annotation_file in annotation_files:
        df = process_annotation_file(annotation_file, audio_files_df, species)
        if not df.empty:
            all_annotations.append(df)
    
    if not all_annotations:
        logging.error("No annotations were processed successfully")
        return
    
    # Combine all annotations
    combined_df = pd.concat(all_annotations, ignore_index=True)
    logging.info(f"Combined {len(combined_df)} total annotations")
    
    # Select final columns
    final_columns = [
        'Species',
        'location', 
        'startSeconds',
        'durationSeconds',
        'audiofile_path',
        'lowFreq',
        'highFreq',
        'sampleRate',
        'source_annotation_file'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in final_columns if col in combined_df.columns]
    final_df = combined_df[available_columns].copy()
    
    # Save processed annotations
    output_path = f"DataInput_New/{species}/Processed/{species}_annotations_processed.csv"
    final_df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(final_df)} processed annotations to {output_path}")
    
    # Print summary statistics
    logging.info("\nSUMMARY STATISTICS:")
    logging.info(f"Total annotations: {len(final_df)}")
    
    # Count by location
    if 'location' in final_df.columns:
        location_counts = final_df['location'].value_counts()
        logging.info("Location distribution:")
        for location, count in location_counts.items():
            # Calculate match rate per location
            location_data = final_df[final_df['location'] == location]
            matched_in_location = location_data['audiofile_path'].notna().sum()
            match_rate = (matched_in_location / len(location_data)) * 100 if len(location_data) > 0 else 0
            logging.info(f"  {location}: {count} annotations ({matched_in_location} matched, {match_rate:.1f}%)")
    
    # Count matched vs unmatched
    matched = final_df['audiofile_path'].notna().sum()
    unmatched = len(final_df) - matched
    overall_match_rate = (matched / len(final_df)) * 100 if len(final_df) > 0 else 0
    logging.info(f"Overall: {matched} matched, {unmatched} unmatched ({overall_match_rate:.1f}% match rate)")
    
    if matched > 0:
        avg_duration = final_df[final_df['durationSeconds'].notna()]['durationSeconds'].mean()
        logging.info(f"Average annotation duration: {avg_duration:.3f} seconds")
    
    logging.info("Processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Humpback or Orca annotations')
    parser.add_argument('species', choices=['Humpback', 'Orca'], 
                        help='Species to process: Humpback or Orca')
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.species)
    
    try:
        main(args.species)
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise
