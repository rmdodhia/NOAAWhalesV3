# DataInput_New Directory Structure

This document describes the actual directory structure of the NOAAWhalesV2 project as of July 31, 2025.

## Top-Level Structure

```
DataInput_New/
├── Beluga/
├── Humpback/
├── Orca/
└── Combined/
```

## Species-Specific Directory Structure

Each species follows a consistent organization pattern:

### Beluga

```
Beluga/
├── Audio/
│   ├── 201D/                     # Contains ~1000 .wav files
│   │   ├── 604536840.180809230202.wav
│   │   ├── 604536840.180809231302.wav
│   │   ├── 604536840.180809232402.wav
│   │   └── ... (~1000 more files)
│   ├── 206D/                     # Contains audio files
│   ├── 213D/                     # Contains audio files
│   ├── 214D/                     # Contains audio files
│   ├── 215D/                     # Contains audio files
│   ├── 216D/                     # Contains audio files
│   ├── 218D/                     # Contains audio files
│   └── 223D/                     # Contains audio files
├── Annotations/
│   ├── 201D_PG_WandM_Detector-processed.csv
│   ├── 206D_PG_WandM_Detector-processed.csv
│   ├── 213D_PG_WandM_Detector-processed.csv
│   ├── 214D_PG_WandM_Detector-processed.csv
│   └── ... (more deployment CSVs)
├── Metadata/                     # Deployment info and catalogs
├── Processed/
│   ├── LabelsOverlap400ms/       # Directory containing processed labels
│   │   ├── Beluga_201D_overlap400ms_spectrogram_labels.csv
│   │   ├── Beluga_206D_overlap400ms_spectrogram_labels.csv
│   │   ├── Beluga_213D_overlap400ms_spectrogram_labels.csv
|   │   └── Beluga_labels.csv
│   │   └── ... (more deployment label files)
│   ├── SpectrogramsOverlap400ms/ # Directory containing spectrograms
│   ├── Beluga_201D_overlap400ms_spectrogram_labels.csv
│   ├── Beluga_206D_overlap400ms_spectrogram_labels.csv
│   ├── Beluga_213D_overlap400ms_spectrogram_labels.csv
│   ├── Beluga_214D_overlap400ms_spectrogram_labels.csv
│   ├── Beluga_215D_overlap400ms_spectrogram_labels.csv
│   ├── Beluga_223D_overlap400ms_spectrogram_labels.csv
│   ├── Beluga_annotations_processed.csv
└── [Clean structure with no generated files]
```

### Humpback

```
Humpback/
├── Audio/
│   ├── AL16_BS4/                 # Arctic deployment audio files
│   │   ├── AU-ALBS04_20161010_090000.wav
│   │   ├── AU-ALBS04_20161010_091000.wav
│   │   ├── AU-ALBS04_20161010_092000.wav
│   │   └── ... (200+ more files)
│   ├── AL16_NM1/                 # Arctic deployment audio files
│   ├── Chinitna/                 # Location-based audio files
│   │   ├── Chinitna_HB1_191002205500.wav
│   │   ├── Chinitna_HB2_191007205404.wav
│   │   ├── Chinitna_HB3_191009205342.wav
│   │   └── ... (20+ more files)
│   ├── Iniskin/                  # Location-based audio files
│   └── PtGraham/                 # Location-based audio files
├── Annotations/
│   ├── AL16_BS4/                 # Corresponding annotation files
│   │   ├── AU-ALBS04_20161010_090000.Table.1.selections.txt
│   │   ├── AU-ALBS04_20161010_091000.Table.1.selections.txt
│   │   ├── AU-ALBS04_20161010_092000.Table.1.selections.txt
│   │   └── ... (matching .selections.txt files)
│   ├── AL16_NM1/
│   ├── Chinitna/
│   │   ├── Chinitna_HB1_191002205500.Table.1.selections.txt
│   │   ├── Chinitna_HB2_191007205404.Table.1.selections.txt
│   │   ├── Chinitna_HB3_191009205342.Table.1.selections.txt
│   │   └── ... (matching .selections.txt files)
│   ├── Iniskin/
│   └── PtGraham/
├── Metadata/                     # Deployment info and catalogs
├── Processed/
│   ├── LabelsOverlap400ms/       # Directory containing processed labels
│   │   ├── Humpback_AL16_BS4_overlap400ms_spectrogram_labels.csv
│   │   ├── Humpback_Chinitna_overlap400ms_spectrogram_labels.csv
│   │   ├── Humpback_Iniskin_overlap400ms_spectrogram_labels.csv
|   │   └── Humpback_labels.csv
│   │   └── ... (location-specific label files)
│   ├── SpectrogramsOverlap400ms/ # Directory containing spectrograms
│   ├── Humpback_annotations_processed.csv
└── Metadata/                     # Deployment info and catalogs
```

### Orca

```
Orca/
├── Audio/
│   ├── Chinitna/                 # Location-based audio files
│   │   ├── Chinitna_encounter_1.wav
│   │   ├── Chinitna_encounter_2.wav
│   │   ├── Chinitna_encounter_3.wav
│   │   ├── Chinitna_encounter_4 (1).wav
│   │   └── ... (11 files total)
│   ├── Iniskin/                  # Location-based audio files
│   ├── PtGraham/                 # Location-based audio files
│   └── SWCorner/                 # Location-based audio files
├── Annotations/
│   ├── Chinitna/                 # Corresponding annotation files
│   │   ├── Chinitna_encounter_1.Table.1.selections.txt
│   │   ├── Chinitna_encounter_2.Table.1.selections.txt
│   │   ├── Chinitna_encounter_3.Table.1.selections.txt
│   │   ├── Chinitna_encounter_4 (1).Table.1.selections.txt
│   │   └── ... (matching .selections.txt files)
│   ├── Iniskin/
│   ├── PtGraham/
│   └── SWCorner/
├── Metadata/                     # Deployment info and catalogs
├── Processed/
│   ├── LabelsOverlap400ms/       # Directory containing processed labels
│   │   ├── Orca_Chinitna_overlap400ms_spectrogram_labels.csv
│   │   ├── Orca_Iniskin_overlap400ms_spectrogram_labels.csv
│   │   ├── Orca_PtGraham_overlap400ms_spectrogram_labels.csv
│   │   ├── Orca_SWCorner_overlap400ms_spectrogram_labels.csv
│   │   └── Orca_labels.csv
│   ├── SpectrogramsOverlap400ms/ # Directory containing spectrograms
│   ├── Orca_Chinitna_overlap400ms_spectrogram_labels.csv
│   ├── Orca_Iniskin_overlap400ms_spectrogram_labels.csv
│   ├── Orca_PtGraham_overlap400ms_spectrogram_labels.csv
│   ├── Orca_SWCorner_overlap400ms_spectrogram_labels.csv
│   ├── Orca_annotations_processed.csv
└── Metadata/                     # Deployment info and catalogs
```

## Combined Directory

```
Combined/
├── labels_overlap400ms_binary.csv      # Binary classification labels
└── labels_overlap400ms_three_species.csv # Multi-species classification labels
```

## File Naming Conventions

### Audio Files
- **Beluga**: `{deployment_id}.{yymmddhhmmss}.wav` (e.g., `604536840.180809230202.wav`)
- **Humpback**: 
  - Arctic: `AU-{deployment}_{yyyymmdd}_{hhmmss}.wav`
  - LCI: `{location}_HB{#}_{yymmddhhmmss}.wav`
- **Orca**: `{location}_encounter_{#}.wav`

### Processed Files
- `{Species}_annotations_processed.csv` - Main processed annotations
- `{Species}_labels.csv` - Generated labels for training
- `{Species}_{location}_overlap400ms_spectrogram_labels.csv` - Location-specific labels

## Key Differences from Documentation

1. **Beluga structure**: Uses deployment IDs (201D, 206D, etc.) instead of location names
2. **Processed labels**: Multiple CSV files for different processing stages and location breakdowns
3. **Label directories**: Both `LabelsOverlap400ms/` and `SpectrogramsOverlap400ms/` subdirectories exist
4. **Combined outputs**: Cross-species analysis files in the `Combined/` directory

## Usage Notes

- For the `check_label_distribution.py` script, label files are located in `{Species}/Processed/{Species}_labels.csv`
- The directory structure supports both location-based analysis and deployment-based analysis

## Processing Pipeline Files

The structure indicates a processing pipeline that:
1. Takes raw audio and annotations from `Audio/` and `Annotations/`
2. Processes them into standardized formats in `Processed/`
3. Generates training labels with 400ms overlap windows
4. Creates location-specific and combined datasets
