"""
LABEL DISTRIBUTION ANALYSIS AND DATA SPLIT RECOMMENDATION SCRIPT

This script analyzes the distribution of whale call labels across different species and locations,
then recommends how to split the data into train/validation/test sets.

What it does:
1. Reads label CSV files from each species directory in the DataInput_New folder
2. Combines all labels and analyzes the distribution by species and location
3. Creates visualizations showing label distributions (percentages and counts)
4. Recommends optimal data splits (train/validation/test) to ensure each split has
   a minimum number of positive samples for each species

Parameters to configure:
- base_dir: The base directory containing species folders (default: "DataInput_New")
- MIN_POS: Minimum number of positive samples per species per split (default: 1000)
  Adjust this value based on your dataset size and training requirements
- species: Specify which species to include in the analysis

Expected directory structure:
DataInput_New/
  ├── Species1/
  │   └── Processed/
  │       └── LabelsOverlap400ms/
  │           └── Species1_labels.csv
  ├── Species2/
  │   └── Processed/
  │       └── LabelsOverlap400ms/
  │           └── Species2_labels.csv
  └── ...

Usage:
    # Analyze all available species
    python generate_assignations.py
    
    # Analyze specific species only
    python generate_assignations.py --species Beluga Humpback
    
    # Custom minimum positive samples
    python generate_assignations.py --species Beluga --min_pos 500
    
    # Custom base directory
    python generate_assignations.py --base_dir /path/to/data --species Orca

Output files:
- label_distribution_by_species (%).png: Stacked bar chart showing percentage distribution
- label_distribution_by_species.png: Side-by-side bar chart showing absolute counts
- Console output: Recommended location assignments for train/validation/test splits

The script uses a greedy algorithm to assign locations to splits, prioritizing the most
underrepresented species in each split to ensure balanced training data.
"""

import os
import pandas as pd
import argparse
from glob import glob
from datetime import datetime
import logging

# Setup logging
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "Logs", "CheckLabelDistribution")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"label_distribution_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will still print to console
    ]
)

def log_and_print(message):
    """Helper function to both log and print messages"""
    logging.info(message)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze label distribution and recommend data splits for whale species",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze all available species
    python generate_assignations.py
    
    # Analyze specific species only
    python generate_assignations.py --species Beluga Humpback
    
    # Custom minimum positive samples
    python generate_assignations.py --species Beluga --min_pos 500
    
    # Custom base directory and species
    python generate_assignations.py --base_dir /path/to/data --species Orca Beluga
        """
    )
    
    parser.add_argument(
        '--base_dir',
    default='/home/radodhia/ssdprivate/NOAAWhalesV3/DataInput_New',
    help='Base directory containing species folders (default: /home/radodhia/ssdprivate/NOAAWhalesV3/DataInput_New)'
    )
    
    parser.add_argument(
        '--species',
        nargs='*',
        default=None,
        help='Species to include in analysis (e.g., --species Beluga Humpback). If not specified, all available species will be used.'
    )
    
    parser.add_argument(
        '--min_pos',
        type=int,
        default=1000,
        help='Minimum number of positive samples per species per split (default: 1000)'
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    base_dir = args.base_dir
    specified_species = args.species
    MIN_POS = args.min_pos
    
    log_and_print(f"Starting analysis with parameters:")
    log_and_print(f"  Base directory: {base_dir}")
    log_and_print(f"  Specified species: {specified_species if specified_species else 'All available'}")
    log_and_print(f"  Minimum positive samples per split: {MIN_POS}")

    # Find all species directories
    all_species_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter species based on user specification
    if specified_species:
        # Check if all specified species exist
        missing_species = [s for s in specified_species if s not in all_species_dirs]
        if missing_species:
            log_and_print(f"WARNING: The following specified species were not found: {missing_species}")
            log_and_print(f"Available species: {all_species_dirs}")
        
        species_dirs = [s for s in specified_species if s in all_species_dirs]
        if not species_dirs:
            log_and_print("ERROR: None of the specified species were found in the base directory.")
            return
    else:
        species_dirs = all_species_dirs
    
    log_and_print(f"Processing species: {species_dirs}")

    # Collect all label dataframes
    dfs = []
    for species in species_dirs:
        label_path = os.path.join(base_dir, species, "Processed/LabelsOverlap400ms", f"{species}_labels.csv")
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            df['species'] = species  # Add species column
            dfs.append(df)
            log_and_print(f"Loaded {len(df)} records for {species}")
        else:
            log_and_print(f"WARNING: Label file not found for {species} at {label_path}")

    # Combine all dataframes
    if dfs:
        all_labels = pd.concat(dfs, ignore_index=True)
        log_and_print(f"Combined total records: {len(all_labels)}")
        
        # Count by species and location
        if 'location' in all_labels.columns and 'label' in all_labels.columns:
            counts = all_labels.groupby(['species', 'location', 'label']).size().reset_index(name='count')
            log_and_print("Label counts by species, location, and label:")
            log_and_print(str(counts))
        else:
            log_and_print("Required columns 'location' or 'label' not found in the CSV files.")
            return
    else:
        log_and_print("No label files found for the specified species.")
        return

    import matplotlib.pyplot as plt

    if dfs and 'location' in all_labels.columns and 'label' in all_labels.columns:
        # Pivot for plotting
        pivot_counts = counts.pivot_table(index='species', columns='label', values='count', aggfunc='sum', fill_value=0)
        # Stacked barchart in percentages
        percent = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
        percent.plot(kind='bar', stacked=True)
        plt.ylabel('Percentage')
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        species_list = ', '.join(species_dirs)
        plt.title(f'Label Distribution by Species (Percentages) - {now}\nSpecies: {species_list}')
        plt.legend(title='Label')
        plt.tight_layout()
        output_file = f"label_distribution_by_species_({'_'.join(species_dirs)})_percent.png"
        plt.savefig(output_file)
        log_and_print(f"Saved percentage distribution plot: {output_file}")
        plt.close()

        # Side-by-side barchart with total numbers
        pivot_counts.plot(kind='bar')
        plt.ylabel('Count')
        plt.title(f'Label Distribution by Species (Counts) - {now}\nSpecies: {species_list}')
        plt.legend(title='Label')
        plt.tight_layout()
        output_file = f"label_distribution_by_species_({'_'.join(species_dirs)})_counts.png"
        plt.savefig(output_file)
        log_and_print(f"Saved count distribution plot: {output_file}")
        plt.close()

    # --- SUGGESTED SPLIT LOGIC ---
    # Pivot to get positive counts per (species, location)
    pos_counts = counts[counts['label'] == 1].pivot(index='location', columns='species', values='count').fillna(0).astype(int)

    # Print table for review
    log_and_print("\nPositive (label=1) counts per location/species:")
    log_and_print(str(pos_counts))

    # Prepare for assignment
    locations = pos_counts.index.tolist()
    species = pos_counts.columns.tolist()

    # Initialize splits and their species counts
    def empty_split():
        return {s: 0 for s in species}
    splits = {'train': [], 'validation': [], 'test': []}
    split_counts = {'train': empty_split(), 'validation': empty_split(), 'test': empty_split()}

    # Sort locations by total positives only (ignore min_species)
    pos_counts['total'] = pos_counts.sum(axis=1)
    pos_counts_sorted = pos_counts.sort_values(['total'], ascending=False)

    # Greedy assignment: assign each location to the split where it helps the most underrepresented species
    for loc in pos_counts_sorted.index:
        loc_counts = pos_counts.loc[loc]
        # For each split, compute how much this location helps the least-represented species
        best_split = None
        best_gain = -1
        for split in splits:
            gain = sum(
                max(0, min(MIN_POS - split_counts[split][s], loc_counts[s]))
                for s in species
            )
            if gain > best_gain:
                best_gain = gain
                best_split = split
        splits[best_split].append(loc)
        for s in species:
            split_counts[best_split][s] += loc_counts[s]

    # Print suggested splits and their species coverage
    log_and_print(f"\nMinimum positive samples per species per split: {MIN_POS}")
    log_and_print("\nSuggested location splits (with positive counts per species):")
    for split, locs in splits.items():
        log_and_print(f"{split}: {locs}")
        split_df = pos_counts.loc[locs]
        sums = split_df.sum()
        log_and_print(str(sums))
        for s in species:
            if sums[s] < MIN_POS:
                log_and_print(f"  WARNING: {split} has only {sums[s]} positives for {s} (less than {MIN_POS})")

    # Print summary table: positive, negative, and proportion of positive samples per species per split
    neg_counts = {}
    pos_props = {}
    for split, locs in splits.items():
        split_df = all_labels[all_labels['location'].isin(locs)]
        neg_counts[split] = {}
        pos_props[split] = {}
        for s in species:
            n_pos = split_df[(split_df['species'] == s) & (split_df['label'] == 1)].shape[0]
            n_neg = split_df[(split_df['species'] == s) & (split_df['label'] == 0)].shape[0]
            total = n_pos + n_neg
            neg_counts[split][s] = n_neg
            pos_props[split][s] = round(n_pos / total, 3) if total > 0 else 0.0
    summary = pd.DataFrame({split: [split_counts[split][s] for s in species] for split in splits}).T
    # Combine pos, neg, and prop columns into one table
    combined = pd.DataFrame(index=splits.keys())
    for s in species:
        combined[f"{s} (pos)"] = [split_counts[split][s] for split in splits]
        combined[f"{s} (neg)"] = [neg_counts[split][s] for split in splits]
        combined[f"{s} (pos_prop)"] = [pos_props[split][s] for split in splits]
    log_and_print("\nSummary: Samples per species per split (stage):")
    log_and_print(str(combined))

    # Optionally, you can adjust the splits above to better balance or to meet your own criteria.
    # You can now use these splits to assign locations in your main pipeline.

    log_and_print(f"\nAnalysis completed. Log file saved to: {log_file}")

if __name__ == "__main__":
    main()
