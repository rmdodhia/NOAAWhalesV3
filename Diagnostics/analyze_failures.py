#!/usr/bin/env python3
"""
Comprehensive Failure Analysis Script for ResNet18 Binary Whale Classification
==============================================================================

This script provides advanced analysis of test results CSV files to identify and visualize
misclassifications with detailed statistical reporting and confidence-based filtering.

Enhanced Features:
- Complete Classification Metrics: Outputs TP/TN/FP/FN counts and percentages
- Dual Statistical Views: Overall metrics + filtered metrics (when confidence filtering applied)
- Species-Specific Analysis: Filter by species with per-species image limits for targeted analysis
- Advanced Confidence Filtering: Filter failures by confidence threshold with two measures
- Flexible Confidence Measures: Choose between overall confidence or whale-specific probability
- Smart Sorting: Sort failures by confidence level (high-to-low or low-to-high) for targeted analysis
- Comprehensive PDF Reports: Visual failure analysis with 4 spectrograms per page
- Detailed Metadata Display: Shows species, location, predictions, and confidence scores
- Statistical Summary Output: Console output with accuracy, error rates, and percentage breakdowns

Key Analytical Capabilities:
- Identifies false positives and false negatives with full context
- Calculates accuracy and error rates for both overall and filtered datasets  
- Enables species-specific failure analysis with configurable per-species image limits
- Enables high-confidence failure analysis to identify systematic model errors
- Supports comparative analysis between different confidence measures
- Provides actionable insights through confidence-sorted failure visualization

Usage Examples:
    # Basic failure analysis with complete metrics
    python analyze_failures.py --input TestResults/test_results_binary_testlocation_223D_vallocation_216D.csv --num_images 20
    
    # Species-specific analysis (40 images per species per failure type)
    python analyze_failures.py --input results.csv --species humpback orca --num_images 40
    
    # High-confidence failures analysis (>80% overall confidence)
    python analyze_failures.py -i results.csv --confidence 80 --confidence_measure confidence_percent -n 30
    
    # Species-specific high-confidence analysis
    python analyze_failures.py -i results.csv --species beluga --confidence 75 --confidence_measure confidence_percent -n 20
    
    # Low whale probability failures (systematic false negatives)
    python analyze_failures.py -i results.csv --confidence_measure whale_probability_percent --sort_order asc -n 50
    
    # Comprehensive multi-species analysis with custom output
    python analyze_failures.py -i results.csv --species humpback orca beluga --confidence 75 -n 30 -o Reports/multi_species_failures.pdf
"""

import argparse
import os
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FailureAnalyzer:
    """
    Analyzes test results and creates PDF reports of misclassified spectrograms.
    """
    
    def __init__(self, base_data_dir='/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput_New'):
        self.base_data_dir = base_data_dir
        
    def load_test_results(self, csv_path, confidence_cutoff=None, confidence_measure='confidence_percent', sort_order='desc', species_filter=None):
        """
        Load test results CSV file and identify failures.
        
        Args:
            csv_path (str): Path to test results CSV file
            confidence_cutoff (float): Filter by confidence threshold (0-100)
            confidence_measure (str): Which confidence measure to use ('confidence_percent' or 'whale_probability_percent')
            sort_order (str): Sort order for confidence ('asc' or 'desc')
            species_filter (list): List of species to include (None for all species)
            
        Returns:
            tuple: (false_positives_df, false_negatives_df, classification_summary)
        """
        logger.info(f"Loading test results from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Test results file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} test results")
        
        # Apply species filtering if specified
        if species_filter is not None and len(species_filter) > 0:
            logger.info(f"Applying species filter: {species_filter}")
            original_len = len(df)
            # Convert species filter to lowercase for case-insensitive matching
            species_filter_lower = [s.lower() for s in species_filter]
            df = df[df['species'].str.lower().isin(species_filter_lower)].copy()
            logger.info(f"Filtered from {original_len} to {len(df)} results based on species")
            
            if len(df) == 0:
                logger.warning(f"No data found for species: {species_filter}")
                # Return empty DataFrames but maintain structure
                empty_df = pd.DataFrame()
                empty_summary = {
                    'total_samples': 0, 'true_positives': 0, 'true_negatives': 0,
                    'false_positives': 0, 'false_negatives': 0, 'tp_percentage': 0,
                    'tn_percentage': 0, 'fp_percentage': 0, 'fn_percentage': 0,
                    'accuracy': 0, 'error_rate': 0
                }
                return empty_df, empty_df, empty_summary
        
        # Calculate full classification metrics before filtering
        total_samples = len(df)
        true_positives = len(df[(df['actual_label'] == 1) & (df['predicted_label'] == 1)])
        true_negatives = len(df[(df['actual_label'] == 0) & (df['predicted_label'] == 0)])
        false_positives_count = len(df[(df['actual_label'] == 0) & (df['predicted_label'] == 1)])
        false_negatives_count = len(df[(df['actual_label'] == 1) & (df['predicted_label'] == 0)])
        
        # Calculate percentages
        tp_pct = (true_positives / total_samples) * 100 if total_samples > 0 else 0
        tn_pct = (true_negatives / total_samples) * 100 if total_samples > 0 else 0
        fp_pct = (false_positives_count / total_samples) * 100 if total_samples > 0 else 0
        fn_pct = (false_negatives_count / total_samples) * 100 if total_samples > 0 else 0
        
        # Calculate accuracy and error rate
        accuracy = ((true_positives + true_negatives) / total_samples) * 100 if total_samples > 0 else 0
        error_rate = ((false_positives_count + false_negatives_count) / total_samples) * 100 if total_samples > 0 else 0
        
        # Store classification summary
        classification_summary = {
            'total_samples': total_samples,
            'true_positives': true_positives,
            'true_negatives': true_negatives, 
            'false_positives': false_positives_count,
            'false_negatives': false_negatives_count,
            'tp_percentage': tp_pct,
            'tn_percentage': tn_pct,
            'fp_percentage': fp_pct,
            'fn_percentage': fn_pct,
            'accuracy': accuracy,
            'error_rate': error_rate
        }
        
        # Apply confidence filtering if specified
        if confidence_cutoff is not None:
            logger.info(f"Applying confidence filter: {confidence_measure} > {confidence_cutoff}")
            original_len = len(df)
            df = df[df[confidence_measure] > confidence_cutoff].copy()
            logger.info(f"Filtered from {original_len} to {len(df)} results based on confidence")
            
            # Update classification summary for filtered data
            filtered_total = len(df)
            filtered_tp = len(df[(df['actual_label'] == 1) & (df['predicted_label'] == 1)])
            filtered_tn = len(df[(df['actual_label'] == 0) & (df['predicted_label'] == 0)])
            filtered_fp = len(df[(df['actual_label'] == 0) & (df['predicted_label'] == 1)])
            filtered_fn = len(df[(df['actual_label'] == 1) & (df['predicted_label'] == 0)])
            
            classification_summary['filtered_stats'] = {
                'total_samples': filtered_total,
                'true_positives': filtered_tp,
                'true_negatives': filtered_tn,
                'false_positives': filtered_fp,
                'false_negatives': filtered_fn,
                'tp_percentage': (filtered_tp / filtered_total) * 100 if filtered_total > 0 else 0,
                'tn_percentage': (filtered_tn / filtered_total) * 100 if filtered_total > 0 else 0,
                'fp_percentage': (filtered_fp / filtered_total) * 100 if filtered_total > 0 else 0,
                'fn_percentage': (filtered_fn / filtered_total) * 100 if filtered_total > 0 else 0,
                'accuracy': ((filtered_tp + filtered_tn) / filtered_total) * 100 if filtered_total > 0 else 0,
                'error_rate': ((filtered_fp + filtered_fn) / filtered_total) * 100 if filtered_total > 0 else 0
            }
        
        # Identify misclassifications
        failures = df[df['actual_label'] != df['predicted_label']].copy()
        logger.info(f"Found {len(failures)} total misclassifications")
        
        # Separate false positives and false negatives
        false_positives = failures[
            (failures['actual_label'] == 0) & (failures['predicted_label'] == 1)
        ].copy()
        
        false_negatives = failures[
            (failures['actual_label'] == 1) & (failures['predicted_label'] == 0)
        ].copy()
        
        logger.info(f"False Positives: {len(false_positives)}")
        logger.info(f"False Negatives: {len(false_negatives)}")
        
        # Sort by confidence measure, then by species and location
        ascending = (sort_order == 'asc')
        if len(false_positives) > 0:
            false_positives = false_positives.sort_values(
                [confidence_measure, 'species', 'location'], 
                ascending=[ascending, True, True]
            )
        
        if len(false_negatives) > 0:
            false_negatives = false_negatives.sort_values(
                [confidence_measure, 'species', 'location'], 
                ascending=[ascending, True, True]
            )
        
        logger.info(f"Sorted by {confidence_measure} ({sort_order}), then species and location")
        
        return false_positives, false_negatives, classification_summary
    
    def reconstruct_spectrogram_path(self, image_filename, species, location):
        """
        Reconstruct full path to spectrogram file.
        
        Args:
            image_filename (str): Name of the spectrogram file
            species (str): Species name
            location (str): Location name
            
        Returns:
            str: Full path to spectrogram file
        """
        # Capitalize species name for directory structure
        species_dir = species.capitalize() if species else 'Unknown'
        
        full_path = os.path.join(
            self.base_data_dir,
            species_dir,
            'Processed',
            'SpectrogramsOverlap400ms',
            location,
            image_filename
        )
        
        return full_path
    
    def load_spectrogram(self, spectrogram_path):
        """
        Load spectrogram tensor from file.
        
        Args:
            spectrogram_path (str): Path to spectrogram .pt file
            
        Returns:
            numpy.ndarray: Spectrogram data as 2D array
        """
        try:
            spec_tensor = torch.load(spectrogram_path, map_location='cpu')
            
            # Handle different tensor shapes
            if spec_tensor.ndim == 3:
                # If 3D (C, H, W), take first channel
                spec_array = spec_tensor[0].numpy()
            elif spec_tensor.ndim == 2:
                # If 2D (H, W), use as is
                spec_array = spec_tensor.numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {spec_tensor.shape}")
                
            return spec_array
            
        except Exception as e:
            logger.error(f"Error loading spectrogram {spectrogram_path}: {e}")
            return None
    
    def create_spectrogram_plot(self, ax, spec_array, title, metadata):
        """
        Create a single spectrogram plot.
        
        Args:
            ax: Matplotlib axis object
            spec_array (numpy.ndarray): Spectrogram data
            title (str): Plot title
            metadata (dict): Metadata for the spectrogram
        """
        if spec_array is None:
            ax.text(0.5, 0.5, 'Failed to load\nspectrogram', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=10, fontweight='bold')
            return
        
        # Plot spectrogram with higher quality interpolation
        im = ax.imshow(spec_array, aspect='auto', origin='lower', cmap='viridis', 
                      interpolation='bilinear', rasterized=False)  # Added interpolation for smoother display
        
        # Set title with metadata
        ax.set_title(title, fontsize=12, fontweight='bold')  # Increased from 10
        
        # Add metadata text
        info_text = (f"Species: {metadata.get('species', 'N/A')}\n"
                    f"Location: {metadata.get('location', 'N/A')}\n"
                    f"Actual: {metadata.get('actual_label', 'N/A')} | "
                    f"Predicted: {metadata.get('predicted_label', 'N/A')}")
        
        # Add confidence if available
        if 'confidence_percent' in metadata and pd.notna(metadata['confidence_percent']):
            info_text += f"\nConfidence: {metadata['confidence_percent']:.1f}%"
        
        if 'whale_probability_percent' in metadata and pd.notna(metadata['whale_probability_percent']):
            info_text += f"\nWhale Prob: {metadata['whale_probability_percent']:.1f}%"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,  # Increased from 8
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels with increased font size
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
    
    def create_failure_report(self, false_positives, false_negatives, num_images, output_path, species_filter=None, quality='medium'):
        """
        Create PDF report with failure analysis.
        
        Args:
            false_positives (pd.DataFrame): False positive cases
            false_negatives (pd.DataFrame): False negative cases
            num_images (int): Maximum number of images to include per species per failure type
            output_path (str): Output PDF path
            species_filter (list): List of species to include (affects per-species image counts)
            quality (str): Image quality setting ('low', 'medium', 'high')
        """
        logger.info(f"Creating failure report: {output_path}")
        
        with PdfPages(output_path) as pdf:
            # Create False Positives section
            if len(false_positives) > 0:
                self._create_section(pdf, false_positives, "FALSE POSITIVES", 
                                   "Model predicted whale (1) but actual was no-whale (0)", num_images, species_filter, quality)
            
            # Create False Negatives section  
            if len(false_negatives) > 0:
                self._create_section(pdf, false_negatives, "FALSE NEGATIVES",
                                   "Model predicted no-whale (0) but actual was whale (1)", num_images, species_filter, quality)
        
        logger.info(f"PDF report saved: {output_path}")
    
    def _create_section(self, pdf, failures_df, section_title, section_description, num_images, species_filter=None, quality='medium'):
        """
        Create a section in the PDF for either false positives or false negatives.
        
        Args:
            pdf: PdfPages object
            failures_df (pd.DataFrame): Failure cases
            section_title (str): Section title
            section_description (str): Section description
            num_images (int): Maximum number of images to show per species
            species_filter (list): List of species to include (affects per-species image counts)
            quality (str): Image quality setting ('low', 'medium', 'high')
        """
        logger.info(f"Creating {section_title} section with up to {num_images} images per species")
        
        # Quality settings: (figure_size, figure_dpi, save_dpi)
        quality_settings = {
            'low': ((16, 12), 75, 100),      # Small files, basic quality
            'medium': ((18, 14), 100, 150),  # Balanced quality and size
            'high': ((20, 16), 150, 300)     # High quality, large files
        }
        
        figsize, fig_dpi, save_dpi = quality_settings.get(quality, quality_settings['medium'])
        logger.info(f"Using {quality} quality: figsize={figsize}, dpi={save_dpi}")
        
        # If species filter is specified, limit to num_images per species
        if species_filter is not None and len(species_filter) > 0:
            failures_to_show = pd.DataFrame()
            species_counts = {}
            
            for species in species_filter:
                species_failures = failures_df[failures_df['species'].str.lower() == species.lower()].head(num_images)
                failures_to_show = pd.concat([failures_to_show, species_failures], ignore_index=True)
                species_counts[species] = len(species_failures)
                logger.info(f"  {species}: {len(species_failures)} images")
            
            logger.info(f"Total images for {section_title}: {len(failures_to_show)}")
        else:
            # Original behavior: limit total images
            failures_to_show = failures_df.head(num_images)
            logger.info(f"Total images for {section_title}: {len(failures_to_show)}")
        
        if len(failures_to_show) == 0:
            logger.info(f"No failures to show for {section_title}")
            return
        
        # Group into pages of 4 spectrograms each
        spectrograms_per_page = 4
        
        for page_start in range(0, len(failures_to_show), spectrograms_per_page):
            page_end = min(page_start + spectrograms_per_page, len(failures_to_show))
            page_failures = failures_to_show.iloc[page_start:page_end]
            
            # Create figure for this page with quality-based settings
            fig = plt.figure(figsize=figsize, dpi=fig_dpi)
            
            # Add page title with proper spacing
            page_num = (page_start // spectrograms_per_page) + 1
            total_pages = (len(failures_to_show) - 1) // spectrograms_per_page + 1
            
            fig.suptitle(f"{section_title} - Page {page_num}/{total_pages}\n{section_description}", 
                        fontsize=18, fontweight='bold', y=0.98)  # Increased from 16
            
            # Create subplots (2x2 grid)
            for idx, (_, row) in enumerate(page_failures.iterrows()):
                ax = plt.subplot(2, 2, idx + 1)
                
                # Reconstruct spectrogram path
                spec_path = self.reconstruct_spectrogram_path(
                    row['image_filename'], row['species'], row['location']
                )
                
                # Load spectrogram
                spec_array = self.load_spectrogram(spec_path)
                
                # Create plot
                title = f"{row['image_filename']}"
                metadata = row.to_dict()
                
                self.create_spectrogram_plot(ax, spec_array, title, metadata)
            
            # Hide empty subplots if less than 4 spectrograms on page
            for idx in range(len(page_failures), spectrograms_per_page):
                ax = plt.subplot(2, 2, idx + 1)
                ax.set_visible(False)
            
            # Adjust layout to prevent overlap with title
            plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at top for title
            pdf.savefig(fig, bbox_inches='tight', dpi=save_dpi)  # Use quality-based DPI
            plt.close(fig)
        
        logger.info(f"Completed {section_title} section: {len(failures_to_show)} spectrograms")

def main():
    """
    Main function to run the failure analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze ResNet18 test failures and create PDF report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python analyze_failures.py --input TestResults/test_results_binary_testlocation_223D_vallocation_216D.csv --num_images 20
    
    # Filter by confidence (only show failures with >80% confidence)
    python analyze_failures.py -i results.csv --confidence 80 --confidence_measure confidence_percent -n 30
    
    # Sort by whale probability (low to high)
    python analyze_failures.py -i results.csv --confidence_measure whale_probability_percent --sort_order asc -n 50
    
    # Analyze specific species (40 images per species per failure type = 160 total)
    python analyze_failures.py -i results.csv --species humpback orca -n 40
    
    # High quality analysis (larger files)
    python analyze_failures.py -i results.csv --quality high -n 30
    
    # Low quality for quick analysis (smaller files)
    python analyze_failures.py -i results.csv --quality low -n 50
    
    # Species-specific analysis with confidence filtering
    python analyze_failures.py -i results.csv --species beluga --confidence 75 -n 30
    
    # Custom output location
    python analyze_failures.py -i results.csv -o Diagnostics/Failures/high_confidence_failures.pdf
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to test results CSV file'
    )
    
    parser.add_argument(
        '--num_images', '-n',
        type=int,
        default=20,
        help='Maximum number of images per failure type. If --species specified, this applies per species per failure type (default: 20)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='Failures/failure_analysis.pdf',
        help='Output PDF file path (default: Failures/failure_analysis.pdf)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=None,
        help='Confidence cutoff threshold (0-100). Only include failures with confidence > this value'
    )
    
    parser.add_argument(
        '--confidence_measure',
        choices=['confidence_percent', 'whale_probability_percent'],
        default='confidence_percent',
        help='Which confidence measure to use for filtering and sorting (default: confidence_percent)'
    )
    
    parser.add_argument(
        '--sort_order',
        choices=['asc', 'desc'],
        default='desc',
        help='Sort order for confidence: asc (low to high) or desc (high to low) (default: desc)'
    )
    
    parser.add_argument(
        '--species', '-s',
        nargs='+',
        default=None,
        help='Species to include in analysis (e.g., --species humpback orca). If specified, -n applies per species per failure type.'
    )
    
    parser.add_argument(
        '--quality',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Image quality setting: low (small files), medium (balanced), high (large files) (default: medium)'
    )
    
    parser.add_argument(
        '--base_dir',
        default='/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput_New',
        help='Base directory for spectrogram data (default: /home/radodhia/ssdprivate/NOAAWhalesV2/DataInput_New)'
    )
    
    args = parser.parse_args()
    
    # Validate confidence cutoff
    if args.confidence is not None and (args.confidence < 0 or args.confidence > 100):
        parser.error("Confidence cutoff must be between 0 and 100")
    
    try:
        # Create analyzer
        analyzer = FailureAnalyzer(base_data_dir=args.base_dir)
        
        # Load and analyze test results with confidence filtering
        false_positives, false_negatives, classification_summary = analyzer.load_test_results(
            args.input, 
            confidence_cutoff=args.confidence,
            confidence_measure=args.confidence_measure,
            sort_order=args.sort_order,
            species_filter=args.species
        )
        
        if len(false_positives) == 0 and len(false_negatives) == 0:
            if args.confidence is not None:
                logger.info(f"No failures found with {args.confidence_measure} > {args.confidence}!")
            else:
                logger.info("No failures found in test results!")
            return
        
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate report
        analyzer.create_failure_report(
            false_positives, false_negatives, args.num_images, args.output, args.species, args.quality
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print("FAILURE ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"Input file: {args.input}")
        
        # Overall classification metrics
        print(f"\nOVERALL CLASSIFICATION METRICS:")
        print(f"{'─'*40}")
        print(f"Total samples: {classification_summary['total_samples']:,}")
        print(f"True Positives (TP):  {classification_summary['true_positives']:,} ({classification_summary['tp_percentage']:.1f}%)")
        print(f"True Negatives (TN):  {classification_summary['true_negatives']:,} ({classification_summary['tn_percentage']:.1f}%)")
        print(f"False Positives (FP): {classification_summary['false_positives']:,} ({classification_summary['fp_percentage']:.1f}%)")
        print(f"False Negatives (FN): {classification_summary['false_negatives']:,} ({classification_summary['fn_percentage']:.1f}%)")
        print(f"{'─'*40}")
        print(f"Accuracy: {classification_summary['accuracy']:.1f}%")
        print(f"Error Rate: {classification_summary['error_rate']:.1f}%")
        
        # Filtered metrics if confidence filtering was applied
        if args.confidence is not None and 'filtered_stats' in classification_summary:
            filtered = classification_summary['filtered_stats']
            print(f"\nFILTERED METRICS ({args.confidence_measure} > {args.confidence}):")
            print(f"{'─'*40}")
            print(f"Filtered samples: {filtered['total_samples']:,}")
            print(f"True Positives (TP):  {filtered['true_positives']:,} ({filtered['tp_percentage']:.1f}%)")
            print(f"True Negatives (TN):  {filtered['true_negatives']:,} ({filtered['tn_percentage']:.1f}%)")
            print(f"False Positives (FP): {filtered['false_positives']:,} ({filtered['fp_percentage']:.1f}%)")
            print(f"False Negatives (FN): {filtered['false_negatives']:,} ({filtered['fn_percentage']:.1f}%)")
            print(f"{'─'*40}")
            print(f"Filtered Accuracy: {filtered['accuracy']:.1f}%")
            print(f"Filtered Error Rate: {filtered['error_rate']:.1f}%")
        
        print(f"\nFAILURE ANALYSIS PARAMETERS:")
        print(f"{'─'*40}")
        if args.species is not None:
            print(f"Species filter: {args.species}")
            print(f"Images per species per failure type: {args.num_images}")
            total_expected = len(args.species) * args.num_images * 2  # 2 failure types (FP + FN)
            print(f"Expected max total images: {total_expected}")
        else:
            print(f"Species filter: None (all species)")
            print(f"Max images per failure type: {args.num_images}")
        
        print(f"Image quality: {args.quality}")
        
        if args.confidence is not None:
            print(f"Confidence filter: {args.confidence_measure} > {args.confidence}")
        else:
            print(f"Confidence filter: None")
        print(f"Sort order: {args.confidence_measure} ({args.sort_order})")
        print(f"Output PDF: {args.output}")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
