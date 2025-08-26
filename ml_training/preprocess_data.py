#!/usr/bin/env python3
"""
Advanced Data Preprocessing Pipeline for Swipe Typing ML
Handles data cleaning, normalization, analysis, and export

Features:
- Comprehensive data validation and cleaning
- Advanced trace normalization and feature engineering
- Statistical analysis and visualization
- Data quality assessment and reporting
- Export in multiple formats for different training frameworks
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import interpolate
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MIN_TRACE_POINTS = 3
MAX_TRACE_POINTS = 200
MIN_WORD_LENGTH = 1
MAX_WORD_LENGTH = 50
MIN_SAMPLES_PER_WORD = 2
INTERPOLATION_POINTS = 100

# Quality thresholds
MIN_TRACE_DURATION = 50  # milliseconds
MAX_TRACE_DURATION = 10000  # milliseconds
MIN_TRACE_DISTANCE = 0.01  # normalized coordinates
MAX_VELOCITY = 2.0  # normalized per second


class DataQualityAnalyzer:
    """Analyzes data quality and provides detailed reports."""
    
    def __init__(self):
        self.quality_metrics = {}
        self.issues = []
    
    def analyze_sample(self, record: dict) -> dict:
        """Analyze quality of a single sample."""
        issues = []
        metrics = {}
        
        try:
            trace_points = record['trace_points']
            target_word = record['target_word']
            metadata = record['metadata']
            
            # Basic validation
            if len(trace_points) < MIN_TRACE_POINTS:
                issues.append(f"Too few trace points: {len(trace_points)}")
            
            if len(trace_points) > MAX_TRACE_POINTS:
                issues.append(f"Too many trace points: {len(trace_points)}")
            
            if len(target_word) < MIN_WORD_LENGTH or len(target_word) > MAX_WORD_LENGTH:
                issues.append(f"Invalid word length: {len(target_word)}")
            
            # Coordinate validation
            coords_valid = True
            for i, point in enumerate(trace_points):
                if not (0.0 <= point['x'] <= 1.0 and 0.0 <= point['y'] <= 1.0):
                    issues.append(f"Invalid coordinates at point {i}: ({point['x']}, {point['y']})")
                    coords_valid = False
                    break
                
                if point['t_delta_ms'] < 0 or point['t_delta_ms'] > MAX_TRACE_DURATION:
                    issues.append(f"Invalid timestamp at point {i}: {point['t_delta_ms']}")
                    break
            
            if coords_valid and len(trace_points) >= 2:
                # Calculate trace statistics
                coords = np.array([(p['x'], p['y']) for p in trace_points])
                times = np.array([p['t_delta_ms'] for p in trace_points])
                
                # Duration analysis
                duration = times[-1] - times[0]
                metrics['duration'] = duration
                
                if duration < MIN_TRACE_DURATION:
                    issues.append(f"Trace too short: {duration}ms")
                elif duration > MAX_TRACE_DURATION:
                    issues.append(f"Trace too long: {duration}ms")
                
                # Distance analysis
                distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
                total_distance = np.sum(distances)
                metrics['total_distance'] = total_distance
                
                if total_distance < MIN_TRACE_DISTANCE:
                    issues.append(f"Trace too short spatially: {total_distance:.4f}")
                
                # Velocity analysis
                if duration > 0:
                    velocities = distances / (np.diff(times) + 1e-6) * 1000  # per second
                    max_velocity = np.max(velocities) if len(velocities) > 0 else 0
                    metrics['max_velocity'] = max_velocity
                    
                    if max_velocity > MAX_VELOCITY:
                        issues.append(f"Velocity too high: {max_velocity:.3f}")
                
                # Smoothness analysis (jerk)
                if len(coords) >= 3:
                    accelerations = np.diff(velocities) / (np.diff(times[1:]) + 1e-6) * 1000
                    jerk = np.std(accelerations) if len(accelerations) > 0 else 0
                    metrics['jerk'] = jerk
            
            # Registered keys validation
            if 'registered_keys' in record:
                reg_keys = record['registered_keys']
                if len(reg_keys) == 0:
                    issues.append("No registered keys")
                elif len(reg_keys) > 20:  # Reasonable limit
                    issues.append(f"Too many registered keys: {len(reg_keys)}")
            
            # Metadata validation
            required_fields = ['collection_source', 'screen_width_px', 'screen_height_px']
            for field in required_fields:
                if field not in metadata:
                    issues.append(f"Missing metadata field: {field}")
        
        except Exception as e:
            issues.append(f"Analysis error: {str(e)}")
        
        return {
            'issues': issues,
            'metrics': metrics,
            'valid': len(issues) == 0
        }
    
    def analyze_dataset(self, data_file: str) -> dict:
        """Analyze entire dataset and provide comprehensive report."""
        logger.info(f"Analyzing dataset quality: {data_file}")
        
        total_samples = 0
        valid_samples = 0
        all_issues = []
        all_metrics = []
        word_frequencies = {}
        source_distribution = {}
        
        with open(data_file, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc="Analyzing samples")):
                try:
                    record = json.loads(line)
                    total_samples += 1
                    
                    analysis = self.analyze_sample(record)
                    
                    if analysis['valid']:
                        valid_samples += 1
                    else:
                        all_issues.extend([(line_num, issue) for issue in analysis['issues']])
                    
                    all_metrics.append(analysis['metrics'])
                    
                    # Collect statistics
                    word = record['target_word'].lower()
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
                    
                    source = record['metadata']['collection_source']
                    source_distribution[source] = source_distribution.get(source, 0) + 1
                    
                except Exception as e:
                    all_issues.append((line_num, f"Parse error: {str(e)}"))
                    total_samples += 1
        
        # Analyze metrics
        valid_metrics = [m for m in all_metrics if m]
        metrics_df = pd.DataFrame(valid_metrics) if valid_metrics else pd.DataFrame()
        
        # Generate report
        report = {
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'validity_rate': valid_samples / total_samples if total_samples > 0 else 0,
            'issues': all_issues,
            'word_frequencies': word_frequencies,
            'source_distribution': source_distribution,
            'metrics_summary': metrics_df.describe().to_dict() if not metrics_df.empty else {}
        }
        
        return report


class AdvancedTraceProcessor:
    """Advanced trace preprocessing with interpolation and feature engineering."""
    
    def __init__(self, interpolation_points: int = INTERPOLATION_POINTS):
        self.interpolation_points = interpolation_points
        self.feature_stats = {}
    
    def process_trace(self, trace_points: List[dict], smooth: bool = True) -> np.ndarray:
        """
        Process a single trace with advanced normalization and feature engineering.
        
        Returns array of shape (interpolation_points, num_features) where features are:
        [x, y, t_norm, velocity, acceleration, curvature, pressure_proxy]
        """
        if len(trace_points) < MIN_TRACE_POINTS:
            raise ValueError(f"Trace too short: {len(trace_points)} points")
        
        # Extract basic coordinates and time
        coords = np.array([(p['x'], p['y']) for p in trace_points], dtype=np.float32)
        times = np.array([p['t_delta_ms'] for p in trace_points], dtype=np.float32)
        
        # Normalize timestamps to [0, 1]
        if len(times) > 1 and times[-1] > times[0]:
            times_norm = (times - times[0]) / (times[-1] - times[0])
        else:
            times_norm = np.linspace(0, 1, len(times))
        
        # Apply smoothing if requested
        if smooth and len(coords) > 5:
            coords = self._smooth_coordinates(coords)
        
        # Interpolate to fixed length
        coords_interp, times_interp = self._interpolate_trace(coords, times_norm)
        
        # Calculate advanced features
        features = self._calculate_features(coords_interp, times_interp)
        
        return features
    
    def _smooth_coordinates(self, coords: np.ndarray, window_length: int = 5) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to coordinates."""
        if len(coords) < window_length:
            return coords
        
        # Ensure odd window length
        if window_length % 2 == 0:
            window_length += 1
        
        smoothed = np.zeros_like(coords)
        for dim in range(coords.shape[1]):
            smoothed[:, dim] = savgol_filter(coords[:, dim], window_length, 3)
        
        return smoothed
    
    def _interpolate_trace(self, coords: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate trace to fixed number of points."""
        if len(coords) <= 2:
            # For very short traces, just repeat endpoints
            coords_interp = np.array([coords[0]] * self.interpolation_points)
            times_interp = np.linspace(0, 1, self.interpolation_points)
            return coords_interp, times_interp
        
        # Create interpolation functions
        t_new = np.linspace(times[0], times[-1], self.interpolation_points)
        
        # Interpolate coordinates
        f_x = interpolate.interp1d(times, coords[:, 0], kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
        f_y = interpolate.interp1d(times, coords[:, 1], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        
        coords_interp = np.column_stack([f_x(t_new), f_y(t_new)])
        
        return coords_interp, t_new
    
    def _calculate_features(self, coords: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate comprehensive feature set."""
        features = []
        
        # Basic features: x, y, t_normalized
        x, y = coords[:, 0], coords[:, 1]
        features.extend([x, y, times])
        
        # Velocity features
        if len(coords) > 1:
            dt = np.diff(times)
            dt[dt == 0] = 1e-6  # Avoid division by zero
            
            dx = np.diff(x)
            dy = np.diff(y)
            
            # Velocity magnitude and direction
            velocity = np.sqrt(dx**2 + dy**2) / dt
            velocity = np.concatenate([[0], velocity])  # Pad to original length
            
            # Velocity direction (angle)
            velocity_angle = np.arctan2(dy, dx)
            velocity_angle = np.concatenate([[0], velocity_angle])
            
            features.extend([velocity, velocity_angle])
        else:
            features.extend([np.zeros_like(x), np.zeros_like(x)])
        
        # Acceleration
        if len(coords) > 2:
            acceleration = np.diff(features[3])  # diff of velocity
            acceleration = np.concatenate([[0, 0], acceleration])
            features.append(acceleration)
        else:
            features.append(np.zeros_like(x))
        
        # Distance from start point
        start_distance = np.sqrt((x - x[0])**2 + (y - y[0])**2)
        features.append(start_distance)
        
        # Distance from center of trace
        center_x, center_y = np.mean(x), np.mean(y)
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        features.append(center_distance)
        
        # Cumulative distance
        if len(coords) > 1:
            segment_distances = np.sqrt(dx**2 + dy**2)
            cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])
            features.append(cumulative_distance)
        else:
            features.append(np.zeros_like(x))
        
        # Combine all features
        feature_array = np.column_stack(features)
        
        return feature_array.astype(np.float32)
    
    def fit_normalizer(self, all_features: List[np.ndarray]) -> StandardScaler:
        """Fit normalization parameters on all traces."""
        logger.info("Fitting feature normalizer...")
        
        # Flatten all features
        all_flattened = []
        for features in all_features:
            all_flattened.extend(features.tolist())
        
        feature_matrix = np.array(all_flattened)
        
        # Fit standard scaler
        scaler = StandardScaler()
        scaler.fit(feature_matrix)
        
        # Store statistics
        self.feature_stats = {
            'mean': scaler.mean_,
            'std': scaler.scale_,
            'feature_names': ['x', 'y', 't_norm', 'velocity', 'velocity_angle', 
                            'acceleration', 'start_distance', 'center_distance', 
                            'cumulative_distance']
        }
        
        return scaler


class DataProcessor:
    """Main data processing pipeline."""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.quality_analyzer = DataQualityAnalyzer()
        self.trace_processor = AdvancedTraceProcessor()
        
        self.processed_samples = []
        self.statistics = {}
    
    def process_dataset(self, input_file: str, analyze_quality: bool = True, 
                       create_visualizations: bool = True) -> dict:
        """Process entire dataset with comprehensive pipeline."""
        
        logger.info(f"=== PROCESSING DATASET: {input_file} ===")
        
        # Step 1: Quality analysis
        quality_report = None
        if analyze_quality:
            quality_report = self.quality_analyzer.analyze_dataset(input_file)
            self._save_quality_report(quality_report)
        
        # Step 2: Load and process valid samples
        valid_samples = self._load_and_filter_data(input_file)
        logger.info(f"Loaded {len(valid_samples)} valid samples")
        
        if len(valid_samples) == 0:
            logger.error("No valid samples found!")
            return {}
        
        # Step 3: Process traces with advanced features
        processed_traces = self._process_traces(valid_samples)
        
        # Step 4: Normalize features
        normalizer = self.trace_processor.fit_normalizer(processed_traces)
        normalized_traces = [normalizer.transform(trace) for trace in processed_traces]
        
        # Step 5: Create final dataset
        final_dataset = self._create_final_dataset(valid_samples, normalized_traces)
        
        # Step 6: Generate statistics and visualizations
        stats = self._generate_statistics(final_dataset)
        
        if create_visualizations:
            self._create_visualizations(final_dataset, stats)
        
        # Step 7: Export in multiple formats
        self._export_dataset(final_dataset, normalizer)
        
        return {
            'quality_report': quality_report,
            'statistics': stats,
            'num_samples': len(final_dataset),
            'output_dir': str(self.output_dir)
        }
    
    def _load_and_filter_data(self, input_file: str) -> List[dict]:
        """Load data and filter out invalid samples."""
        valid_samples = []
        
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    record = json.loads(line)
                    analysis = self.quality_analyzer.analyze_sample(record)
                    
                    if analysis['valid']:
                        valid_samples.append(record)
                        
                except Exception as e:
                    logger.warning(f"Error loading sample: {e}")
        
        return valid_samples
    
    def _process_traces(self, samples: List[dict]) -> List[np.ndarray]:
        """Process all traces with advanced feature engineering."""
        processed_traces = []
        
        for sample in tqdm(samples, desc="Processing traces"):
            try:
                features = self.trace_processor.process_trace(sample['trace_points'])
                processed_traces.append(features)
            except Exception as e:
                logger.warning(f"Error processing trace: {e}")
                continue
        
        return processed_traces
    
    def _create_final_dataset(self, samples: List[dict], traces: List[np.ndarray]) -> List[dict]:
        """Create final processed dataset."""
        final_dataset = []
        
        for sample, trace in zip(samples, traces):
            processed_sample = {
                'trace_features': trace.tolist(),  # Convert to serializable format
                'target_word': sample['target_word'].lower(),
                'registered_keys': sample['registered_keys'],
                'metadata': sample['metadata'],
                'trace_id': sample.get('trace_id', f"trace_{len(final_dataset)}")
            }
            final_dataset.append(processed_sample)
        
        return final_dataset
    
    def _generate_statistics(self, dataset: List[dict]) -> dict:
        """Generate comprehensive dataset statistics."""
        logger.info("Generating dataset statistics...")
        
        words = [s['target_word'] for s in dataset]
        sources = [s['metadata']['collection_source'] for s in dataset]
        
        word_counts = pd.Series(words).value_counts()
        source_counts = pd.Series(sources).value_counts()
        
        # Trace feature statistics
        all_traces = [np.array(s['trace_features']) for s in dataset]
        trace_lengths = [len(trace) for trace in all_traces]
        
        stats = {
            'total_samples': len(dataset),
            'unique_words': len(word_counts),
            'word_distribution': word_counts.to_dict(),
            'source_distribution': source_counts.to_dict(),
            'trace_statistics': {
                'mean_length': np.mean(trace_lengths),
                'std_length': np.std(trace_lengths),
                'min_length': np.min(trace_lengths),
                'max_length': np.max(trace_lengths)
            },
            'words_with_few_samples': (word_counts < MIN_SAMPLES_PER_WORD).sum(),
            'feature_statistics': self.trace_processor.feature_stats
        }
        
        return stats
    
    def _create_visualizations(self, dataset: List[dict], stats: dict):
        """Create comprehensive data visualizations."""
        logger.info("Creating visualizations...")
        
        # Word frequency distribution
        plt.figure(figsize=(12, 6))
        word_counts = pd.Series([s['target_word'] for s in dataset]).value_counts()
        
        plt.subplot(1, 2, 1)
        word_counts.head(20).plot(kind='bar')
        plt.title('Top 20 Word Frequencies')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.hist(word_counts.values, bins=50, alpha=0.7)
        plt.title('Word Frequency Distribution')
        plt.xlabel('Frequency')
        plt.ylabel('Number of Words')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'word_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Trace characteristics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample some traces for visualization
        sample_traces = dataset[:min(100, len(dataset))]
        
        # Trace lengths
        trace_lengths = [len(s['trace_features']) for s in sample_traces]
        axes[0, 0].hist(trace_lengths, bins=30, alpha=0.7)
        axes[0, 0].set_title('Trace Length Distribution')
        axes[0, 0].set_xlabel('Number of Points')
        axes[0, 0].set_ylabel('Frequency')
        
        # Source distribution
        sources = [s['metadata']['collection_source'] for s in dataset]
        source_counts = pd.Series(sources).value_counts()
        axes[0, 1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Collection Sources')
        
        # Velocity distribution (sample)
        velocities = []
        for sample in sample_traces[:20]:  # Sample subset
            trace = np.array(sample['trace_features'])
            if trace.shape[1] > 3:  # Has velocity feature
                velocities.extend(trace[:, 3].tolist())  # Velocity is 4th feature
        
        if velocities:
            axes[1, 0].hist(velocities, bins=50, alpha=0.7)
            axes[1, 0].set_title('Velocity Distribution (Sample)')
            axes[1, 0].set_xlabel('Normalized Velocity')
            axes[1, 0].set_ylabel('Frequency')
        
        # Example traces
        axes[1, 1].set_title('Sample Trace Patterns')
        for i, sample in enumerate(sample_traces[:5]):
            trace = np.array(sample['trace_features'])
            x, y = trace[:, 0], trace[:, 1]
            axes[1, 1].plot(x, y, alpha=0.7, label=sample['target_word'])
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Y Coordinate')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trace_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved")
    
    def _save_quality_report(self, report: dict):
        """Save detailed quality analysis report."""
        report_path = self.output_dir / 'quality_report.json'
        
        # Convert non-serializable objects
        serializable_report = report.copy()
        if 'metrics_summary' in report and report['metrics_summary']:
            # Convert DataFrame.describe() result to serializable format
            serializable_report['metrics_summary'] = {
                k: {str(stat): float(val) for stat, val in v.items()} 
                for k, v in report['metrics_summary'].items()
            }
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        # Also create human-readable report
        readable_path = self.output_dir / 'quality_report.txt'
        with open(readable_path, 'w') as f:
            f.write("=== DATA QUALITY REPORT ===\n\n")
            f.write(f"Total Samples: {report['total_samples']}\n")
            f.write(f"Valid Samples: {report['valid_samples']}\n")
            f.write(f"Validity Rate: {report['validity_rate']:.2%}\n\n")
            
            f.write(f"Source Distribution:\n")
            for source, count in report['source_distribution'].items():
                f.write(f"  {source}: {count}\n")
            
            f.write(f"\nTop Words:\n")
            word_items = sorted(report['word_frequencies'].items(), 
                              key=lambda x: x[1], reverse=True)
            for word, count in word_items[:20]:
                f.write(f"  {word}: {count}\n")
            
            if report['issues']:
                f.write(f"\nQuality Issues ({len(report['issues'])}):\n")
                issue_summary = {}
                for line_num, issue in report['issues'][:100]:  # First 100 issues
                    issue_type = issue.split(':')[0] if ':' in issue else issue
                    issue_summary[issue_type] = issue_summary.get(issue_type, 0) + 1
                
                for issue_type, count in sorted(issue_summary.items(), 
                                               key=lambda x: x[1], reverse=True):
                    f.write(f"  {issue_type}: {count} occurrences\n")
        
        logger.info(f"Quality report saved to {report_path}")
    
    def _export_dataset(self, dataset: List[dict], normalizer):
        """Export dataset in multiple formats."""
        logger.info("Exporting processed dataset...")
        
        # Export as NDJSON for TensorFlow training
        ndjson_path = self.output_dir / 'processed_training_data.ndjson'
        with open(ndjson_path, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')
        
        # Export normalizer parameters
        normalizer_path = self.output_dir / 'feature_normalizer.json'
        with open(normalizer_path, 'w') as f:
            params = {
                'mean': normalizer.mean_.tolist(),
                'scale': normalizer.scale_.tolist(),
                'feature_names': self.trace_processor.feature_stats['feature_names']
            }
            json.dump(params, f, indent=2)
        
        # Export summary statistics
        stats_path = self.output_dir / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            stats = self._generate_statistics(dataset)
            # Make serializable
            serializable_stats = {}
            for key, value in stats.items():
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    serializable_stats[key] = value
                elif hasattr(value, 'tolist'):
                    serializable_stats[key] = value.tolist()
                else:
                    serializable_stats[key] = str(value)
            json.dump(serializable_stats, f, indent=2)
        
        # Create CSV summary for external analysis
        csv_path = self.output_dir / 'dataset_summary.csv'
        summary_data = []
        for sample in dataset:
            trace_array = np.array(sample['trace_features'])
            summary_data.append({
                'trace_id': sample['trace_id'],
                'target_word': sample['target_word'],
                'trace_length': len(trace_array),
                'collection_source': sample['metadata']['collection_source'],
                'mean_x': np.mean(trace_array[:, 0]),
                'mean_y': np.mean(trace_array[:, 1]),
                'total_distance': np.sum(trace_array[:, -1]) if trace_array.shape[1] > 8 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Dataset exported to {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - processed_training_data.ndjson ({len(dataset)} samples)")
        logger.info(f"  - feature_normalizer.json")
        logger.info(f"  - dataset_statistics.json")
        logger.info(f"  - dataset_summary.csv")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Advanced swipe data preprocessing pipeline')
    parser.add_argument('input_file', help='Path to input NDJSON file')
    parser.add_argument('--output-dir', default='processed_data', help='Output directory')
    parser.add_argument('--skip-quality', action='store_true', help='Skip quality analysis')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    parser.add_argument('--interpolation-points', type=int, default=INTERPOLATION_POINTS,
                       help='Number of interpolation points for traces')
    args = parser.parse_args()
    
    # Create processor
    processor = DataProcessor(args.output_dir)
    processor.trace_processor.interpolation_points = args.interpolation_points
    
    # Process dataset
    results = processor.process_dataset(
        args.input_file,
        analyze_quality=not args.skip_quality,
        create_visualizations=not args.no_viz
    )
    
    logger.info("=== PREPROCESSING COMPLETE ===")
    if results:
        logger.info(f"Processed {results['num_samples']} samples")
        logger.info(f"Output saved to: {results['output_dir']}")
        
        if results['quality_report']:
            validity_rate = results['quality_report']['validity_rate']
            logger.info(f"Data validity rate: {validity_rate:.2%}")


if __name__ == '__main__':
    main()