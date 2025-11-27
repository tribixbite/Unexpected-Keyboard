#!/usr/bin/env python3
"""
Model evaluation and validation metrics for swipe typing ML model.
Provides comprehensive performance analysis for trained TensorFlow Lite models.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import time
from pathlib import Path

class SwipeModelEvaluator:
    """Comprehensive evaluation system for swipe typing models."""
    
    def __init__(self, model_path: str, test_data_path: str):
        """
        Initialize evaluator with model and test data.
        
        Args:
            model_path: Path to TFLite model file
            test_data_path: Path to test data (NDJSON format)
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.vocabulary = {}
        self.reverse_vocab = {}
        self.test_data = []
        self.results = {}
        
    def load_model(self):
        """Load TensorFlow Lite model."""
        print(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded successfully")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
    def load_vocabulary(self, vocab_path: str = "vocabulary.json"):
        """Load vocabulary mapping."""
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocabulary.items()}
            print(f"Loaded vocabulary with {len(self.vocabulary)} words")
        else:
            print(f"Warning: Vocabulary file not found: {vocab_path}")
            
    def load_test_data(self):
        """Load test data from NDJSON file."""
        print(f"Loading test data from {self.test_data_path}")
        
        if not os.path.exists(self.test_data_path):
            raise FileNotFoundError(f"Test data file not found: {self.test_data_path}")
            
        self.test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    self.test_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    
        print(f"Loaded {len(self.test_data)} test samples")
        
    def preprocess_sample(self, sample: Dict) -> Tuple[np.ndarray, int]:
        """
        Preprocess a single sample for model input.
        
        Args:
            sample: Sample data from NDJSON
            
        Returns:
            Tuple of (features, target_label)
        """
        # Extract swipe trace points
        trace_points = sample.get('trace_points', [])
        target_word = sample.get('target_word', '')
        
        # Convert trace to feature vector (same as training preprocessing)
        max_points = 50  # Match training configuration
        features = np.zeros((max_points, 4))  # x, y, velocity_x, velocity_y
        
        for i, point in enumerate(trace_points[:max_points]):
            features[i, 0] = point.get('x', 0.0)
            features[i, 1] = point.get('y', 0.0)
            features[i, 2] = point.get('velocity_x', 0.0)
            features[i, 3] = point.get('velocity_y', 0.0)
            
        # Get target label
        target_label = self.vocabulary.get(target_word.lower(), 0)  # 0 for unknown
        
        return features.astype(np.float32), target_label
        
    def predict_sample(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction for a single sample.
        
        Args:
            features: Preprocessed features
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        # Reshape for batch dimension
        input_data = np.expand_dims(features, axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output_data[0]
        
        # Get prediction
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence, probabilities
        
    def evaluate_accuracy(self) -> Dict:
        """Evaluate model accuracy metrics."""
        print("Evaluating model accuracy...")
        
        y_true = []
        y_pred = []
        confidences = []
        prediction_times = []
        
        for i, sample in enumerate(self.test_data):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(self.test_data)}")
                
            # Preprocess sample
            features, target_label = self.preprocess_sample(sample)
            
            # Make prediction with timing
            start_time = time.time()
            predicted_class, confidence, _ = self.predict_sample(features)
            prediction_time = (time.time() - start_time) * 1000  # ms
            
            y_true.append(target_label)
            y_pred.append(predicted_class)
            confidences.append(confidence)
            prediction_times.append(prediction_time)
            
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        avg_confidence = np.mean(confidences)
        avg_prediction_time = np.mean(prediction_times)
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_confidence': float(avg_confidence),
            'avg_prediction_time_ms': float(avg_prediction_time),
            'total_samples': len(self.test_data)
        }
        
        print(f"\nAccuracy Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Avg Confidence: {avg_confidence:.4f}")
        print(f"  Avg Prediction Time: {avg_prediction_time:.2f}ms")
        
        self.results['accuracy'] = results
        return results
        
    def evaluate_top_k_accuracy(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate top-k accuracy."""
        print(f"Evaluating top-k accuracy for k={k_values}")
        
        top_k_accuracies = {k: 0 for k in k_values}
        
        for i, sample in enumerate(self.test_data):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(self.test_data)}")
                
            features, target_label = self.preprocess_sample(sample)
            _, _, probabilities = self.predict_sample(features)
            
            # Get top-k predictions
            top_k_indices = np.argsort(probabilities)[::-1]
            
            # Check if target is in top-k for each k
            for k in k_values:
                if target_label in top_k_indices[:k]:
                    top_k_accuracies[k] += 1
                    
        # Convert to percentages
        total_samples = len(self.test_data)
        top_k_results = {
            k: float(count / total_samples) for k, count in top_k_accuracies.items()
        }
        
        print(f"\nTop-K Accuracy:")
        for k, acc in top_k_results.items():
            print(f"  Top-{k}: {acc:.4f}")
            
        self.results['top_k_accuracy'] = top_k_results
        return top_k_results
        
    def analyze_confusion_matrix(self, top_words: int = 20) -> Dict:
        """Generate and analyze confusion matrix for top words."""
        print(f"Analyzing confusion matrix for top {top_words} words...")
        
        # Get most frequent words in test data
        word_counts = {}
        for sample in self.test_data:
            word = sample.get('target_word', '').lower()
            word_counts[word] = word_counts.get(word, 0) + 1
            
        top_words_list = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_words]
        top_word_labels = [self.vocabulary.get(word, 0) for word, _ in top_words_list]
        
        # Filter test data to only include top words
        filtered_true = []
        filtered_pred = []
        
        for sample in self.test_data:
            features, target_label = self.preprocess_sample(sample)
            
            if target_label in top_word_labels:
                predicted_class, _, _ = self.predict_sample(features)
                filtered_true.append(target_label)
                filtered_pred.append(predicted_class)
                
        if len(filtered_true) == 0:
            print("Warning: No samples found for top words analysis")
            return {}
            
        # Generate confusion matrix
        cm = confusion_matrix(filtered_true, filtered_pred, labels=top_word_labels)
        
        # Create labels for visualization
        word_labels = [self.reverse_vocab.get(label, f'id_{label}') for label in top_word_labels]
        
        # Save confusion matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=word_labels, yticklabels=word_labels)
        plt.title(f'Confusion Matrix - Top {top_words} Words')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate per-class metrics
        report = classification_report(filtered_true, filtered_pred, 
                                     labels=top_word_labels,
                                     target_names=word_labels,
                                     output_dict=True)
        
        results = {
            'confusion_matrix': cm.tolist(),
            'word_labels': word_labels,
            'classification_report': report
        }
        
        print(f"Confusion matrix saved to confusion_matrix.png")
        self.results['confusion_analysis'] = results
        return results
        
    def analyze_performance_by_word_length(self) -> Dict:
        """Analyze performance by target word length."""
        print("Analyzing performance by word length...")
        
        length_stats = {}
        
        for sample in self.test_data:
            target_word = sample.get('target_word', '')
            word_length = len(target_word)
            
            if word_length not in length_stats:
                length_stats[word_length] = {'correct': 0, 'total': 0}
                
            features, target_label = self.preprocess_sample(sample)
            predicted_class, _, _ = self.predict_sample(features)
            
            length_stats[word_length]['total'] += 1
            if predicted_class == target_label:
                length_stats[word_length]['correct'] += 1
                
        # Calculate accuracy by length
        length_accuracy = {}
        for length, stats in length_stats.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                length_accuracy[length] = {
                    'accuracy': float(accuracy),
                    'sample_count': stats['total']
                }
                
        # Sort by length
        sorted_lengths = sorted(length_accuracy.keys())
        
        print(f"\nAccuracy by Word Length:")
        for length in sorted_lengths:
            stats = length_accuracy[length]
            print(f"  Length {length}: {stats['accuracy']:.4f} ({stats['sample_count']} samples)")
            
        self.results['length_analysis'] = length_accuracy
        return length_accuracy
        
    def generate_performance_report(self, output_dir: str = "evaluation_results"):
        """Generate comprehensive performance report."""
        print(f"Generating performance report in {output_dir}/")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary report
        report = {
            'model_info': {
                'model_path': self.model_path,
                'test_data_path': self.test_data_path,
                'vocabulary_size': len(self.vocabulary),
                'test_samples': len(self.test_data)
            },
            'evaluation_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save JSON report
        with open(f"{output_dir}/evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate markdown report
        self._generate_markdown_report(report, f"{output_dir}/evaluation_report.md")
        
        # Create performance plots
        self._create_performance_plots(output_dir)
        
        print(f"Performance report saved to {output_dir}/")
        
    def _generate_markdown_report(self, report: Dict, output_path: str):
        """Generate markdown performance report."""
        with open(output_path, 'w') as f:
            f.write("# Swipe Typing Model Evaluation Report\n\n")
            
            # Model info
            f.write("## Model Information\n")
            f.write(f"- **Model Path**: {report['model_info']['model_path']}\n")
            f.write(f"- **Test Data**: {report['model_info']['test_data_path']}\n")
            f.write(f"- **Vocabulary Size**: {report['model_info']['vocabulary_size']:,}\n")
            f.write(f"- **Test Samples**: {report['model_info']['test_samples']:,}\n")
            f.write(f"- **Evaluation Date**: {report['timestamp']}\n\n")
            
            # Accuracy metrics
            if 'accuracy' in self.results:
                acc = self.results['accuracy']
                f.write("## Overall Performance\n")
                f.write(f"- **Accuracy**: {acc['accuracy']:.4f}\n")
                f.write(f"- **Precision**: {acc['precision']:.4f}\n")
                f.write(f"- **Recall**: {acc['recall']:.4f}\n")
                f.write(f"- **F1 Score**: {acc['f1_score']:.4f}\n")
                f.write(f"- **Average Confidence**: {acc['avg_confidence']:.4f}\n")
                f.write(f"- **Average Prediction Time**: {acc['avg_prediction_time_ms']:.2f}ms\n\n")
                
            # Top-k accuracy
            if 'top_k_accuracy' in self.results:
                f.write("## Top-K Accuracy\n")
                for k, acc in self.results['top_k_accuracy'].items():
                    f.write(f"- **Top-{k}**: {acc:.4f}\n")
                f.write("\n")
                
            # Word length analysis
            if 'length_analysis' in self.results:
                f.write("## Performance by Word Length\n")
                f.write("| Length | Accuracy | Sample Count |\n")
                f.write("|--------|----------|-------------|\n")
                for length, stats in sorted(self.results['length_analysis'].items()):
                    f.write(f"| {length} | {stats['accuracy']:.4f} | {stats['sample_count']} |\n")
                f.write("\n")
                
    def _create_performance_plots(self, output_dir: str):
        """Create performance visualization plots."""
        # Top-k accuracy plot
        if 'top_k_accuracy' in self.results:
            plt.figure(figsize=(10, 6))
            k_values = list(self.results['top_k_accuracy'].keys())
            accuracies = list(self.results['top_k_accuracy'].values())
            
            plt.bar(k_values, accuracies, color='skyblue', alpha=0.7)
            plt.xlabel('K Value')
            plt.ylabel('Accuracy')
            plt.title('Top-K Accuracy Performance')
            plt.ylim(0, 1.0)
            
            # Add value labels on bars
            for k, acc in zip(k_values, accuracies):
                plt.text(k, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(f'{output_dir}/top_k_accuracy.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Word length performance plot
        if 'length_analysis' in self.results:
            lengths = sorted(self.results['length_analysis'].keys())
            accuracies = [self.results['length_analysis'][l]['accuracy'] for l in lengths]
            sample_counts = [self.results['length_analysis'][l]['sample_count'] for l in lengths]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Accuracy by length
            ax1.plot(lengths, accuracies, marker='o', linewidth=2, markersize=6)
            ax1.set_xlabel('Word Length')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy by Word Length')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.0)
            
            # Sample count by length
            ax2.bar(lengths, sample_counts, alpha=0.7, color='orange')
            ax2.set_xlabel('Word Length')
            ax2.set_ylabel('Sample Count')
            ax2.set_title('Test Samples by Word Length')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/length_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate swipe typing ML model')
    parser.add_argument('--model', required=True, help='Path to TFLite model file')
    parser.add_argument('--test-data', required=True, help='Path to test data (NDJSON)')
    parser.add_argument('--vocabulary', default='vocabulary.json', help='Path to vocabulary file')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--top-words', type=int, default=20, help='Number of top words for confusion matrix')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SwipeModelEvaluator(args.model, args.test_data)
    
    try:
        # Load components
        evaluator.load_model()
        evaluator.load_vocabulary(args.vocabulary)
        evaluator.load_test_data()
        
        # Run evaluations
        print("\n" + "="*50)
        print("STARTING MODEL EVALUATION")
        print("="*50)
        
        evaluator.evaluate_accuracy()
        evaluator.evaluate_top_k_accuracy()
        evaluator.analyze_confusion_matrix(args.top_words)
        evaluator.analyze_performance_by_word_length()
        
        # Generate report
        evaluator.generate_performance_report(args.output_dir)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()