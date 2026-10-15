"""
Incremental Learning Logger

Comprehensive logging system for tracking edge cases, performance metrics,
and system health during incremental learning experiments.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np


class IncrementalLogger:
    """Logger for incremental learning experiments with edge case tracking."""
    
    def __init__(self, log_dir: str = './incremental_logs', experiment_name: str = 'experiment'):
        """
        Args:
            log_dir: Directory for saving logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
        # Create log directory structure
        self.base_dir = os.path.join(log_dir, experiment_name)
        self.stage_logs_dir = os.path.join(self.base_dir, 'stages')
        self.edge_cases_dir = os.path.join(self.base_dir, 'edge_cases')
        self.metrics_dir = os.path.join(self.base_dir, 'metrics')
        
        for dir_path in [self.base_dir, self.stage_logs_dir, self.edge_cases_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize tracking structures
        self.stage_logs = []
        self.edge_cases = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        self.memory_bank_history = []
        self.training_events = []
        
        # Current stage info
        self.current_stage = None
        self.stage_start_time = None
        
    def start_stage(self, stage_id: int, stage_info: Dict[str, Any]):
        """Start logging for a new stage.
        
        Args:
            stage_id: Stage identifier
            stage_info: Stage configuration and metadata
        """
        self.current_stage = stage_id
        self.stage_start_time = time.time()
        
        stage_log = {
            'stage_id': stage_id,
            'start_time': self.stage_start_time,
            'stage_info': stage_info,
            'events': []
        }
        
        self.stage_logs.append(stage_log)
        
        # Log stage start
        self.log_event('stage_start', {
            'stage_id': stage_id,
            'stage_name': stage_info.get('stage_name', f'Stage {stage_id}'),
            'classes': stage_info.get('class_indices', []),
            'epochs': stage_info.get('epochs', 0)
        })
        
        print(f"\n{'='*60}")
        print(f"📝 INCREMENTAL LOGGER: Stage {stage_id} Started")
        print(f"   Stage Name: {stage_info.get('stage_name', 'Unknown')}")
        print(f"   Classes: {stage_info.get('class_indices', [])}")
        print(f"{'='*60}\n")
    
    def end_stage(self, stage_id: int, summary: Optional[Dict[str, Any]] = None):
        """End logging for current stage.
        
        Args:
            stage_id: Stage identifier
            summary: Optional stage summary
        """
        if self.current_stage != stage_id:
            print(f"⚠️  Warning: Ending stage {stage_id} but current stage is {self.current_stage}")
        
        stage_duration = time.time() - self.stage_start_time
        
        # Find and update stage log
        for stage_log in self.stage_logs:
            if stage_log['stage_id'] == stage_id:
                stage_log['end_time'] = time.time()
                stage_log['duration'] = stage_duration
                stage_log['summary'] = summary or {}
                break
        
        # Log stage end
        self.log_event('stage_end', {
            'stage_id': stage_id,
            'duration': stage_duration,
            'summary': summary
        })
        
        # Save stage log
        self._save_stage_log(stage_id)
        
        self.current_stage = None
        self.stage_start_time = None
    
    def log_edge_case(self, case_type: str, details: Dict[str, Any]):
        """Log an edge case occurrence.
        
        Args:
            case_type: Type of edge case (e.g., 'insufficient_exemplars', 'memory_overflow')
            details: Detailed information about the edge case
        """
        edge_case = {
            'type': case_type,
            'stage_id': self.current_stage,
            'timestamp': time.time(),
            'details': details
        }
        
        self.edge_cases[case_type].append(edge_case)
        
        # Also add to current stage events if applicable
        if self.current_stage is not None:
            self.log_event(f'edge_case_{case_type}', details)
        
        # Log critical edge cases immediately
        if case_type in ['memory_overflow', 'extraction_failure', 'empty_class']:
            self._save_edge_case(edge_case)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a general training event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'type': event_type,
            'stage_id': self.current_stage,
            'timestamp': time.time(),
            'data': data
        }
        
        self.training_events.append(event)
        
        # Add to current stage log
        if self.current_stage is not None:
            for stage_log in self.stage_logs:
                if stage_log['stage_id'] == self.current_stage:
                    stage_log['events'].append(event)
                    break
    
    def log_memory_bank_state(self, memory_bank_stats: Dict[str, Any]):
        """Log memory bank state and statistics.
        
        Args:
            memory_bank_stats: Memory bank statistics dictionary
        """
        state = {
            'stage_id': self.current_stage,
            'timestamp': time.time(),
            'stats': memory_bank_stats
        }
        
        self.memory_bank_history.append(state)
        
        # Check for edge cases in memory bank
        if memory_bank_stats.get('edge_cases'):
            edge_info = memory_bank_stats['edge_cases']
            
            if edge_info.get('empty_classes'):
                self.log_edge_case('empty_classes', {
                    'classes': edge_info['empty_classes'],
                    'count': len(edge_info['empty_classes'])
                })
            
            if edge_info.get('is_at_max_capacity'):
                self.log_edge_case('memory_at_capacity', {
                    'total_exemplars': memory_bank_stats['total_exemplars'],
                    'utilization': memory_bank_stats['memory_utilization']
                })
    
    def log_performance_metric(self, metric_name: str, value: float, 
                              stage_id: Optional[int] = None):
        """Log a performance metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'mAP', 'loss')
            value: Metric value
            stage_id: Optional stage identifier
        """
        metric = {
            'name': metric_name,
            'value': value,
            'stage_id': stage_id or self.current_stage,
            'timestamp': time.time()
        }
        
        self.performance_metrics[metric_name].append(metric)
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate comprehensive experiment summary.
        
        Returns:
            Summary dictionary with all experiment information
        """
        total_duration = time.time() - self.start_time
        
        # Aggregate edge case statistics
        edge_case_summary = {}
        for case_type, cases in self.edge_cases.items():
            edge_case_summary[case_type] = {
                'count': len(cases),
                'stages_affected': list(set(c['stage_id'] for c in cases if c['stage_id'] is not None))
            }
        
        # Aggregate performance metrics
        metric_summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                metric_values = [v['value'] for v in values]
                metric_summary[metric_name] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'final': metric_values[-1] if metric_values else None
                }
        
        # Memory bank evolution
        memory_evolution = []
        for state in self.memory_bank_history:
            memory_evolution.append({
                'stage_id': state['stage_id'],
                'total_exemplars': state['stats'].get('total_exemplars', 0),
                'memory_utilization': state['stats'].get('memory_utilization', 0),
                'cache_hit_rate': state['stats'].get('cache_hit_rate', 0)
            })
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_duration': total_duration,
            'total_stages': len(self.stage_logs),
            'total_events': len(self.training_events),
            'edge_case_summary': edge_case_summary,
            'performance_summary': metric_summary,
            'memory_bank_evolution': memory_evolution,
            'stage_summaries': [self._summarize_stage(s) for s in self.stage_logs]
        }
        
        return summary
    
    def _summarize_stage(self, stage_log: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for a single stage."""
        stage_events = stage_log.get('events', [])
        edge_case_events = [e for e in stage_events if 'edge_case' in e['type']]
        
        return {
            'stage_id': stage_log['stage_id'],
            'duration': stage_log.get('duration', 0),
            'classes': stage_log['stage_info'].get('class_indices', []),
            'total_events': len(stage_events),
            'edge_cases': len(edge_case_events),
            'summary': stage_log.get('summary', {})
        }
    
    def _save_stage_log(self, stage_id: int):
        """Save log for a specific stage."""
        for stage_log in self.stage_logs:
            if stage_log['stage_id'] == stage_id:
                filename = f"stage_{stage_id}_log.json"
                filepath = os.path.join(self.stage_logs_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(stage_log, f, indent=2, default=str)
                
                print(f"📁 Stage {stage_id} log saved to: {filepath}")
                break
    
    def _save_edge_case(self, edge_case: Dict[str, Any]):
        """Save critical edge case immediately."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"edge_case_{edge_case['type']}_{timestamp}.json"
        filepath = os.path.join(self.edge_cases_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(edge_case, f, indent=2, default=str)
    
    def save_experiment_summary(self):
        """Save complete experiment summary."""
        summary = self.generate_experiment_summary()
        
        # Main summary file
        summary_file = os.path.join(self.base_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Edge cases compilation
        edge_cases_file = os.path.join(self.edge_cases_dir, 'all_edge_cases.json')
        with open(edge_cases_file, 'w') as f:
            json.dump(dict(self.edge_cases), f, indent=2, default=str)
        
        # Performance metrics
        metrics_file = os.path.join(self.metrics_dir, 'performance_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.performance_metrics), f, indent=2, default=str)
        
        # Memory bank history
        memory_file = os.path.join(self.metrics_dir, 'memory_bank_history.json')
        with open(memory_file, 'w') as f:
            json.dump(self.memory_bank_history, f, indent=2, default=str)
        
        print(f"\n📊 Experiment summary saved to: {self.base_dir}")
        
        return summary_file
    
    def print_summary(self):
        """Print experiment summary to console."""
        summary = self.generate_experiment_summary()
        
        print("\n" + "="*70)
        print("📈 INCREMENTAL LEARNING EXPERIMENT SUMMARY")
        print("="*70)
        
        print(f"\n📋 Experiment: {summary['experiment_name']}")
        print(f"   Duration: {summary['total_duration']:.2f} seconds")
        print(f"   Stages: {summary['total_stages']}")
        print(f"   Total Events: {summary['total_events']}")
        
        # Edge cases
        if summary['edge_case_summary']:
            print(f"\n⚠️  Edge Cases Encountered:")
            for case_type, info in summary['edge_case_summary'].items():
                print(f"   - {case_type}: {info['count']} occurrences in stages {info['stages_affected']}")
        
        # Performance metrics
        if summary['performance_summary']:
            print(f"\n📊 Performance Metrics:")
            for metric, stats in summary['performance_summary'].items():
                print(f"   {metric}:")
                print(f"     Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"     Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                if stats['final'] is not None:
                    print(f"     Final: {stats['final']:.4f}")
        
        # Memory bank evolution
        if summary['memory_bank_evolution']:
            print(f"\n🧠 Memory Bank Evolution:")
            for state in summary['memory_bank_evolution']:
                print(f"   Stage {state['stage_id']}: {state['total_exemplars']} exemplars "
                      f"({state['memory_utilization']:.1f}% utilization, "
                      f"{state['cache_hit_rate']:.1f}% cache hits)")
        
        # Stage summaries
        print(f"\n📝 Stage Summaries:")
        for stage_summary in summary['stage_summaries']:
            print(f"   Stage {stage_summary['stage_id']}:")
            print(f"     Duration: {stage_summary['duration']:.2f}s")
            print(f"     Classes: {stage_summary['classes']}")
            print(f"     Events: {stage_summary['total_events']}")
            print(f"     Edge Cases: {stage_summary['edge_cases']}")
        
        print("="*70)


class EdgeCaseTracker:
    """Specialized tracker for edge case patterns and trends."""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.critical_thresholds = {
            'extraction_failure_rate': 0.2,
            'insufficient_exemplar_rate': 0.3,
            'memory_overflow_frequency': 2,
            'empty_class_threshold': 3
        }
        
    def analyze_patterns(self, edge_cases: Dict[str, List]) -> Dict[str, Any]:
        """Analyze edge case patterns for systemic issues.
        
        Args:
            edge_cases: Dictionary of edge cases by type
            
        Returns:
            Pattern analysis results
        """
        analysis = {
            'systemic_issues': [],
            'recommendations': [],
            'severity': 'low'
        }
        
        # Check extraction failure patterns
        if 'extraction_failure' in edge_cases:
            failure_rate = len(edge_cases['extraction_failure']) / max(1, sum(len(v) for v in edge_cases.values()))
            if failure_rate > self.critical_thresholds['extraction_failure_rate']:
                analysis['systemic_issues'].append('High extraction failure rate')
                analysis['recommendations'].append('Check coordinate system alignment')
                analysis['severity'] = 'high'
        
        # Check insufficient exemplar patterns
        if 'insufficient_exemplars' in edge_cases:
            insufficient_rate = len(edge_cases['insufficient_exemplars']) / max(1, len(edge_cases))
            if insufficient_rate > self.critical_thresholds['insufficient_exemplar_rate']:
                analysis['systemic_issues'].append('Many classes with insufficient exemplars')
                analysis['recommendations'].append('Reduce exemplars_per_class setting')
                analysis['severity'] = max(analysis['severity'], 'medium')
        
        # Check memory overflow frequency
        if 'memory_overflow' in edge_cases:
            if len(edge_cases['memory_overflow']) >= self.critical_thresholds['memory_overflow_frequency']:
                analysis['systemic_issues'].append('Frequent memory overflow')
                analysis['recommendations'].append('Increase max_total_exemplars or use more aggressive reduction')
                analysis['severity'] = 'high'
        
        # Check empty classes
        if 'empty_classes' in edge_cases:
            total_empty = sum(len(case['details'].get('classes', [])) for case in edge_cases['empty_classes'])
            if total_empty >= self.critical_thresholds['empty_class_threshold']:
                analysis['systemic_issues'].append(f'{total_empty} classes with no training data')
                analysis['recommendations'].append('Review class distribution in dataset')
                analysis['severity'] = 'high'
        
        return analysis