"""
Memory Bank Edge Case Handler

This module provides sophisticated handling of edge cases in incremental learning
memory banks, including recovery strategies and detailed reporting.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import time


class MemoryBankHandler:
    """Handler for memory bank edge cases and recovery strategies."""
    
    def __init__(self, 
                 memory_bank,
                 min_exemplars_per_class: int = 2,
                 priority_mode: str = 'balanced',  # 'balanced', 'recent', 'performance'
                 log_dir: Optional[str] = None):
        """
        Args:
            memory_bank: The memory bank instance to monitor
            min_exemplars_per_class: Minimum exemplars to maintain per class
            priority_mode: Strategy for prioritizing exemplar retention
            log_dir: Directory for saving edge case logs
        """
        self.memory_bank = memory_bank
        self.min_exemplars_per_class = min_exemplars_per_class
        self.priority_mode = priority_mode
        self.log_dir = log_dir or './memory_bank_logs'
        
        # Edge case tracking
        self.edge_case_history = []
        self.recovery_actions = []
        self.class_statistics = defaultdict(lambda: {
            'total_seen': 0,
            'successfully_stored': 0,
            'extraction_failures': 0,
            'insufficient_samples': 0,
            'overflow_reductions': 0
        })
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
    def handle_insufficient_exemplars(self, class_id: int, available: int, requested: int) -> Dict[str, Any]:
        """Handle case where a class has fewer objects than requested exemplars.
        
        Args:
            class_id: The class with insufficient exemplars
            available: Number of available objects
            requested: Number of requested exemplars
            
        Returns:
            Recovery action dictionary
        """
        edge_case = {
            'type': 'insufficient_exemplars',
            'class_id': class_id,
            'available': available,
            'requested': requested,
            'timestamp': time.time()
        }
        
        # Determine recovery action
        if available == 0:
            action = {
                'action': 'skip_class',
                'reason': 'No objects available',
                'stored': 0
            }
        elif available < self.min_exemplars_per_class:
            action = {
                'action': 'store_all_with_warning',
                'reason': f'Below minimum threshold ({self.min_exemplars_per_class})',
                'stored': available,
                'warning': 'Class may be under-represented in memory'
            }
        else:
            action = {
                'action': 'store_available',
                'reason': 'Storing all available objects',
                'stored': available
            }
        
        # Update statistics
        self.class_statistics[class_id]['insufficient_samples'] += 1
        self.class_statistics[class_id]['total_seen'] += available
        
        # Log edge case
        edge_case['recovery_action'] = action
        self.edge_case_history.append(edge_case)
        self.recovery_actions.append(action)
        
        return action
    
    def handle_memory_overflow(self, current_count: int, max_count: int) -> Dict[str, Any]:
        """Handle memory bank overflow situation.
        
        Args:
            current_count: Current number of exemplars
            max_count: Maximum allowed exemplars
            
        Returns:
            Recovery strategy dictionary
        """
        overflow_amount = current_count - max_count
        overflow_percentage = (overflow_amount / max_count) * 100
        
        edge_case = {
            'type': 'memory_overflow',
            'current_count': current_count,
            'max_count': max_count,
            'overflow_amount': overflow_amount,
            'overflow_percentage': overflow_percentage,
            'timestamp': time.time()
        }
        
        # Determine reduction strategy based on priority mode
        if self.priority_mode == 'balanced':
            strategy = self._balanced_reduction_strategy(overflow_amount)
        elif self.priority_mode == 'recent':
            strategy = self._recent_priority_strategy(overflow_amount)
        elif self.priority_mode == 'performance':
            strategy = self._performance_based_strategy(overflow_amount)
        else:
            strategy = self._balanced_reduction_strategy(overflow_amount)
        
        # Apply the strategy
        reduction_report = self._apply_reduction_strategy(strategy)
        
        edge_case['strategy'] = strategy
        edge_case['reduction_report'] = reduction_report
        self.edge_case_history.append(edge_case)
        
        return reduction_report
    
    def _balanced_reduction_strategy(self, overflow_amount: int) -> Dict[str, Any]:
        """Create a balanced reduction strategy."""
        strategy = {
            'name': 'balanced',
            'description': 'Reduce all classes proportionally while maintaining minimum',
            'min_per_class': self.min_exemplars_per_class,
            'target_reduction': overflow_amount,
            'priority_weights': {}  # Equal priority for all classes
        }
        
        # Calculate per-class weights (all equal for balanced)
        for class_id in self.memory_bank.exemplars.keys():
            strategy['priority_weights'][class_id] = 1.0
            
        return strategy
    
    def _recent_priority_strategy(self, overflow_amount: int) -> Dict[str, Any]:
        """Create a strategy that prioritizes recent classes."""
        strategy = {
            'name': 'recent_priority',
            'description': 'Preserve more exemplars from recent classes',
            'min_per_class': self.min_exemplars_per_class,
            'target_reduction': overflow_amount,
            'priority_weights': {}
        }
        
        # Higher weight for higher class IDs (more recent)
        class_ids = sorted(self.memory_bank.exemplars.keys())
        for i, class_id in enumerate(class_ids):
            # Linear weight increase: older classes get lower weight
            strategy['priority_weights'][class_id] = (i + 1) / len(class_ids)
            
        return strategy
    
    def _performance_based_strategy(self, overflow_amount: int) -> Dict[str, Any]:
        """Create a strategy based on class performance metrics."""
        strategy = {
            'name': 'performance_based',
            'description': 'Preserve exemplars based on extraction success rate',
            'min_per_class': self.min_exemplars_per_class,
            'target_reduction': overflow_amount,
            'priority_weights': {}
        }
        
        # Calculate weights based on success rates
        for class_id in self.memory_bank.exemplars.keys():
            stats = self.class_statistics[class_id]
            if stats['total_seen'] > 0:
                success_rate = stats['successfully_stored'] / stats['total_seen']
            else:
                success_rate = 0.5  # Default weight
            strategy['priority_weights'][class_id] = success_rate
            
        return strategy
    
    def _apply_reduction_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce the bank's actual balanced cap and report measured changes.

        Legacy priority suggestions were previously only reported, never applied.
        Keep the bank's established deterministic policy as the single authority.
        """
        before = {cid: len(xs) for cid, xs in self.memory_bank.exemplars.items()}
        initial = self.memory_bank.get_total_exemplar_count()
        self.memory_bank._reduce_exemplars()
        self.memory_bank._rebuild_cache()
        affected = []
        for cid, count in before.items():
            final = self.memory_bank.get_class_exemplar_count(cid)
            if final < count:
                self.class_statistics[cid]['overflow_reductions'] += 1
                affected.append(dict(class_id=cid, initial=count, final=final,
                                     reduction=count - final))
        final = self.memory_bank.get_total_exemplar_count()
        return dict(strategy_name='balanced_bank_budget',
                    suggested_strategy=strategy['name'], initial_count=initial,
                    target_reduction=strategy['target_reduction'],
                    classes_affected=affected, actual_reduction=initial - final,
                    final_count=final)

    def handle_extraction_failure(self, class_id: int, scene_id: str, 
                                 reason: str = "unknown") -> Dict[str, Any]:
        """Handle failed point cloud extraction.
        
        Args:
            class_id: Class of the failed extraction
            scene_id: Scene where extraction failed
            reason: Reason for failure
            
        Returns:
            Recovery action dictionary
        """
        edge_case = {
            'type': 'extraction_failure',
            'class_id': class_id,
            'scene_id': scene_id,
            'reason': reason,
            'timestamp': time.time()
        }
        
        # Update statistics
        self.class_statistics[class_id]['extraction_failures'] += 1
        
        # Determine recovery action
        failures = self.class_statistics[class_id]['extraction_failures']
        if failures > 10:
            action = {
                'action': 'critical_warning',
                'reason': f'High failure rate for class {class_id}',
                'recommendation': 'Check coordinate system or data integrity'
            }
        else:
            action = {
                'action': 'skip_exemplar',
                'reason': 'Single extraction failure',
                'recommendation': 'Continue with other exemplars'
            }
        
        edge_case['recovery_action'] = action
        self.edge_case_history.append(edge_case)
        
        return action
    
    def generate_report(self, stage_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive edge case report.
        
        Args:
            stage_id: Optional stage identifier for the report
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'stage_id': stage_id,
            'timestamp': time.time(),
            'memory_bank_status': self.memory_bank.get_statistics(),
            'edge_case_summary': self._summarize_edge_cases(),
            'class_health': self._assess_class_health(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _summarize_edge_cases(self) -> Dict[str, Any]:
        """Summarize all edge cases encountered."""
        summary = {
            'total_edge_cases': len(self.edge_case_history),
            'by_type': defaultdict(int),
            'critical_cases': []
        }
        
        for case in self.edge_case_history:
            summary['by_type'][case['type']] += 1
            
            # Identify critical cases
            if case['type'] == 'memory_overflow' and case.get('overflow_percentage', 0) > 50:
                summary['critical_cases'].append({
                    'type': case['type'],
                    'severity': 'high',
                    'details': f"Overflow by {case['overflow_percentage']:.1f}%"
                })
            elif case['type'] == 'insufficient_exemplars' and case.get('available', 0) == 0:
                summary['critical_cases'].append({
                    'type': case['type'],
                    'severity': 'medium',
                    'details': f"Class {case['class_id']} has no exemplars"
                })
        
        return dict(summary)
    
    def _assess_class_health(self) -> Dict[str, Any]:
        """Assess health status of each class."""
        health_report = {
            'healthy_classes': [],
            'warning_classes': [],
            'critical_classes': []
        }
        
        for class_id, stats in self.class_statistics.items():
            health_score = self._calculate_health_score(stats)
            
            class_info = {
                'class_id': class_id,
                'health_score': health_score,
                'stats': stats
            }
            
            if health_score >= 0.8:
                health_report['healthy_classes'].append(class_info)
            elif health_score >= 0.5:
                health_report['warning_classes'].append(class_info)
            else:
                health_report['critical_classes'].append(class_info)
        
        return health_report
    
    def _calculate_health_score(self, stats: Dict[str, Any]) -> float:
        """Calculate health score for a class (0-1)."""
        if stats['total_seen'] == 0:
            return 0.0
        
        # Success rate component
        success_rate = stats['successfully_stored'] / max(1, stats['total_seen'])
        
        # Failure penalty
        failure_penalty = min(1.0, stats['extraction_failures'] * 0.1)
        
        # Insufficient samples penalty
        insufficient_penalty = min(0.5, stats['insufficient_samples'] * 0.1)
        
        # Overflow reduction penalty
        overflow_penalty = min(0.3, stats['overflow_reductions'] * 0.05)
        
        # Calculate final score
        health_score = max(0.0, success_rate - failure_penalty - insufficient_penalty - overflow_penalty)
        
        return health_score
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on observed edge cases."""
        recommendations = []
        
        # Check for systematic issues
        total_failures = sum(s['extraction_failures'] for s in self.class_statistics.values())
        total_insufficient = sum(s['insufficient_samples'] for s in self.class_statistics.values())
        
        if total_failures > 20:
            recommendations.append(
                "High extraction failure rate detected. Check coordinate system alignment and bbox formats."
            )
        
        if total_insufficient > 10:
            recommendations.append(
                f"Many classes have insufficient exemplars. Consider reducing exemplars_per_class "
                f"from {self.memory_bank.exemplars_per_class} to a lower value."
            )
        
        if self.memory_bank.exemplar_count >= self.memory_bank.max_total_exemplars * 0.9:
            recommendations.append(
                f"Memory bank near capacity. Consider increasing max_total_exemplars "
                f"from {self.memory_bank.max_total_exemplars} or using more aggressive reduction."
            )
        
        # Check cache performance
        stats = self.memory_bank.get_statistics()
        if stats.get('cache_hit_rate') is not None and stats['cache_hit_rate'] < 50:
            recommendations.append(
                "Low cache hit rate. Consider increasing cache size or improving cache strategy."
            )
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save report to JSON file.
        
        Args:
            report: Report dictionary to save
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"memory_bank_report_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy(data)
        
        clean_report = clean_for_json(report)
        
        with open(filepath, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        print(f"📊 Report saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of edge cases and health status."""
        print("\n" + "="*60)
        print("📋 MEMORY BANK EDGE CASE SUMMARY")
        print("="*60)
        
        # Edge case counts
        edge_case_summary = self._summarize_edge_cases()
        print(f"\n📊 Edge Cases Encountered: {edge_case_summary['total_edge_cases']}")
        for case_type, count in edge_case_summary['by_type'].items():
            print(f"  - {case_type}: {count}")
        
        # Critical cases
        if edge_case_summary['critical_cases']:
            print(f"\n🚨 Critical Cases: {len(edge_case_summary['critical_cases'])}")
            for case in edge_case_summary['critical_cases'][:5]:
                print(f"  - {case['type']} ({case['severity']}): {case['details']}")
        
        # Class health
        health_report = self._assess_class_health()
        print(f"\n💚 Class Health Status:")
        print(f"  Healthy: {len(health_report['healthy_classes'])} classes")
        print(f"  Warning: {len(health_report['warning_classes'])} classes")
        print(f"  Critical: {len(health_report['critical_classes'])} classes")
        
        if health_report['critical_classes']:
            print(f"\n🔴 Critical Classes:")
            for class_info in health_report['critical_classes'][:5]:
                print(f"  - Class {class_info['class_id']}: health score {class_info['health_score']:.2f}")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*60)
