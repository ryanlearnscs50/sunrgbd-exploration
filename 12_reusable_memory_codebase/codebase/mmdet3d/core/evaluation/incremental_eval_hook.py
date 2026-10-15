"""
Incremental Learning Evaluation Hook

This hook temporarily removes the seen_classes_mask during evaluation
to allow the model to make predictions for all classes, then restores
the mask for continued training.
"""

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class IncrementalEvalHook(Hook):
    """Custom evaluation hook for incremental learning.
    
    Temporarily removes seen_classes_mask during evaluation to allow
    predictions for all classes, then restores it for training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_mask = None
    
    def before_val_epoch(self, runner):
        """Remove seen_classes_mask before validation."""
        model = runner.model
        
        # Store and remove seen_classes_mask
        if hasattr(model, 'module'):  # Distributed training
            model = model.module
            
        if hasattr(model, 'head') and hasattr(model.head, 'seen_classes_mask'):
            self.original_mask = model.head.seen_classes_mask
            model.head.seen_classes_mask = None
            runner.logger.info("🎯 INCREMENTAL EVAL: Removed seen_classes_mask for full evaluation")
        elif hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'seen_classes_mask'):
            self.original_mask = model.bbox_head.seen_classes_mask  
            model.bbox_head.seen_classes_mask = None
            runner.logger.info("🎯 INCREMENTAL EVAL: Removed seen_classes_mask (bbox_head) for full evaluation")
    
    def after_val_epoch(self, runner):
        """Restore seen_classes_mask after validation."""
        model = runner.model
        
        # Restore seen_classes_mask
        if hasattr(model, 'module'):  # Distributed training
            model = model.module
            
        if self.original_mask is not None:
            if hasattr(model, 'head') and hasattr(model.head, 'seen_classes_mask'):
                model.head.seen_classes_mask = self.original_mask
                runner.logger.info("🎯 INCREMENTAL EVAL: Restored seen_classes_mask for training")
            elif hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'seen_classes_mask'):
                model.bbox_head.seen_classes_mask = self.original_mask
                runner.logger.info("🎯 INCREMENTAL EVAL: Restored seen_classes_mask (bbox_head) for training")
            
            self.original_mask = None
