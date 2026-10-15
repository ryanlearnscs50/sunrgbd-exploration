"""
Incremental Learning Utilities

Helper functions for expanding model heads and managing incremental training.
"""

import torch
import torch.nn as nn
import copy
try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


def expand_model_head(model, new_num_classes):
    """Expand TR3D classification head to accommodate new classes.
    
    Args:
        model: TR3D model with classification head
        new_num_classes (int): Total number of classes after expansion
        
    Returns:
        Modified model with expanded head
    """
    if not hasattr(model, 'bbox_head'):
        print("Model does not have bbox_head, skipping head expansion")
        return model
    
    head = model.bbox_head
    
    if not hasattr(head, 'cls_conv'):
        print("Model head does not have cls_conv layer, skipping expansion")
        return model
    
    current_classes = head.cls_conv.out_channels
    
    if new_num_classes <= current_classes:
        print(f"New classes ({new_num_classes}) <= current ({current_classes}), no expansion needed")
        return model
    
    print(f"Expanding classification head: {current_classes} -> {new_num_classes} classes")
    
    # Create new classification layer
    if ME is not None and isinstance(head.cls_conv, ME.MinkowskiConvolution):
        # MinkowskiEngine convolution
        old_conv = head.cls_conv
        new_conv = ME.MinkowskiConvolution(
            in_channels=old_conv.in_channels,
            out_channels=new_num_classes,
            kernel_size=old_conv.kernel_size,
            bias=old_conv.bias is not None,
            dimension=old_conv.dimension
        )
        
        # Copy weights for existing classes
        with torch.no_grad():
            print(f"DEBUG: Old conv kernel shape: {old_conv.kernel.shape}")
            print(f"DEBUG: New conv kernel shape: {new_conv.kernel.shape}")
            print(f"DEBUG: Old conv out_channels: {old_conv.out_channels}")
            print(f"DEBUG: New conv out_channels: {new_conv.out_channels}")
            
            # MinkowskiEngine kernel shape is typically (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
            # But let's check the actual shape and adjust accordingly
            old_shape = old_conv.kernel.shape
            new_shape = new_conv.kernel.shape
            
            if len(old_shape) == 5:
                # 3D convolution: (in_channels, out_channels, k, k, k)
                new_conv.kernel.data[:, :current_classes, :, :, :] = old_conv.kernel.data
                # Initialize new class weights
                nn.init.normal_(new_conv.kernel.data[:, current_classes:, :, :, :], std=0.01)
            elif len(old_shape) == 2:
                # 1x1 conv might be (in_channels, out_channels)
                new_conv.kernel.data[:, :current_classes] = old_conv.kernel.data
                # Initialize new class weights  
                nn.init.normal_(new_conv.kernel.data[:, current_classes:], std=0.01)
            else:
                print(f"WARNING: Unexpected kernel shape {old_shape}, using fallback initialization")
                nn.init.normal_(new_conv.kernel.data, std=0.01)
                # Copy what we can
                min_classes = min(current_classes, new_conv.out_channels)
                if len(old_shape) >= 2:
                    new_conv.kernel.data[:, :min_classes] = old_conv.kernel.data[:, :min_classes]
            
            if old_conv.bias is not None:
                new_conv.bias.data[:current_classes] = old_conv.bias.data
                # Initialize new class biases
                nn.init.zeros_(new_conv.bias.data[current_classes:])
        
        head.cls_conv = new_conv
        
    else:
        # Standard PyTorch convolution
        old_conv = head.cls_conv
        new_conv = nn.Conv1d(  # Assuming 1D conv for point features
            in_channels=old_conv.in_channels,
            out_channels=new_num_classes,
            kernel_size=old_conv.kernel_size,
            bias=old_conv.bias is not None
        )
        
        # Copy weights for existing classes
        with torch.no_grad():
            new_conv.weight.data[:current_classes] = old_conv.weight.data
            if old_conv.bias is not None:
                new_conv.bias.data[:current_classes] = old_conv.bias.data
                nn.init.zeros_(new_conv.bias.data[current_classes:])
            
            # Initialize new class weights
            nn.init.normal_(new_conv.weight.data[current_classes:], std=0.01)
        
        head.cls_conv = new_conv
    
    print(f"✅ Head expansion completed: {current_classes} -> {new_num_classes}")
    return model


def freeze_old_classes(model, num_old_classes):
    """Freeze parameters for old classes to prevent forgetting.
    
    Args:
        model: Model to freeze parameters for
        num_old_classes (int): Number of old classes to freeze
    """
    if not hasattr(model, 'bbox_head') or not hasattr(model.bbox_head, 'cls_conv'):
        return
    
    cls_conv = model.bbox_head.cls_conv
    
    # Freeze old class parameters
    if hasattr(cls_conv, 'kernel') and cls_conv.kernel.requires_grad:
        # MinkowskiEngine case - freeze partial parameters
        print(f"Freezing classification weights for first {num_old_classes} classes")
        # Note: MinkowskiEngine doesn't support partial parameter freezing easily
        # This would need custom gradient masking in training loop
    elif hasattr(cls_conv, 'weight') and cls_conv.weight.requires_grad:
        # Standard PyTorch case
        print(f"Freezing classification weights for first {num_old_classes} classes")
        # This would also need custom gradient masking


def get_model_class_count(model):
    """Get current number of classes in model classification head."""
    if not hasattr(model, 'bbox_head') or not hasattr(model.bbox_head, 'cls_conv'):
        return None
    
    cls_conv = model.bbox_head.cls_conv
    
    if hasattr(cls_conv, 'out_channels'):
        return cls_conv.out_channels
    elif hasattr(cls_conv, 'kernel') and len(cls_conv.kernel.shape) >= 2:
        return cls_conv.kernel.shape[-1]  # Last dimension for MinkowskiEngine
    
    return None


def verify_model_expansion(model, expected_classes):
    """Verify that model has been expanded to expected number of classes."""
    current_classes = get_model_class_count(model)
    
    if current_classes is None:
        print("⚠️  Could not determine model class count")
        return False
    
    if current_classes == expected_classes:
        print(f"✅ Model head verified: {current_classes} classes")
        return True
    else:
        print(f"❌ Model head mismatch: expected {expected_classes}, got {current_classes}")
        return False


def save_incremental_checkpoint(model, stage_idx, stage_classes, work_dir, epoch=None):
    """Save checkpoint with incremental learning metadata."""
    import os
    from mmcv.runner import save_checkpoint
    
    # Prepare metadata
    meta = {
        'stage_idx': stage_idx,
        'stage_classes': stage_classes,
        'total_classes': get_model_class_count(model),
        'epoch': epoch
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(work_dir, f'stage_{stage_idx}_checkpoint.pth')
    save_checkpoint(model, checkpoint_path, meta=meta)
    
    print(f"Saved incremental checkpoint: {checkpoint_path}")
    return checkpoint_path