# Pseudo Label Configuration for Incremental Learning
# This configuration provides settings for generating and using pseudo labels
# to enhance replay scenes in incremental learning scenarios.

# Default pseudo label configuration
pseudo_label_config = dict(
    # Core pseudo labeling settings
    confidence_threshold=0.7,           # Minimum confidence for pseudo labels (0.0-1.0)
    nms_threshold=0.3,                  # IoU threshold for 3D NMS
    max_pseudo_per_scene=50,            # Maximum pseudo labels per scene
    debug_mode=True,                    # Enable detailed logging
    
    # Cache settings
    use_cache=True,                     # Cache pseudo labels to disk
    cache_prefix='pseudo_labels',       # Prefix for cache files
    
    # Quality control
    min_box_size=0.1,                  # Minimum bounding box size (meters)
    max_distance_from_camera=20.0,     # Maximum distance from camera center
    
    # Integration settings
    merge_strategy='append',            # How to merge with existing labels ('append', 'replace')
    weight_pseudo_loss=0.5,            # Weight for pseudo label loss (if supported)
)

# Conservative configuration (higher quality, fewer labels)
conservative_pseudo_label_config = dict(
    confidence_threshold=0.8,           # Higher confidence requirement
    nms_threshold=0.2,                  # Stricter NMS
    max_pseudo_per_scene=20,            # Fewer labels per scene
    debug_mode=True,
    use_cache=True,
    cache_prefix='pseudo_labels_conservative',
    min_box_size=0.2,
    max_distance_from_camera=15.0,
    merge_strategy='append',
    weight_pseudo_loss=0.3,             # Lower weight for pseudo labels
)

# Aggressive configuration (more labels, lower threshold)
aggressive_pseudo_label_config = dict(
    confidence_threshold=0.6,           # Lower confidence threshold
    nms_threshold=0.4,                  # More lenient NMS
    max_pseudo_per_scene=80,            # More labels per scene
    debug_mode=True,
    use_cache=True,
    cache_prefix='pseudo_labels_aggressive',
    min_box_size=0.05,
    max_distance_from_camera=25.0,
    merge_strategy='append',
    weight_pseudo_loss=0.7,             # Higher weight for pseudo labels
)

# Balanced configuration (recommended for most use cases)
balanced_pseudo_label_config = dict(
    confidence_threshold=0.75,          # Balanced confidence
    nms_threshold=0.3,                  # Standard NMS
    max_pseudo_per_scene=40,            # Moderate number of labels
    debug_mode=False,                   # Less verbose for production
    use_cache=True,
    cache_prefix='pseudo_labels_balanced',
    min_box_size=0.1,
    max_distance_from_camera=20.0,
    merge_strategy='append',
    weight_pseudo_loss=0.5,
)

# Development/debugging configuration
debug_pseudo_label_config = dict(
    confidence_threshold=0.5,           # Lower threshold for more examples
    nms_threshold=0.5,                  # More lenient for debugging
    max_pseudo_per_scene=10,            # Fewer for faster debugging
    debug_mode=True,                    # Extensive logging
    use_cache=False,                    # No caching for debugging
    cache_prefix='pseudo_labels_debug',
    min_box_size=0.05,
    max_distance_from_camera=30.0,
    merge_strategy='append',
    weight_pseudo_loss=0.5,
)