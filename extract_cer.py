import os
from tensorboard.backend.event_processing import event_accumulator

# Parse tensorboard logs
models = [
    ('ukrainian_aspect_ratio', 'models/ukrainian_aspect_ratio/runs/Oct22_08-19-07_PLSV1223'),
    ('ukrainian_normalized', 'models/ukrainian_model_normalized/runs/Oct21_17-08-12_PLSV1223'),
    ('efendiev_3', 'models/efendiev_3_model/runs/Oct20_09-05-20_PLSV1223'),
]

for name, log_dir in models:
    if not os.path.exists(log_dir):
        continue
    
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    
    # Look for CER metrics
    for tag in tags['scalars']:
        if 'cer' in tag.lower() or 'eval' in tag.lower():
            events = ea.Scalars(tag)
            print(f"\n{tag}:")
            for i, event in enumerate(events[-10:]):  # Last 10 entries
                print(f"  Step {event.step:6d}: {event.value:.4f}")
