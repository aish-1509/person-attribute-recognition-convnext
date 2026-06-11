# Model Checkpoint

Download the public v1.0.0 checkpoint:

```bash
curl -L \
  -o models/best_model.pth \
  https://github.com/aish-1509/person-attribute-recognition-convnext/releases/download/v1.0.0/best_model.pth
```

The checkpoint is stored as a GitHub release asset because its size exceeds GitHub's 100 MB limit for normal repository files. It contains:

- ConvNeXt-Small backbone and six classification heads
- learned task-weighting parameters
- task names and label maps
- expected image height and width

Verify the SHA-256 checksum against the release notes before loading it.
