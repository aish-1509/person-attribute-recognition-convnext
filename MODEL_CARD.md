# Model Card

## Model

**Name:** Multi-Task ConvNeXt Person Attribute Recognition

**Version:** 1.0.0

**Framework:** PyTorch and `timm`

**Architecture:** ConvNeXt-Small shared backbone with six linear classification heads

**Input:** One RGB person crop resized to 256 x 128 pixels

**Outputs:** Age group, headgear, gender label, glasses label, upper-body clothing color, and lower-body clothing color

## Intended Use

This model is a research and portfolio prototype for demonstrating multi-task image classification, imbalance-aware training, group-safe evaluation, and inference packaging. It may be useful as a starting point for non-consequential analytics on already-cropped pedestrian imagery.

It is not intended for identity recognition, surveillance decisions, employment decisions, access control, law enforcement, or any high-impact automated decision.

## Training and Evaluation Design

The source data uses Market-1501-style filenames in which multiple images may correspond to the same person. A random image split would leak identities across train and test sets. The project therefore extracts person IDs and uses `GroupShuffleSplit` with a seed search to produce:

| Split | Images | Share |
|---|---:|---:|
| Train | 56,040 | 80.0% |
| Validation | 6,939 | 9.9% |
| Test | 7,021 | 10.0% |

Verified identity overlap between splits: **0**.

The best checkpoint was chosen by validation mean macro-F1, not accuracy or loss. This gives minority classes equal importance during model selection.

## Held-Out Test Performance

| Attribute | Accuracy | Macro Recall | Macro F1 |
|---|---:|---:|---:|
| Age | 0.9909 | 0.9180 | 0.9334 |
| Headgear | 0.9872 | 0.9661 | 0.8849 |
| Gender | 0.9548 | 0.9545 | 0.9541 |
| Glasses | 0.8476 | 0.7448 | 0.7425 |
| Upper-body color | 0.8137 | 0.7734 | 0.7688 |
| Lower-body color | 0.8100 | 0.7135 | 0.7011 |

Mean macro-F1 across all six attributes: **0.8308**.

## Limitations

- The training data is dominated by CCTV-style pedestrian crops. Performance can drop on eye-level, high-resolution, artistic, or otherwise shifted imagery.
- Glasses occupy very few pixels at 256 x 128 resolution and remain difficult to classify.
- Dark clothing colors are visually ambiguous under shadows, compression, and occlusion.
- Rare headgear classes have limited support despite loss reweighting.
- The binary gender label reflects the source annotation schema and must not be interpreted as gender identity.
- Confidence values are softmax scores and have not been calibrated for operational thresholds.
- Aggregate metrics do not establish equal performance across demographic groups or camera environments.

## Responsible Use

Review data rights, consent, local law, demographic performance, and error costs before any deployment. Keep a human in the loop, expose uncertainty, log model versions, and monitor domain drift. Do not infer attributes when the image does not provide adequate visual evidence.

## Artifact Integrity

The release checkpoint stores model weights, image dimensions, task names, and label maps. Its SHA-256 checksum is published with the GitHub release. Only load checkpoints from a trusted source.
