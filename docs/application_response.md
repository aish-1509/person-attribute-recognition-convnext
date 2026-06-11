# Portfolio Materials

- **GitHub project:** [Leakage-Safe Multi-Task Person Attribute Recognition](https://github.com/aish-1509/person-attribute-recognition-convnext)
- **Technical notebook:** Full data audit, split design, model training, evaluation, and diagnostics in [`notebooks/person_attribute_recognition.ipynb`](../notebooks/person_attribute_recognition.ipynb)
- **Visual defense:** [`PAR_Defense_Deck.pdf`](PAR_Defense_Deck.pdf)
- **Evidence:** Per-epoch logs, held-out predictions, learning curves, confusion matrices, reusable inference code, and a public model checkpoint release

## One AI Project I Am Most Proud Of

I built a leakage-safe multi-task computer vision system that predicts six pedestrian attributes from a single image: age group, headgear, gender label, glasses, upper-body color, and lower-body color.

The most important problem was not simply choosing a model. During data exploration, I recognized that filenames encoded person identities and that a normal random split would place images of the same person in both training and evaluation. That would produce impressive but misleading results. I designed a group-aware split with zero identity overlap, then built the full pipeline: data validation, color-safe augmentation, a shared ConvNeXt-Small backbone with six prediction heads, focal loss, learned task weighting, checkpointing, error analysis, and a packaged inference CLI.

I used Python, PyTorch, `timm`, Albumentations, scikit-learn, OpenCV, pandas, and Google Colab. The model reached **0.8308 mean macro-F1** on 7,021 held-out images while using one shared network for all six tasks.

This was an evaluated take-home prototype rather than a production system with external users. Its impact was demonstrating that I can find hidden business and data risks, make defensible product and modeling trade-offs, and turn an experiment into a clear, reproducible technical artifact.

## Short Version

I built a multi-task computer vision system that predicts six pedestrian attributes with one ConvNeXt model. I personally owned the pipeline end to end, from discovering identity leakage in the proposed data split through group-safe evaluation, imbalance-aware training, failure analysis, and inference packaging. Using PyTorch, `timm`, Albumentations, scikit-learn, OpenCV, and Colab, the system achieved **0.8308 mean macro-F1** on 7,021 held-out images with zero person-identity overlap across splits. It was a take-home prototype, not a live product, but it demonstrates how I approach messy AI problems: challenge the metric, protect evaluation quality, make practical trade-offs, and communicate the system clearly enough for others to use and assess.
