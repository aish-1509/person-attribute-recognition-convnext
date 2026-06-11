import pytest
import torch
from PIL import Image

from par_model.model import MultiTaskPAR
from par_model.predict import preprocess_image


def test_preprocess_image_shape_and_dtype() -> None:
    image = Image.new("RGB", (40, 80), color=(128, 64, 32))

    tensor = preprocess_image(image, height=256, width=128)

    assert tensor.shape == (1, 3, 256, 128)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


@pytest.mark.slow
def test_model_returns_one_head_per_task() -> None:
    tasks = {"Age": 3, "Glasses": 2}
    model = MultiTaskPAR(tasks, model_name="resnet18", pretrained=False)
    model.eval()

    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, 64, 64))

    assert set(outputs) == set(tasks)
    assert outputs["Age"].shape == (1, 3)
    assert outputs["Glasses"].shape == (1, 2)
