from torchvision.models import alexnet, resnet18

from ninpy.debug import get_imagenet_image
from ninpy.torch2 import get_num_weight_from_name


def test_get_imagenet_img():
    """Checking for an example imagenet link is dead for not."""
    test_img = get_imagenet_image(preprocess=True)
    model = alexnet(pretrained=False)
    out = model(test_img)[0]
    assert out is not None


def test_get_num_weight_from_name():
    model = resnet18(pretrained=False)
    num_weight = get_num_weight_from_name(model, "fc")

    assert num_weight == 512 * 1_000
    num_weight = get_num_weight_from_name(model, "layer1.0.conv1")
    assert num_weight == 64 * 64 * 3 * 3
