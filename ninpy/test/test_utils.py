from torchvision.models import alexnet
from ninpy.debug import get_imagenet_img
from ninpy.torch_utils import get_num_weight_from_name


def test_get_imagenet_img():
    """Only checking for working or not.
    """
    test_img = get_imagenet_img()
    model = alexnet(pretrained=False)
    out = model(test_img)[0]
    # pred_label = out.softmax(0).argmax()
    # assert pred_label == 258
    assert out is not None


def test_get_num_weight_from_name():
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)

    num_weight = get_num_weight_from_name(model, 'fc')
    assert num_weight == 512*1_000
    num_weight = get_num_weight_from_name(model, 'layer1.0.conv1')
    assert num_weight == 64*64*3*3