# --------------------------------------------------------
# this code is borrowed from https://github.com/cuishuhao/GVB/blob/master/GVB-GD/pre_process.py
# --------------------------------------------------------

from torchvision import transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def transform_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])


def transform_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])
