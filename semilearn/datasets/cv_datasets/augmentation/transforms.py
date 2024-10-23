from torchvision import transforms

from .randaugment import RandAugment


mean, std = {}, {}
mean["utkface"] = [0.59632254, 0.45671629, 0.39103324]
std["utkface"] = [0.25907077, 0.23132719, 0.22686818]


def get_val_transforms(crop_size, dataset_name):
    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean[dataset_name.lower()],
                std[dataset_name.lower()],
            ),
        ]
    )


def get_weak_transforms(crop_size, crop_ratio, dataset_name):
    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset_name.lower()], std[dataset_name.lower()]),
        ]
    )


def get_strong_transforms(crop_size, crop_ratio, dataset_name):
    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset_name.lower()], std[dataset_name.lower()]),
        ]
    )
