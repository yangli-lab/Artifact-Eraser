from torchvision import transforms


def get_transform():
    transform_dict = {
        'gt_train':transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'source': None,
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'inference': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    return transform_dict

def DFDC_Transform(if_normalize = True):
    transform_dict = {
        'test': transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255]
            , [38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255])
        ]
    ),
        'gt_train': transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255],
                [38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255])
            ]
        )
    }
    return transform_dict

