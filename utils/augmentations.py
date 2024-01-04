import albumentations as A
import random

class Albumentations:
    def __init__(self):
        self.transform = None
        import albumentations as A

        T = [
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                # A.Rotate(limit=10),
            ]  # transforms
        self.transform = A.Compose(T)


    def __call__(self, image, mask, p=1.0):
        if self.transform and random.random() < p:
            augmented = self.transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']
        return image_aug, mask_aug
