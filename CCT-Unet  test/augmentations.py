# -*- coding: utf-8 -*-
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transform
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms


class JointTransform:
    def __init__(self, crop_size=(190, 190), resize_Before=(384, 384), resize_After=(224, 224), rotation=False, elastic_transform=False, RandomAffine=False, RandomGaussianBlur = False):
        self.resize_Before = resize_Before
        self.resize_After = resize_After
        self.crop = crop_size
        self.rotation = rotation
        self.elastic_transform = elastic_transform
        self.RandomAffine = RandomAffine
        self.RandomGaussianBlur = RandomGaussianBlur

    def __call__(self, img, mask):
        if self.resize_Before:
            Resize_Before = Resize(self.resize_Before)
            img, mask = Resize_Before(img, mask)
        if self.crop:
            Crop = crop(self.crop)
            img, mask = Crop(img, mask)
        if self.resize_After:
            Resize_After = Resize(self.resize_After)
            img, mask = Resize_After(img, mask)
        if self.RandomGaussianBlur:
            Gaussianblur = RandomGaussianBlur(p=0.5)
            img, mask = Gaussianblur(img, mask)

        if self.rotation:
            Random_rotation = RandomRotation(5)
            img, mask = Random_rotation(img, mask)
        if self.RandomAffine:
            Random_Affine = RandomAffine(90)
            img, mask = Random_Affine(img, mask)

        if self.elastic_transform:
            Elastic_Transform = Elastic_transform(p=0.7)
            img, mask = Elastic_Transform(img, mask)

        TT = ToTensor()
        img, mask = TT(img, mask)

        # NM = Normalize(mean=[ 0.2439 ], std=[ 0.1821 ])
        # img, mask = NM(img, mask)

        return img, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)

        return img, mask


# 对 mask 无影响
"""

"""


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return F.normalize(img, self.mean, self.std), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        return F.adjust_saturation(img, saturation_factor), mask


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
        return F.adjust_hue(img, hue_factor), mask


class AdjustBrightness(object):
    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, img, mask):
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        return F.adjust_brightness(img, brightness_factor), mask


class AdjustContrast(object):
    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, img, mask):
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        return F.adjust_contrast(img, contrast_factor), mask


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        transforms = [ AdjustSaturation(saturation),
                       AdjustBrightness(brightness),
                       AdjustContrast(contrast),
                       AdjustHue(hue) ]
        random.shuffle(transforms)
        transform = Compose(transforms)
        return transform

    def __call__(self, img, mask):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img, mask)


# 对 mask 有影响
"""

"""


class ToTensor(object):
    def __call__(self, img, mask):
        return F.to_tensor(img), mask  # F.to_tensor(mask)二分割使用


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        if isinstance(self.size, int):
            if w > h:
                new_w, new_h = self.size * w / h, self.size
            else:
                new_w, new_h = self.size, self.size * h / w
        else:
            new_w, new_h = self.size

        new_w, new_h = int(new_w), int(new_h)

        return (img.resize((new_w, new_h), Image.BILINEAR),
                mask.resize((new_w, new_h), Image.NEAREST))


class RandomAffine(object):
    def __init__(self, degree):
        self.degree = degree
        self.p = 0.5

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        if random.random() < self.p:
            return (
                F.affine(img,
                         translate=(0, 0),
                         scale=1.0,
                         angle=rotate_degree,
                         interpolation=Image.BILINEAR,
                         fill=0,
                         shear=0.0),
                F.affine(mask,
                         translate=(0, 0),
                         scale=1.0,
                         angle=rotate_degree,
                         interpolation=Image.NEAREST,
                         fill=250,
                         shear=0.0))
        return img, mask


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree
        self.p = 0.5

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        if random.random() < self.p:
            return (
                F.rotate(img,
                         angle=rotate_degree,
                         interpolation=Image.BILINEAR),
                F.rotate(mask,
                         angle=rotate_degree,
                         interpolation=Image.NEAREST)
            )
        return img, mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class center_crop(object):
    def __init__(self, size):
        self.size = size
        self.p = 0.5

    def __call__(self, img, mask):
        CenterCrop = transform.CenterCrop(size=self.size)
        if random.random() < self.p:
            img = CenterCrop(img)
            mask = CenterCrop(mask)
            return (
                img.resize((224, 224), Image.BILINEAR),
                mask.resize((224, 224), Image.NEAREST),
            )
        return img, mask


class RandomResizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        ResizedCrop = transforms.RandomResizedCrop(224, scale=(0.2, 1.))
        image = ResizedCrop(img)
        mask = ResizedCrop(mask)
        return image, mask


class crop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        CenterCrop = transform.CenterCrop(size=self.size)
        cropped_image = CenterCrop(img)
        cropped_mask = CenterCrop(mask)
        return cropped_image, cropped_mask


class Elastic_transform(object):
    def __init__(self, p=0.7):
        self.p = p

    def elastic_transform(self, image, image_mask, alpha, sigma, random_state=None):
        # assert len(image.shape)==2
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[ 0 ]), np.arange(shape[ 1 ]), indexing='ij')

        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape), map_coordinates(image_mask, indices,
                                                                                        order=1).reshape(shape)

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.array(img)
            mask = np.array(mask)
            img, mask = self.elastic_transform(img, mask, img.shape[ 1 ] * 2, img.shape[ 1 ] * 0.08,
                                               img.shape[ 1 ] * 0.08)
            return img, mask
        return img, mask


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        Gaussianblur = transforms.RandomApply([GaussianBlur([.1, 2.])], p=self.p)
        return Gaussianblur(img), mask