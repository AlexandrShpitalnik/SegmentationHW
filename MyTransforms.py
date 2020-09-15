
from collections.abc import Iterable
from PIL import Image
import numpy as np
import math
import random
import numbers

import torchvision.transforms.functional as F


class AbstractTransform(object):
    """Base class for all transforms.
class Compose(object):
    Its role is to simulate parametric polymorphism so that
    if the transform is called with image only it return img only
    and if the transform is call with image and keypoints, it return both.
    This is done in order to add keypoint transformation without breaking previous interface.
    """

    def __init__(self):
        pass

    def __call__(self, img, keypoints = None):
        kp = []
        if keypoints is not None:
            kp = keypoints
        img, kp = self.run(img, kp)
        if keypoints is not None:
            return img, kp
        return img


class Compose(AbstractTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def run(self, img, keypoints):
        for t in self.transforms:
            img, keypoints = t(img, keypoints)
        return img, keypoints

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(AbstractTransform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __init__(self):
        super(ToTensor, self).__init__()

    def run(self, pic, keypoints):
        return F.to_tensor(pic), keypoints

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToPILImage(AbstractTransform):
    """Convert a tensor or an ndarray to PIL Image.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
               ``short``).
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def run(self, pic, keypoints):
        return F.to_pil_image(pic, self.mode), keypoints

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class RandomRotation(AbstractTransform):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def run(self, img, keypoints):

        if self.center is None:
            self.center = [img.width / 2, img.height / 2]
        angle = self.get_params(self.degrees)
        inrad = -math.radians(angle)
        for pointPair in keypoints:
            x, y = pointPair
            x -= self.center[0]
            y -= self.center[1]
            pointPair[0] = math.cos(inrad) * x - math.sin(inrad) * y
            pointPair[1] = math.sin(inrad) * x + math.cos(inrad) * y
            pointPair[0] += self.center[0]
            pointPair[1] += self.center[1]
        return F.rotate(img, angle, self.resample, self.expand, self.center), keypoints

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomCrop(AbstractTransform):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def run(self, img, keypoints):
        w, h = img.size
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img.crop((left, top, left + new_h, top + new_w))

        for pointPair in keypoints:
            pointPair[0] -= left
            pointPair[1] -= top

        return img, keypoints

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class Resize(AbstractTransform):


    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def run(self, img, keypoints):

        w, h = img.size

        if h > w:
            new_h, new_w = self.size * h / w, self.size
        else:
            new_h, new_w = self.size, self.size * w / h

        new_h, new_w = int(new_h), int(new_w)


        img = F.resize(img, (new_h, new_w))

        x_k = new_w / w
        y_k = new_h / h

        for pointPair in keypoints:
            pointPair[0] *= x_k
            pointPair[1] *= y_k
        return img, keypoints

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
