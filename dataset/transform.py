import cv2
import numpy as np
import torch


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, valid_mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "valid_mask" in sample:
                sample["valid_mask"] = cv2.resize(
                    sample["valid_mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        self.type_mappings = {
            'image': np.float32,
            'depth': np.float32,
            'valid_mask': bool,
        }

    def prepare_data(self, data, data_type):
        if len(data.shape) == 3: # multi-layered, rgb, etc.
            data = np.transpose(data, (2, 0, 1))
        data = data.astype(data_type)
        data = np.ascontiguousarray(data)
        data = torch.from_numpy(data)
        return data

    def __call__(self, sample):
        for key in self.type_mappings.keys():
            if key in sample:
                sample[key] = self.prepare_data(sample[key], self.type_mappings[key])
        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        assert h >= self.size[0] and w >= self.size[1], 'Wrong size'
        
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]
        
        sample['image'] = sample['image'][h_start: h_end, w_start: w_end, :]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start: h_end, w_start: w_end]
        
        if "valid_mask" in sample:
            sample["valid_mask"] = sample["valid_mask"][h_start: h_end, w_start: w_end]

        return sample

class Flip(object):
    """Flip the sample horizontally.
    """

    def __init__(self, p=0.5):
        self.p = p
        pass

    def __call__(self, sample):
        # image shape (H, W, 3)
        if np.random.rand() < self.p:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)
    
        if "depth" in sample:
            sample["depth"] = cv2.flip(sample["depth"], 1)

        if "valid_mask" in sample:
            sample["valid_mask"] = cv2.flip(sample["valid_mask"], 1)

        return sample
