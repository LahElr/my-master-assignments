from itertools import count
from typing import List, Union
import numpy
import utils
import cv2


def count_pixel(image: numpy.ndarray, limit: int = 255) -> List[int]:
    '''
    This function count the apperence of each value in [0,`limit`] in an image

    parameters:
    `image`: the image channel, in shape [H,W]
    `limit`: the upper limit of the pixel value

    return: a list of `limit`+1 integers
    '''
    if image.shape.__len__() != 2:
        raise ValueError(
            f"he.count_pixel: the parameter `image` should be a slice of ONLY ONE color channel, i.e. has a shape of [H,W], but got shape {image.shape}."
        )
    counts = [0 for i in range(0, limit + 1)]
    image = image.reshape(-1)
    for p in image:
        counts[p] += 1
    return counts


class simple_histogram_equlization_mapper:
    '''
    This simple mapper uses this formula to carry out histogram equlization:

    $s_k=(L-1)\sum^{k}_{j=0}p_j$
    '''

    def __init__(self,
                 image: numpy.ndarray,
                 channel: str,
                 bin_count: int,
                 pixel_in_bin_function: str = "zero",
                 limit: int = 255,
                 clip: float = None,
                 clip_way: str = "distribute"):
        '''
        parameters:
        `image`: the image
        `channel`: "red","green","blue" or "average", which color channel to count
        `bin_count`: the number of bins of histogram
        `limit`: the upper limit of the pixel value
        `pixel_in_bin_function`: how to decide the position of the pixel in its new bin
        `clip`: `None` by default, if a float in (0,1), apply the contras limit method by cliping values with over `image_size * clip` appereances, if an integer >= 1, use the `clip`.
        `clip_way`: "distribute" or "zoom", works only when `clip` is not `None`, how to do normalization after clipping.
        '''
        image = utils.channel_map(image,
                                  channel)  # map the image to single color
        self.image = image

        self.image_size = image.shape[0] * image.shape[1]
        self.limit = limit
        self.bin_count = bin_count
        self.bin_size = (limit + 1) / self.bin_count
        self.pixels_per_bin = self.image_size / self.bin_count

        pixel_value_count = count_pixel(image,
                                        limit)  # how many pixel with value `i`

        # do clip if needed
        if clip is not None:
            if clip <= 0:
                raise ValueError(
                    f"he.simple_histogram_equlization_mapper.__init__: the param `clip` should either `None`, a float in (0,1) or an positive integer, but got {clip}."
                )
            if 0 < clip < 1:
                clip = int(clip * self.image_size)
            # print(clip)
            spare_pixel_count = 0
            for i, x in enumerate(pixel_value_count):
                if x > clip:
                    spare_pixel_count += x - clip
                    pixel_value_count[i] = clip
            # print(spare_pixel_count)
            # normalize
            clipped_pixel_sum = sum(pixel_value_count)
            if clip_way == "distribute":
                for i in range(len(pixel_value_count)):
                    pixel_value_count[i] += spare_pixel_count / (self.limit+1)
                    # pixel_value_count[i] = pixel_value_count[i]*self.image_size / clipped_pixel_sum
            elif clip_way == "zoom":
                for i in range(len(pixel_value_count)):
                    # pixel_value_count[i] += spare_pixel_count / (self.limit+1)
                    pixel_value_count[i] = pixel_value_count[i]*self.image_size / clipped_pixel_sum
            # print(sum(pixel_value_count),self.image_size,spare_pixel_count)
            # exit(0)

        self.pixel_value_sums = [0 for i in range(len(pixel_value_count))
                                 ]  # how many pixels with value `<=i`
        self.pixel_value_sums[0] = pixel_value_count[0]
        for i in range(1, len(pixel_value_count)):
            self.pixel_value_sums[i] = self.pixel_value_sums[
                i - 1] + pixel_value_count[i]

        self.difference_pixels_to_bin_start_functions = {
            "reverse":
            self.difference_pixels_to_bin_start_by_reverse,
            "original":
            self.difference_pixels_to_bin_start_by_original,
            "zero":
            self.difference_pixels_to_bin_start_zero,
            "full_equalization":
            self.difference_pixels_to_bin_start_full_equalization
        }
        self.pixel_in_bin_function = pixel_in_bin_function

    def __call__(
            self,
            x: Union[numpy.ndarray, int] = None,
            pixel_in_bin_function: str = None) -> Union[numpy.ndarray, int]:
        '''
        transforms a pixel or an image

        parameters:
        `x`: the pixel, `int`, or an `numpy.ndarray` with shape [H,W], by default, use the image that the heer was defined with
        `pixel_in_bin_function`: how to decide the position of the pixel in its new bin, if you want to specially specify one

        returns:
        the transformed pixel or image
        '''
        if type(x) is int:
            return self.transforme_pixel(x, pixel_in_bin_function)
        elif type(x) is numpy.ndarray:
            return self.transform_image(x, pixel_in_bin_function)
        elif x is None:
            return self.transform_image(self.image, pixel_in_bin_function)
        else:
            raise TypeError(
                f"he.simple_histogram_equlization_mapper.__call__: The parameter `x` should be either int or numpy.ndarray of shape [H,W], but got {type(x)}"
            )

    def transform_pixel(self,
                        in_pixel: int,
                        pixel_in_bin_function: str = None) -> int:
        '''
        Give the mapped value of a pixel

        parameters:
        `in_pixel`: the pixel value to be transformed
        `pixel_in_bin_function`: how to decide the position of the pixel in its new bin, if you want to specially specify one
        '''
        if pixel_in_bin_function is None:
            # use the default one
            pixel_in_bin_function = self.pixel_in_bin_function

        if in_pixel < 0 or in_pixel > self.limit:
            raise ValueError(
                f"he.simple_histogram_equlization_mapper.__call__: The value of `in_pixel` should be in the range of [0,{self.limit}], but specified {in_pixel}."
            )

        # if pixel_in_bin_function == "full_equalization":
        #     ret = self.full_equalization(in_pixel)
        # else:
        target_bin_number = int(
            self.pixel_value_sums[in_pixel] /
            self.pixels_per_bin)  # which bin the pixel will go
        differenct_pixel_to_bin_start = self.difference_pixels_to_bin_start_functions[
            pixel_in_bin_function](
                target_bin_number,
                in_pixel)  # where in the new bin the pixel will go
        ret = target_bin_number * self.bin_size + differenct_pixel_to_bin_start

        # abnormal values
        if ret > self.limit:
            # overflow: just cut
            ret = self.limit
        if ret < 0:
            ret = 0
        return ret, differenct_pixel_to_bin_start

    def transform_image(self,
                        image: numpy.ndarray,
                        pixel_in_bin_function: str = None) -> numpy.ndarray:
        '''
        Give the hisogram equalized image

        parameters:
        `image`: the image to be transformed, single color, shape: [H,W]
        `pixel_in_bin_function`: how to decide the position of the pixel in its new bin, if you want to specially specify one
        '''
        position_bias_counter = 0  # for log
        if len(image.shape) != 2:
            raise ValueError(
                f"he.simple_histogram_equlization_mapper.transform_image: the image to be transformed must be a single color channel image of shape [H, W], but got {image.shape}."
            )

        ret = numpy.zeros(shape=image.shape, dtype=image.dtype)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x = self.transform_pixel(image[i, j], pixel_in_bin_function)
                ret[i, j] = x[0]
                position_bias_counter += x[1]
        # print(
        #     f"sumed bias for {pixel_in_bin_function} is {position_bias_counter}"
        # )
        return ret

    def difference_pixels_to_bin_start_by_reverse(self, target_bin_number,
                                                  in_pixel) -> int:
        '''
        decide the new position of the pixel in the new bin by its relative postion between the original of the value of the start and end of the new bin
        
        eg. the pixel, whose original value is 27, is allocated to new bin #1, and the bin size is 10, so bin#1 contains the pixels with new value in [10,20),
        the value of the start and the end, 10 and 20, corresponds the 10th smallest value and 20th smallest value in the original image, respectively,
        for example, 22 and 42, between which, 27 is at the 5/20=25% position, thus the pixel's new position should also be at the 25% position among the new bin,
        that is, 10*25%=2.5, rounded to be 2, so the new value of the pixel is 12.
        '''
        bin_start_reverse_rank = int(
            target_bin_number * self.pixels_per_bin
        )  # how many pixels have value <= the start of the mapped bin
        bin_start_reverse = 0  # the original value of the pixel at the start of the mapped bin
        for i, v in enumerate(self.pixel_value_sums):
            if v >= bin_start_reverse_rank:
                bin_start_reverse = i
                break

        bin_end_reverse_rank = self.image_size if target_bin_number >= self.bin_count - 1 else int(
            (target_bin_number + 1) * self.pixels_per_bin
        )  # how many pixels have value <= the end of the mapped bin
        bin_end_reverse = 0  # the original value of the pixel at the end of the mapped bin
        for i, v in enumerate(self.pixel_value_sums):
            if v >= bin_end_reverse_rank:
                bin_end_reverse = i
                break

        if bin_end_reverse - bin_start_reverse > 0:
            return int(((in_pixel - bin_start_reverse) /
                        (bin_end_reverse - bin_start_reverse)) * self.bin_size)
        else:
            return 0

    def difference_pixels_to_bin_start_by_original(self, target_bin_number,
                                                   in_pixel) -> int:
        '''
        decide the exact position of the pixel in the bin by just use the difference 
        of value of its original position to the bin start pixel's original position

        eg. the pixel, whose original value is 27, is allocated to new bin #1, and the bin size is 10, so bin#1 contains the pixels with new value in [10,20),
        the value of the start and the end, 10 and 20, corresponds the 10th smallest value and 20th smallest value in the original image, respectively,
        for example, 22 and 42, the difference of 27 to 22 is 5, so the pixel's new value is 10+5=15
        '''
        # target_bin_number = int(self.pixel_value_sums[in_pixel] / self.pixels_per_bin)
        # ret = self.pixel_value_sums[in_pixel] - self.pixel_value_sums[int(
        #     (in_pixel // self.bin_size) * self.bin_size)]
        bin_start_reverse_rank = int(target_bin_number * self.pixels_per_bin)
        bin_start_reverse = 0
        for i, v in enumerate(self.pixel_value_sums):
            if v >= bin_start_reverse_rank:
                bin_start_reverse = i
                break
        ret = in_pixel - bin_start_reverse
        if ret > self.bin_size:
            ret = self.bin_size
        return ret

    def difference_pixels_to_bin_start_zero(self, target_bin_number,
                                            in_pixel) -> int:
        '''
        just put the pixel at the start position of the new bin it belongs to
        '''
        return 0

    def difference_pixels_to_bin_start_full_equalization(
            self, target_bin_number, in_pixel) -> int:
        '''
        decide the position of the pixel as if the `bin_count` is `limit+1`
        '''
        target = int(self.pixel_value_sums[in_pixel] * (self.limit + 1) /
                     self.image_size)  # bin number if bin_count == limit+1
        return target - target_bin_number * self.bin_size

    def full_equalization(self, in_pixel) -> int:
        '''
        decide the position of the pixel as if the `bin_count` is `limit+1`
        '''
        target = int(self.pixel_value_sums[in_pixel] * (self.limit + 1) /
                     self.image_size)  # bin number if bin_count == limit+1
        return target


def open_cv_hsitogram_mapper(image: numpy.ndarray) -> numpy.ndarray:
    '''
    This FUNCTION calls opencv to do histogram equalization, for reference

    parameters:
    `image`: single channel image of shape [H,W]
    '''
    # ret = numpy.zeros(shape=image.shape, dtype=image.dtype)
    return cv2.equalizeHist(image)
    # return ret
