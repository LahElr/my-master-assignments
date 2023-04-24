from tracemalloc import start
from typing import Dict, List
import numpy
import cv2
import matplotlib.pyplot as plt


def read_img(name: str) -> numpy.ndarray:
    '''
    This function can read an image and convert it to RGB channels, shape is (H,W,C)

    #! DO NOTE THE DTYPE IS UINT8
    '''
    image = cv2.imread(name,
                       cv2.IMREAD_COLOR)  #IMREAD_GRAYSCALE IMREAD_UNCHANGED
    # B G R
    image = image[:, :, (2, 1, 0)]  # -> RGB
    return image


def show_img(image: numpy.ndarray, title: str = None):
    '''
    This function can show an image in a window
    '''
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    elif len(image.shape) == 3:
        plt.imshow(image)
    plt.axis('off')
    if title is None:
        title = "image"
    plt.title(title)
    plt.show()
    plt.clf()


def save_img(image: numpy.ndarray, name: str):
    '''
    This function saves the image without showing it
    '''
    if len(image.shape) == 3:
        image = image[:, :, (2, 1, 0)]  # -> BGR
    if name.endswith(".jpeg"):
        cv2.imwrite(name, image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100
                     ])  # 0-100, higher the number, bigger the pic
    elif name.endswith(".png"):
        cv2.imwrite(name, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0
                                  ])  # 0-9, higher the number, smaller the pic
    else:
        cv2.imwrite(name, image)


class histogram_statistic_bag:
    '''
    This class provides an instance that can count different values in ranges
    '''

    def __init__(self, bag_count: int, limit: int = 255):
        '''
        parameters:
        `bag_count`: int, the number of bags
        `limit`: int, 255 by default, the upper limit of the value
        '''
        self.bag_range = [
            n * (limit / bag_count) for n in range(0, bag_count)
        ] + [limit]
        self.bags = {n: 0 for n in range(0, bag_count)}
        self.limit = limit
        # print(self.bag_range)

    def add_to_bag(self, x):
        '''
        parameters:
        `x`: the value to be classified and counted
        '''
        for i, start_point in enumerate(self.bag_range[1:]):
            if x <= start_point:
                self.bags[i] += 1
                break
        else:
            raise ValueError(
                f"utils.histogram_statistic_bag.add_to_bag: the parameter `x` exceeds the limit {self.limit}."
            )

    def bag_values(self):
        return {self.bag_range[i]: self.bags[i] for i in range(len(self.bags))}


def average_channel_mapper(x):
    '''
    This channel mapper maps the 3 colors to their average to convert the image to gray color,
    different result from the opencv's algorithm.
    '''
    x = x.astype(numpy.uint16)
    x = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2]) / 3
    # The `astype` here is grounding
    return x.astype(numpy.uint8)


def channel_map(image: numpy.ndarray, channel: str) -> numpy.ndarray:
    '''
    parameters:
    `image`: the image
    `channel`: "red", "green", "blue", "average" or "gray", which color channel to count
    '''
    channel_mappers = {
        "red": lambda x: x[:, :, 0],
        "green": lambda x: x[:, :, 1],
        "blue": lambda x: x[:, :, 2],
        "average": average_channel_mapper,
        "gray": lambda x: cv2.cvtColor(x[:, :, (2, 1, 0)], cv2.COLOR_BGR2GRAY)
    }

    try:
        channel_mapper = channel_mappers[channel]
    except KeyError as e:
        raise ValueError(
            f"utils.channel_map: unknown value {channel} for parameter `channel`, which should be one of {list(channel_mappers.keys())}"
        )

    # image shape: H,W,C
    image = channel_mapper(image)
    return image


def histogram_statistic(image: numpy.ndarray,
                        channel: str,
                        bag_count: int,
                        limit: int = 255) -> Dict[float, int]:
    '''
    This function counts the number of pixels in different ranges

    parameters:
    `image`: the image
    `channel`: "red","green","blue", "average" or "gray", which color channel to count
    `bag_count`: how many bags to use
    `limit`: the upper limit of pixel value
    '''

    image = channel_map(image, channel)
    bags = histogram_statistic_bag(bag_count, limit)

    for line in image:
        for pixel in line:
            bags.add_to_bag(pixel)

    return bags.bag_values()


def draw_histogram(image: numpy.ndarray,
                   channel: str,
                   bag_count: int,
                   limit: int = 255,
                   fname: str = None,
                   show: bool = False,
                   title: str = None):
    '''
    This function draws a histogram of a given image and saves it if `fname` is specified
    #! no need of calculating bin values first

    parameters:
    `image`: the image
    `channel`: "red","green","blue", "average" or "gray", which color channel to count, works only when there are channels to be selected
    `bag_count`: how many bags to use
    `limit`: the upper limit of pixel value
    `fname`: the name of the saved file
    `show`: only works when `fname` is specified, if true, show the histogram after saving it
    `title`: the title of the figure genreated, if not specified, an auto-generated one would be used
    '''
    # plt.clf()
    # plt.figure()  # create new figure
    bags = histogram_statistic_bag(bag_count, limit)
    if len(image.shape) == 3:
        image = channel_map(image, channel).reshape(-1)
    elif len(image.shape) == 2:
        image = image.reshape(-1)

    plt.hist(image, bins=bags.bag_range)  # plot

    plt.title(f"histogram of channel {channel}" if title is None else title)

    if fname is not None:
        plt.savefig(fname,
                    dpi=None,
                    facecolor="w",
                    edgecolor="w",
                    orientation="portrait",
                    papertype=None,
                    format=None,
                    transparent=True,
                    bbox_inches=None,
                    pad_inches=0.1,
                    frameon=None,
                    metadata=None)
    if fname is None or show is True:
        plt.show()
    plt.clf()
