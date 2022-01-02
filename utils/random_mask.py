"""
Source: https://newbedev.com/algorithm-to-generate-random-2d-polygon

Spiky random polygons

Speed-up is_contains using numba:
https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
"""
import math
import random

import numba
import numpy as np
from numba import njit
from numba.typed import List
from scipy import ndimage


def generate_spiky_mask(
    img_height,
    img_width,
    irregularity: float = 0.35,
    spikeyness: float = 0.2,
    num_verts: int = 12,
):
    """
    create a random polygon mask where mask = 1 is the region to be ignored/masked

    Polygon will be centered around (ctrX, ctrY) i.e image center and with average radius as img_height//4

    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius avg_radius. [0,1] will map to [0, avg_radius]
    num_verts - self-explanatory
    """
    height_mid, width_mid = img_height // 2, img_width // 2
    verts = generate_polygon(
        ctrX=width_mid,
        ctrY=height_mid,
        avg_radius=img_height // 3.5,
        irregularity=irregularity,
        spikeyness=spikeyness,
        num_verts=num_verts,
    )

    mask = np.zeros((img_width, img_height))
    # find the extremes of the polygon and only check for inside points using .contains() to reduce time complexity
    x_coord = [pt[0] for pt in verts]
    y_coord = [pt[1] for pt in verts]
    x_min, x_max = min(x_coord), max(x_coord)
    y_min, y_max = min(y_coord), max(y_coord)

    h_mesh, w_mesh = np.meshgrid(
        range(y_min, y_max), range(x_min, x_max), indexing="ij"
    )
    h_grid, w_grid = h_mesh.ravel(), w_mesh.ravel()

    mask[h_grid, w_grid] = parallel_point_in_polygon(h_grid, w_grid, List(verts))

    return mask


def generate_spiky_mask_efficiently(
    img_height: int,
    img_width: int,
    downsample_factor: int = 2,
    irregularity: float = 0.35,
    spikeyness: float = 0.2,
    num_verts: int = 12,
):
    """
    Generate mask using generate_spiky_mask at low-resolution

    Then upsample

    :param img_height:
    :param img_width:
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius avg_radius. [0,1] will map to [0, avg_radius]
    num_verts - self-explanatory
    :return:
    """
    assert img_width % downsample_factor == img_height % downsample_factor == 0
    mask_LR = generate_spiky_mask(
        img_height // downsample_factor,
        img_width // downsample_factor,
        irregularity=irregularity,
        spikeyness=spikeyness,
        num_verts=num_verts,
    )

    mask_HR = ndimage.zoom(mask_LR, zoom=downsample_factor, order=1)

    # Binarize
    mask_HR[mask_HR < 0.5] = 0
    mask_HR[mask_HR > 0.5] = 1

    return mask_HR


@njit(cache=True, fastmath=True)
def generate_polygon(ctrX, ctrY, avg_radius, irregularity, spikeyness, num_verts):
    """
    Source: https://newbedev.com/algorithm-to-generate-random-2d-polygon

    Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    avg_radius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius avg_radius. [0,1] will map to [0, avg_radius]
    num_verts - self-explanatory

    Returns a list of vertices, in CCW order.
    """

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / num_verts
    spikeyness = clip(spikeyness, 0, 1) * avg_radius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / num_verts) - irregularity
    upper = (2 * math.pi / num_verts) + irregularity
    sum = 0
    for i in numba.prange(num_verts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in numba.prange(num_verts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in numba.prange(num_verts):
        r_i = clip(random.gauss(avg_radius, spikeyness), 0, 2 * avg_radius)
        x = ctrX + r_i * math.cos(angle)
        y = ctrY + r_i * math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return points


@njit(cache=True, fastmath=True)
def clip(x, min, max):
    if min > max:
        return x
    elif x < min:
        return min
    elif x > max:
        return max
    else:
        return x


@njit(cache=True, fastmath=True)
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(cache=True, parallel=True)
def parallel_point_in_polygon(x_ll, y_ll, polygon):
    D = np.empty(len(x_ll), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = point_in_polygon(x_ll[i], y_ll[i], polygon)
    return D


if __name__ == "__main__":
    ## Spiky mask
    from utils.timer import catchtime
    from loguru import logger
    from matplotlib import pyplot as plt
    from utils.profiler import profile

    with catchtime() as t:
        mask_spiky = profile(generate_spiky_mask)(1024, 1024)

    logger.info(f"Elapsed time {t()}")
    print(mask_spiky.shape)

    plt.imshow(mask_spiky)
    plt.show()
