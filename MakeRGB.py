from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import numpy as np
import xml.etree.ElementTree as ET


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def get_sst_cmap():
    violet = (93 / 255.0, 0 / 255.0, 96 / 255.0)
    blue = (29 / 255.0, 0 / 255.0, 255 / 255.0)
    cyan = (0 / 255.0, 255 / 255.0, 233 / 255.0)
    # green = (0/255.0, 255/255.0, 46/255.0)
    grey = (191 / 255.0, 191 / 255.0, 191 / 255.0)

    # yellow = (255/255.0, 248/255.0, 45/255.0)
    yellow = (246 / 255.0, 255 / 255.0, 0 / 255.0)

    orange = (255 / 255.0, 182 / 255.0, 0 / 255.0)
    red = (255 / 255.0, 0 / 255.0, 0 / 255.0)
    dark_red = (128.0 / 255.0, 0.0, 0.0)

    rvb = make_colormap([violet, blue, 0.15,
                         blue, cyan, 0.475,
                         grey, grey, 0.525,
                         yellow, orange, 0.60,
                         orange, red, 0.90,
                         red, dark_red, 1.00,
                         dark_red])

    return rvb


def get_cmap_from_xml(xml_name):
    tree = ET.parse(xml_name)
    root = tree.getroot()

    N = len(root)
    cmap_array = np.zeros((N, 4), dtype=np.float)

    for n in range(0, N):
        r = float(root[n].attrib['r']) / 255.0
        g = float(root[n].attrib['g']) / 255.0
        b = float(root[n].attrib['b']) / 255.0
        cmap_array[n, :] = np.array([r, g, b, 1.0])

    cmap = ListedColormap(cmap_array)

    return cmap


# my_cmap = get_sst_cmap()
my_cmap = get_cmap_from_xml("anc_colormaps_HSL256.xml")
my_cmap.set_bad(color=(92 / 255.0, 51 / 255.0, 23 / 255.0))
