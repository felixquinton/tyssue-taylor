from pathlib import Path
import numpy as np

import napari
from napari.util import io


"""
pip install imagecodecs
libopenjp2-7-dev
liblcms2-dev
libzstd-dev
liblz4-dev
liblzma-dev
libbz2-dev
libblosc-dev
libpng-dev
libwebp-dev
libjpeg-dev
libjxr-dev
"""


base_path = "/home/guillaume/Projets/Tyssue/Organoids/Opera pour Guillaume/"

sub_directories = [
    'MQ ORG419 96puits manipMEMBright B4 FA 20190923__2019-09-23T17_37_45-Measurement 1',
    'MQ ORG419 96puits manipMEMBright B4 PFA 20190923__2019-09-23T18_09_18-Measurement 2',
    'MQ ORG419 96puits manipMEMBright B9 PFA 20190923__2019-09-23T18_26_02-Measurement 1',
    'MQ ORG419 96puits manipMEMBright BA FA 20190923__2019-09-23T17_35_52-Measurement 2',
    'MQ ORG419 96puits manipMEMBright D10 PFA 20190923__2019-09-23T18_39_48-Measurement 1',
    ]

image_dirs = [Path(base_path)/sub_d/"Images/" for sub_d in sub_directories]


# 1 : not a closed organoid



image_dir = image_dirs[3]

all_images = list(image_dir.glob("*.tiff"))

channels = {i : [im for im in all_images
                 if f"ch{i}" in im.as_posix()]
            for i in (1, 2, 3)}

for v in channels.values():
    v.sort()

stacks = {i: io.magic_imread(channels[i], stack=True) for i in (1, 2 ,3)}

with napari.gui_qt():
    viewer = napari.Viewer()
    for i, stack in stacks.items():
        viewer.add_image(stack, name=f"channel {i}")
