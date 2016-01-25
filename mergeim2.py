#!/usr/bin/env python
import sys
import os
import re
from PIL import Image, ImageFont, ImageDraw

font = ImageFont.load_default()
done = set()
dirname = sys.argv[1]

layer = 'goo8c'
units = 128*14
examples = 4
dims = 128

for unit in range(0, units, 128):
    tiled = Image.new('RGB', ((examples + 1) * (dims + 1), 128 * (dims + 1)),
            (255, 255, 255))
    draw = ImageDraw.Draw(tiled)
    for u in range(0, min(128, units - unit)):
        draw.text((50, 50 + u * (dims + 1)), "{}_{}".format(layer, unit + u),
                  (0,0,0), font=font)
        for ex in range(examples):
            fn = os.path.join(dirname, '{}_{}.{}.png'.format(layer, unit+u, ex))
            im = Image.open(fn)
            tiled.paste(im, ((ex + 1) * (dims + 1), u * (dims + 1)))
            im.close()
        print("added {}_{}".format(layer, unit + u))
    filename = "{}_u{}-{}.jpg".format(layer, unit, unit+127)
    print("saving {}".format(filename))
    tiled.save(os.path.join(dirname, filename), 'JPEG')
