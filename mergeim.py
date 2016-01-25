#!/usr/bin/env python
import sys
import os
import re
from PIL import Image

done = set()
dirname = sys.argv[1]
files = [fn for fn in os.listdir(dirname)
         if re.match(r'.*_\d+\.\d+\.png$', fn) and fn.startswith('goo8c')]
for fn in files:
    m = re.match(r'(.*_\d+\.)\d+\.png$', fn)
    if not m or m.group(1) in done:
        continue
    prefix = m.group(1)
    targetname = prefix + 'all.png'
    collect = [fn for fn in files if fn.startswith(prefix)]
    collect.sort(key=lambda x: int(re.match(r'.*\.(\d+)\.png', x).group(1)))
    ims = []
    for fn in collect:
       im = Image.open(os.path.join(dirname, fn))
       ims.append(im.copy())
       im.close()
    width = sum([im.size[0] + 1 for im in ims]) - 1
    height = max([im.size[1] for im in ims])
    x = 0
    tiled = Image.new('RGB', (width, height), (255, 255, 255))
    for im in ims:
        tiled.paste(im, (x, 0))
        x += im.size[0] + 1
    tiled.save(os.path.join(dirname, targetname), 'PNG')
    print('created {}x{} image {}'.format(width, height, targetname))
    done.add(prefix)
