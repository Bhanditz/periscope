#!/usr/bin/env python3

import os
import posixpath
import itertools
import urllib.parse
import numpy
import scipy.ndimage
import base64
from io import BytesIO
from http.server import SimpleHTTPRequestHandler, HTTPServer
from templet import templet
from model import Model
from PIL import Image

class TextMap:
  def __init__(self, filename, trim=0):
    self._data = {}
    self._inv = {}
    self._keys = []
    for line in open(filename, 'r').readlines():
      key, val = line.split()
      key = key[trim:]
      val = int(val)
      self._data[key] = val
      self._inv[val] = key
      self._keys.append(key)
  def name(self, val):
    return self._inv[val]
  def num(self, key):
    return self._data[key]
  def names(self):
    return self._keys

class MiniPlacesData:
  def __init__(self, devkit):
    self._categories = TextMap(os.path.join(devkit, 'categories.txt'), trim=3)
    self._train = TextMap(os.path.join(devkit, 'train.txt'))
    self._val = TextMap(os.path.join(devkit, 'val.txt'))
  def label(self, filename):
    if filename.startswith('val'):
      cat = self._val.num(filename)
    else:
      cat = self._train.num(filename)
    return cat
  def catname(self, cat):
    return self._categories.name(cat)

miniplaces = MiniPlacesData('mp-dev_kit')
models = {
  'see1': Model('see', 'exp-see-1/epoch-024.mdl'),
  'see2': Model('see', 'exp-see-2/epoch-028.mdl'),
}

# force a compile on startup
# debug_fn = models[0].debug_fn()

class Classification:
  def __init__(self, model, imgpath):
    self.model = model
    path = os.path.join('mp-data/images', imgpath)
    self.image = numpy.transpose(
            scipy.ndimage.imread(path), [2, 0, 1]) / 255.0
    results = model.debug_fn()(numpy.expand_dims(self.image, axis=0))
    self.results = []
    for i, layer in enumerate(model.named_layers()):
        self.results.append((layer, results[i]))

class ClassificationCache:
  def __init__(self):
    self._cache = {}

  def lookup(self, modelname, imgpath):
    if imgpath not in self._cache:
      self._cache[(modelname, imgpath)] = Classification(models[modelname], imgpath)  
    return self._cache[(modelname, imgpath)]

cache = ClassificationCache()

# HTTPRequestHandler class
class PeriscopeRequestHandler(SimpleHTTPRequestHandler):
  def do_GET(self):
    if self.path.startswith('/info/'):
      return self.do_info()
    super().do_GET()

  def do_info(self):
    [_, _, modelname, imgpath] = self.path.split('/', 3)
    label = miniplaces.label(imgpath)
    cat = miniplaces.catname(label)
    cl = cache.lookup(modelname, imgpath)
    
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()    
    self.wfile.write(self.info_template(
        imgpath, label, cat, cl).encode('utf-8'))

  @templet
  def info_template(self, imgpath, label, cat, cl):
    """\
    <html>
    <style>
    img { image-rendering: pixelated; margin:1px; }
    </style>
    <body>
    <img src="/img/$imgpath">
    <p>Category: $cat
    <p>Layers
    ${[self.raw_layer_template(layer, result) for layer, result in cl.results]}
    </body>
    </html>
    """

  @templet
  def raw_layer_template(self, layer, result):
    """\
    <p>${layer.name}
    <p>Shape: ${result.shape}
    <p>${{
      if layer.name == 'softmax':
        out.append('<p>')
        out.append(numpy.array_str(layer.W.get_value()))
        out.append('<p>')
        sorted = []
        for i in range(result.shape[1]):
          cat = miniplaces.catname(i)
          sorted.append((result[0, i], cat))
        sorted.sort()
        out.extend(['{}: {}<br>'.format(cat, r) for (r, cat) in sorted])
      elif layer.name == 'input':
        for i in range(result.shape[1]):
          data = result[0, i]
          out.append(self.image_for_array(data, cmin=0, cmax=1,
               title="{}_{}".format(layer.name, i)))
      elif len(result.shape) == 4 and result.shape[2] > 1:
        out.append('<p>Prelu alpha mean: ')
        out.append(str(numpy.mean(layer.prelu.alpha.get_value())))
        out.append('<br>')
        out.append('<p>BN mean and inv_std: ')
        out.append(str(numpy.mean(layer.bn.mean.get_value())))
        out.append(', ')
        out.append(str(numpy.mean(layer.bn.inv_std.get_value())))
        out.append('<br>')
        for i in range(result.shape[1]):
          data = result[0, i]
          data = (data - layer.bn.mean.get_value()[i]) * layer.bn.inv_std.get_value()[i]
          # out.append("<p>{}_{}<br>".format(layer.name, i))
          # out.append("Std: {}<br>".format(numpy.std(data)))
          # out.append("Min: {}<br>".format(numpy.min(data)))
          # out.append("Mean: {}<br>".format(numpy.mean(data)))
          # out.append("Max: {}<br>".format(numpy.max(data)))
          out.append(self.image_for_array(data,
               title="{}_{}".format(layer.name, i)))
      else:
        out.append('<p>')
        out.append(numpy.array_str(layer.W.get_value()))
        out.append(numpy.array_str(layer.prelu.alpha.get_value()))
        out.append('<p>Prelu alpha mean: ')
        out.append(str(numpy.mean(layer.prelu.alpha.get_value())))
        out.append('<p>')
        sorted = []
        result = numpy.array(result)
        for i in range(result.shape[1]):
          data = result[0, i, 0, 0]
          sorted.append((data, '{}_{}'.format(layer.name, i)))
        sorted.sort()
        out.extend(['{}: {}<br>'.format(cat, r) for (r, cat) in sorted])
    }}

    """

  def image_for_array(self, arr, cmin=0, cmax=2, cneg=-4, title=""):
    imb = scipy.misc.bytescale(arr, cmin=cmin, cmax=cmax)
    imo = scipy.misc.bytescale(arr, cmin=cmax, cmax=cmax*2)
    imn = scipy.misc.bytescale(-arr, cmin=-cmin, cmax=-cneg)
    im3 = numpy.repeat(numpy.expand_dims(imb, axis=2), 3, axis=2)
    im3[:,:,0] += imn
    im3[:,:,2] -= imo
    im = Image.frombytes('RGB', (im3.shape[1], im3.shape[0]), im3.tostring())
    # im = scipy.misc.toimage(arr, cmin=cmin, cmax=cmax)
    png_buffer = BytesIO()
    im.save(png_buffer, format="PNG")
    b64 = base64.b64encode(png_buffer.getvalue()).decode('ascii')
    return '<img height=113 src="data:img/png;base64,{}" title="{}">'.format(
        b64, title)

  def translate_path_into_dir(self, subdir, trim, path):
    path = posixpath.normpath(urllib.parse.unquote(path))
    words = path.split('/')
    words = filter(None, words)
    words = itertools.islice(words, trim, None)
    path = subdir
    for word in words:
      drive, word = os.path.splitdrive(word)
      head, word = os.path.split(word)
      if word in (os.curdir, os.pardir): continue
      path = os.path.join(path, word)
    print(path)
    return path 
 
  # Directories to serve
  def translate_path(self, path):
    if path.startswith('/img/'):
      return self.translate_path_into_dir('mp-data/images', 1, path)
    return ''
 
def run():
  print('starting server...')
 
  # Server settings
  # Choose port 8080, for port 80, which is normally used for a http server, you need root access
  server_address = ('127.0.0.1', 8081)
  httpd = HTTPServer(server_address, PeriscopeRequestHandler)
  print('running server...')
  httpd.serve_forever()
 
 
run()
