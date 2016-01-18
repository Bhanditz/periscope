#!/usr/bin/env python3

import os
import posixpath
import itertools
import urllib.parse
import numpy
import scipy.ndimage
import base64
import explore
import scipy
from http.server import SimpleHTTPRequestHandler, HTTPServer
from templet import templet
from model import Model
from response import ResponseProbe
from PIL import Image
from io import BytesIO

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
  'ren1': Model('ren', 'exp-ren-1/epoch-029.mdl'),
  'ren2': Model('ren', 'exp-ren-2/epoch-029.mdl'),
  'ren3': Model('ren', 'exp-ren-3/epoch-029.mdl'),
  'ren7': Model('ren', 'exp-ren-7/epoch-029.mdl'),
}


def html_image(img):
    imbytes = scipy.misc.bytescale(
        numpy.transpose(img, [1, 2, 0]), cmin=0, cmax=1)
    return html_image_from_bytes(imbytes)

def html_image_from_bytes(im3, title=None):
    if title is None:
        attr = ''
    else:
        attr = ' title="{}"'.format(title)
    height = im3.shape[1]
    while height < 64:
        height *= 2
    png_buffer = BytesIO()
    im = Image.frombytes('RGB', (im3.shape[1], im3.shape[0]), im3.tostring())
    im.save(png_buffer, format="PNG")
    b64 = base64.b64encode(png_buffer.getvalue()).decode('ascii')
    return '<img height={} src="data:img/png;base64,{}"{}>'.format(
        height, b64, attr)
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
      self._cache[(modelname, imgpath)] = Classification(
          models[modelname], imgpath)  
    return self._cache[(modelname, imgpath)]

class ResponseProbeCache:
  def __init__(self):
    self._cache = {}

  def lookup(self, modelname, imgpath):
    if imgpath not in self._cache:
      self._cache[(modelname, imgpath)] = ResponseProbe(
          models[modelname], imgpath)  
    return self._cache[(modelname, imgpath)]

classification_cache = ClassificationCache()
response_cache = ResponseProbeCache()

# HTTPRequestHandler class
class PeriscopeRequestHandler(SimpleHTTPRequestHandler):
  def do_GET(self):
    if self.path.startswith('/info/'):
      return self.do_info()
    if self.path.startswith('/explore/'):
      return self.do_explore()
    super().do_GET()

  def do_explore(self):
    [_, _, modelname, imgpath] = self.path.split('/', 3)
    label = miniplaces.label(imgpath)
    cat = miniplaces.catname(label)
    cl = classification_cache.lookup(modelname, imgpath)
    rp = response_cache.lookup(modelname, imgpath)
    template = explore.ExploreTemplate(miniplaces, imgpath, label, cat, cl, rp)

    self.results = cl.results
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()    
    self.wfile.write(template.html().encode('utf-8'))

  def do_info(self):
    [_, _, modelname, imgpath] = self.path.split('/', 3)
    label = miniplaces.label(imgpath)
    cat = miniplaces.catname(label)
    cl = classification_cache.lookup(modelname, imgpath)
    rp = response_cache.lookup(modelname, imgpath)

    self.results = cl.results
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()    
    self.wfile.write(self.info_template(
        imgpath, label, cat, cl, rp).encode('utf-8'))

  def result_for_layer(self, name):
    return [result for layer, result in self.results if layer.name == name][0]

  @templet
  def info_template(self, imgpath, label, cat, cl, rp):
    """\
    <html>
    <style>
    img { image-rendering: pixelated; margin:1px; }
    </style>
    <body>
    <img src="/img/$imgpath">
    <p>Category: $cat
    <p>Layers
    ${[self.layer_template(layer, result, rp, cl)
         for layer, result in cl.results]}
    </body>
    </html>
    """

  @templet
  def layer_template(self, layer, result, rp, cl):
    """\
    <p>${layer.name}
    <p>Shape: ${result.shape}
    ${self.layer_details(layer, result, rp, cl)}
    """

  def layer_details(self, layer, result, rp, cl):
    if layer.name == 'softmax':
      return self.softmax_details(layer, result, rp, cl)
    elif layer.name == 'input':
      return self.input_details(layer, result)
    elif len(result.shape) == 4 and result.shape[2] > 1:
      if hasattr(layer, 'prelu') and hasattr(layer, 'bn'):
        return self.conv_details(layer, result)
      return '<p>' + self.conv_images(layer, result)
    else:
      if hasattr(layer, 'prelu') and hasattr(layer, 'bn'):
        return self.fc_details(layer, result)
      elif layer.name.endswith('c'):
        return self.response_details(layer, result, rp)
    return '<p>' + layer.name

  @templet
  def softmax_details(self, layer, result, rp, cl):
    """\
    <p>W shape ${layer.W.get_value().shape},
    <br>Average |W|: ${abs(layer.W.get_value()).mean()};
    average W^2: ${(layer.W.get_value() ** 2).mean()}<br>
    <p>
    ${{
      input_layer = layer.input_layer
      goo8c = self.result_for_layer('goo8c')[0,:,0,0]
      weights = layer.W.get_value()
      biases = layer.b.get_value()
      sorted = []
      for i in range(result.shape[1]):
        cat = miniplaces.catname(i)
        comps = numpy.multiply(weights[:,i], goo8c)
        pieces = []
        for j in range(comps.shape[0]):
          pieces.append((comps[j], j, 'black' if goo8c[j] > 0 else 'red'))
        pieces.sort()
        pieces.reverse()
        pieces = pieces[:20]
        numbers = '; '.join(['<span style="color:{}">c{}: {}</span>'.format(a, j, c) for c, j, a in pieces])
        component_ri = ''.join([self.explain_response('goo8c', j, rp, cl) for c, j, a in pieces])
        ri = html_image(rp.get_response_image('softmax', i))
        sorted.append(
            (result[0, i], cat, numbers, ri, component_ri))
      sorted.sort()
      sorted = sorted[-5:]
      out.extend(['{}: {} {}<br>{}{}<br>'.format(cat, r, n, im, ri) for (r, cat, n, im, ri) in sorted])
    }}
    """

  def explain_response(self, layername, activation, rp, cl):
    with open(cl.model.ri_path(layername, activation), 'rb') as ri_file:
        b64 = base64.b64encode(ri_file.read()).decode('ascii')
        ri = '<img height={} src="data:img/png;base64,{}">'.format(128, b64)
    return ri + html_image(rp.get_response_image(layername, activation))

  def input_details(self, layer, result):
    out = ['<p>']
    for i in range(result.shape[1]):
      data = result[0, i]
      out.append(self.image_for_array(data, cmin=0, cmax=1,
          title="{}_{}".format(layer.name, i)))
    return ''.join(out)

  @templet
  def conv_details(self, layer, result):
    """\
    <p>Average |W|: ${abs(layer.W.get_value()).mean()};
    average W^2: ${(layer.W.get_value() ** 2).mean()}<br>
    Prelu alpha mean: ${numpy.mean(layer.prelu.alpha.get_value())}<br>
    BN mean: ${numpy.mean(layer.bn.mean.get_value())},
    BN inv_std ${numpy.mean(layer.bn.inv_std.get_value())}<br>
    ${self.conv_images(layer, result)}
    """

  @templet
  def conv_images(self, layer, result):
    """\
    ${{
      result = numpy.array(result)
      out.append('First image shape {}<br>'.format(result[0, 0].shape))
      out.append('First pixel {}<br>'.format(result[0, 0, 0, 0]))
      if hasattr(layer, 'bn'):
        out.append('First pix bnm{} bnis{} normed{}<br>'.format(
            layer.bn.mean.get_value()[0],
            layer.bn.inv_std.get_value()[0],
            (result[0, 0, 0, 0] -
            layer.bn.mean.get_value()[0]) *
            layer.bn.inv_std.get_value()[0]))
      for i in range(result.shape[1]):
        data = result[0, i]
        if hasattr(layer, 'bn'):
          data = data - layer.bn.mean.get_value()[i]
          data = data * layer.bn.inv_std.get_value()[i]
        out.append(self.image_for_array(data,
               title="{}_{}".format(layer.name, i)))
    }}
    """

  @templet
  def fc_details(self, layer, result):
    """\
    <p>Average |W|: ${abs(layer.W.get_value()).mean()};
    average W^2: ${(layer.W.get_value() ** 2).mean()}<br>
    ${numpy.array_str(layer.prelu.alpha.get_value())}
    <p>Prelu alpha mean: ${numpy.mean(layer.prelu.alpha.get_value())}<br>
    BN mean: ${numpy.mean(layer.bn.mean.get_value())},
    BN inv_std ${numpy.mean(layer.bn.inv_std.get_value())}<br>
    ${{
      sorted = []
      result = numpy.array(result)
      for i in range(result.shape[1]):
        data = result[0, i, 0, 0]
        sorted.append((data, '{}_{}'.format(layer.name, i)))
      sorted.sort()
      out.extend(['{}: {}<br>'.format(cat, r) for (r, cat) in sorted])
    }}
    """

  def response_details(self, layer, result, rp):
      index = numpy.argsort(result.flatten())
      return ''.join([self.one_response_image(layer, result, rp, index[j])
          for j in range(len(index))])

  @templet
  def one_response_image(self, layer, result, rp, i):
    """\
    ${html_image(rp.get_response_image(layer.name, i))} ${i}: ${result.flatten()[i]}<br>
    """

  def image_for_array(self, arr, cmin=0, cmax=2, cneg=-4, title=""):
    imb = scipy.misc.bytescale(arr, cmin=cmin, cmax=cmax)
    imo = scipy.misc.bytescale(arr, cmin=cmax, cmax=cmax*2)
    imn = scipy.misc.bytescale(-arr, cmin=-cmin, cmax=-cneg)
    im3 = numpy.repeat(numpy.expand_dims(imb, axis=2), 3, axis=2)
    im3[:,:,2] += imn
    im3[:,:,2] -= imo
    return html_image_from_bytes(im3)

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
