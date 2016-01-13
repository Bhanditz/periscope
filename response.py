import numpy
import scipy
import os.path
import base64
from PIL import Image
from io import BytesIO

grid_slices = [
    (slice(y, y+23), slice(x, x+23))
    for x in range(0, 112, 7) for y in range(0, 112, 7)]

# A clipping rectifier, pegging values less than t1 to 0 and more than t2 to 1.
def peg(ar, t1, t2):
    return numpy.clip((ar - t1) / (t2 - t1), 0, 1)

def response_image(img, vec):
    epsilon = 1e-6
    avg = numpy.mean(vec)
    std = numpy.std(vec)
    respmax = avg + std
    resp = numpy.clip((respmax - vec) / (2 * std + epsilon), 0, 1)
    total = numpy.zeros(img.shape)
    count = numpy.zeros(img.shape)
    for i, s in enumerate(grid_slices):
        total[(slice(None), s[0], s[1])] += resp[i]
        count[(slice(None), s[0], s[1])] += 1
    mask = total / count
    mask = peg((mask - mask.mean()) / (mask.std() + epsilon), 0, 0.5)
    masked = img * mask
    im3 = scipy.misc.bytescale(
        numpy.transpose(masked, [1, 2, 0]), cmin=0, cmax=1)
    return rgbi_image(im3)

def rgbi_image(im3, title=None):
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

class ResponseProbe:
  def __init__(self, model, imgpath):
    self.model = model
    path = os.path.join('mp-data/images', imgpath)
    self.image = numpy.transpose(
            scipy.ndimage.imread(path), [2, 0, 1]) / 255.0
    self.probe = numpy.repeat(numpy.expand_dims(self.image, axis=0),
            len(grid_slices), axis=0)
    gray = numpy.ones((3, 23, 23)) * 0.5
    for i, (sy, sx) in enumerate(grid_slices):
        self.probe[(i, slice(0, 3), sy, sx)] = gray
    results = model.debug_fn()(self.probe)
    self.results = []
    for i, layer in enumerate(model.named_layers()):
        self.results.append((layer, results[i]))

  def get_response_image(self, name, channel, y=0, x=0):
    result = [r for (layer, r) in self.results if layer.name == name][0]
    if len(result.shape) == 4:
      vec = result[:,channel,y,x]
    else:
      vec = result[:,channel]
    return response_image(self.image, vec)

