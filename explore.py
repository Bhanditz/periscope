from templet import templet
from PIL import Image
from io import BytesIO
import base64
import scipy
import numpy

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

class ExploreTemplate:
  def __init__(self, miniplaces, imgpath, label, cat, cl, rp):
    self.miniplaces = miniplaces
    self.imgpath = imgpath
    self.label = label
    self.cat = cat
    self.cl = cl
    self.rp = rp

  def result_for_layer(self, name):
    return [res for layer, res in self.cl.results if layer.name == name][0]

  def layer_named(self, name):
    return [layer for layer, res in self.cl.results if layer.name == name][0]

  @templet
  def html(self):
    """\
    <html>
    <head>
    <style>
    ${self.css()}
    </style>
    ${self.scripts()}
    <script>
    ${self.script()}
    </script>
    </head>
    <body>
    ${self.intro()}
    </body>
    </html>
    """

  @templet
  def css(self):
    """\
    body { font: 13px Arial }
    img { border: 1px solid black; image-rendering: pixelated }
    .goalmatch {
       text-align: center;
       display: inline-block;
       padding: 3px;
       background: gainsboro;
       margin: 3px;
    }
    """

  def scripts(self):
    deps = [
        'ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js'
    ]
    return ''.join([
        '<script src="https://{}"></script>\n'.format(dep) for dep in deps])

  @templet
  def script(self):
    """\
    """

  def explain_response(self, layername, activation):
    wh = "width=128 height=128"
    goals = []
    for i in range(50 if layername == 'goo8c' else 1):
        with open(self.cl.model.ri_path(
                layername, activation, i), 'rb') as gl_file:
            b64 = base64.b64encode(gl_file.read()).decode('ascii')
            gl = '<img {} src="data:img/png;base64,{}">'.format(wh, b64)
        goals.append(gl)
    if layername == 'softmax':
        unit = self.miniplaces.catname(activation)
    else:
        unit = "{}-{}".format(layername, activation)
    
    return self.explain_response_template(unit, goals,
       html_image(self.rp.get_response_image(layername, activation)))

  @templet
  def explain_response_template(self, unit, goals, match):
    """\
    <div class="goalmatch" id="$unit">
    <b>${unit}</b><br>
    match<br>
    ${match}<br>
    goal<br>
    ${'<br>'.join(goals)}
    </div
    >$
    """

  def intro(self):
    # layer = self.layer_named('softmax')
    result = self.result_for_layer('softmax')[0,:]
    goo8c = self.result_for_layer('goo8c')[0,:,0,0]
    top = (-result).argsort()[:5]
    out = [self.intro_template()]
    weights = self.layer_named('softmax').W.get_value()
    for activation in top:
        out.append(self.explain_response('softmax', activation))
        comps = numpy.multiply(weights[:,activation], goo8c)
        topcomps = (-comps).argsort()[:10]
        for j in topcomps:
            out.append(self.explain_response('goo8c', j))
    return ''.join(out)

  @templet
  def intro_template(self):
    """\
    <img src="/img/${self.imgpath}">
    Category: ${self.cat}
    <p>
    """
     
