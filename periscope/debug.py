from periscope import Network
import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, ConcatLayer
from lasagne.layers import DenseLayer, InputLayer, Pool2DLayer
import numpy as np

def is_simple_layer(layer): 
    return not (isinstance(layer, Conv2DLayer) or
        isinstance(layer, DenseLayer) or
        isinstance(layer, InputLayer))

def is_trivial_layer(layer):
    if (not is_simple_layer(layer) or isinstance(layer, Pool2DLayer) or
            isinstance(layer, ConcatLayer)):
        return False
    return (hasattr(layer, 'input_layer') and
            layer.input_shape == layer.output_shape)
    
def conv_padding(pad, filter_size):
    if pad == 'full':
        return tuple(s - 1 for s in filter_size)
    if pad == 'same':
        return tuple(s // 2 for s in filter_size)
    if pad == 'valid':
        return tuple(0 for s in filter_size)
    if isinstance(pad, int):
        return tuple(pad for s in filter_size)
    assert isinstance(pad, tuple)
    return pad

def conv_stride(stride, filter_size):
    if isinstance(stride, int):
        return tuple(stride for s in filter_size)
    assert isinstance(stride, tuple)
    return stride

def receptive_field(layer, area):
    input_area = {}
    input_area[layer] = area
    all_dep_layers = lasagne.layers.get_all_layers(layer)
    input_layer = all_dep_layers[0]
    # How to calculate the receptive field with graph dependencies?
    # Since get_all_layers returns layers in topo sort order, we can
    # walk dependencies in reverse order to solve it all in linear time.
    for inside_layer in reversed(all_dep_layers[1:]):
        inputs = layer_input_area(
            inside_layer, input_area[inside_layer])
        for prev_layer, area in inputs:
            if prev_layer in input_area:
                area = max_input_area(input_area[prev_layer], area)
            input_area[prev_layer] = area
    return input_area[input_layer]

def layer_input_area(layer, area):
    # Convolutions expand the spatial field.
    if hasattr(layer, 'filter_size'):
        return (calc_conv_input_area(layer, area), )
    # Pooling expands the spatial field.
    if hasattr(layer, 'pool_size') and not hasattr(layer, 'axis'):
        return (calc_pool_input_area(layer, area), )
    # Concatenations depend on more than one input layer.
    if hasattr(layer, 'input_layers'):
        return tuple((inp, area) for inp in layer.input_layers)
    # Other operations do not alter the spatial field.
    return ((layer.input_layer, area), )
    
def max_input_area(area1, area2):
    return tuple(
        slice(min(a1.start, a2.start), max(a1.stop, a2.stop))
        for a1, a2 in zip(area1, area2))

def calc_conv_input_area(layer, area):
    input_layer = layer.input_layer
    if len(area) == 0:
       return (input_layer,
               tuple(slice(0, m) for m in input_layer.output_shape[2:]))
    pad = conv_padding(layer.pad, layer.filter_size)
    stride = conv_stride(layer.stride, layer.filter_size)
    return (input_layer, tuple(
            slice(c.start * s - p, c.stop * s + f - 1 - p) for c, s, f, p in
            zip(area, stride, layer.filter_size, pad)))

def calc_pool_input_area(layer, area):
    input_layer = layer.input_layer
    if len(area) == 0:
       return (input_layer,
               tuple(slice(0, m) for m in input_layer.output_shape[2:]))
    pad = conv_padding(layer.pad, layer.pool_size)
    stride = conv_stride(layer.stride, layer.pool_size)
    return (input_layer, tuple(
            slice(c.start * s - p, c.stop * s + f - 1 - p) for c, s, f, p in
            zip(area, stride, layer.pool_size, pad)))


class PurposeMapper:
    """
    Has the logic needed to create response visualizations for each unit
    of a network with respect to the images of a testing corpus, to answer
    the question "What for?"
    Can create a response database, which contains an array of the following:
    for each nontrivial
    """
    def __init__(self, network, corpus, kind, n=50):
        self.net = network
        self.network = network.network
        self.corpus = corpus
        self.kind = kind
        self.out_layer = self.compute_out_layers()
        self.n = n
        # Right now we collect only nonsimple layers
        self.collect = []
        for index, layer in enumerate(self.net.all_layers()):
            if index == 0:
                continue
            if not is_simple_layer(layer):
                self.collect.append((index, self.out_layer[layer]))

    def compute_out_layers(self):
        """
        Layers followed by normalization or nonlinearity layers should
        be considered together with those postprocessing layers.  This
        function computes a map from each layer to its deepest
        postprocessing layer with the same shape.
        """
        all_layers = self.net.all_layers()
        out_layer = {}
        for layer in reversed(all_layers):
            if layer not in out_layer:
                out_layer[layer] = layer
            child = out_layer[layer]
            if is_trivial_layer(layer) and layer.input_layer not in out_layer:
                out_layer[layer.input_layer] = child
        return out_layer

    def compute_prototypes(self, pretty=None):
        layers = [layer for i, layer in self.collect]
        responses = {}
        responselocs = {}
        batch_size = 256
        crop_size = self.net.crop_size
        input_set = self.corpus.batches(
            self.kind,
            batch_size=256,
            shape=crop_size)
        for layer in layers:
            sh = lasagne.layers.get_output_shape(layer)
            responses[layer] = np.zeros((sh[1], input_set.count()))
            responselocs[layer] = np.zeros(
                (sh[1], input_set.count()), dtype=np.int32)
        debug_fn = self.net.debug_fn(layers)
        # Now do the loop
        # TODO: add pretty progress output
        if pretty:
            p = pretty.progress(len(input_set))
        s = 0
        for i, (inp, lab, name) in enumerate(input_set):
            outs = debug_fn(inp)
            for j, layer in enumerate(layers):
                if len(outs[j].shape) == 4:
                    sh = outs[j].shape
                    flat = outs[j].reshape((sh[0], sh[1], sh[2] * sh[3]))
                    responses[layer][:,s:s+len(inp)] = np.transpose(
                         np.max(flat, axis=2))
                    responselocs[layer][:,s:s+len(inp)] = np.transpose(
                         np.argmax(flat, 2))
                else:
                    responses[layer][:,s:s+len(inp)] = np.transpose(
                         outs[j])
                    responselocs[layer][:,s:s+len(inp)] = 0
            if pretty:
                p.update(i + 1)
            s += batch_size
        if pretty:
            p.finish()
        self.prototypes = []
        for index, layer in self.collect:
            pro = (-responses[layer]).argsort(axis=1)[:,:self.n].astype('int32')
            arange = np.arange(len(pro))[:,None]
            self.prototypes.append(
                (index, pro, responselocs[layer][arange, pro]))

    def generate_prototype_images(self):
        pass

class Debugger:
    """
    For detailed debugging of the response of a network to a single image;
    has the functions needed to ask "why?"  E.g., why did this network
    make a mistake on this particular image?
    """
    def __init__(self, network, image):
        if len(image.shape) == 3:
           image = image[np.newaxis,:]
        self.net = network
        self.img = image
        self.debug_fn = network.debug_fn()
        activations = self.debug_fn(image)
        self.acts = dict(zip(network.all_layers(), activations))
        # todo: consider also collecting response regions

    def activation(self, layer, coord):
        return self.acts[layer][coord]

    def inside_shape(self, coord, shape):
        for c, s in zip(coord, shape):
            if c < 0 or c >= s:
                return False
        return True

    def major_nontrivial_inputs(self, layer, coord, num=10):
        parts = self.follow_nontrivial_inputs(layer, coord, num)
        results = []
        for layer, coord, amount in parts:
            while is_simple_layer(layer):
                layer, coord = self.follow_trivial_inputs(layer, coord)
            results.append((layer, coord, amount))

    def follow_trivial_inputs(self, layer, coord):
        if is_simple_layer(layer):
        # Convolutions, Dense, and Input layers are not trivial: stop here.
            return (layer, coord)
        # Follow just the maximum for a max pool layer.
        if isinstance(layer, MaxPool2DLayer):
            return self.major_maxpool_inputs(layer, coord, num)[0][:2]
        # Concatenations depend on more than one input layer.
        if isinstance(layer, ConcatLayer):
            return get_concat_input(layer, coord)[:2]
        # Other operations do just do some pass-through.
        input_layer = layer.input_layer
        return (input_layer, coord)

    def follow_nontrivial_inputs(self, layer, coord, num=10):
        """
        Given an activation in a layer, identifies the top N units in
        convolutional layers that lead directly to this activation.
        """
        # Convolutions do some processing.
        if isinstance(layer, Conv2DLayer):
            return self.major_conv_inputs(layer, coord, num)
        # Dense layers do some processing.
        if isinstance(layer, DenseLayer):
            return self.major_dense_inputs(layer, coord, num)
        # Other layers do not.
        return [(layer, coord, 1)]

    def get_concat_input(self, layer, coord):
        """
        Given a concat layer unit, returns the input unit which contributed
        to this unit, in the form [input_layer, (coord, amount)].
        """
        assert layer.axis == 1
        input_chan = coord[1:]
        for input_layer in layer.input_layers:
            ia = self.acts[input_layer]
            if input_chan < ia.shape[1]:
                in_coord = (input_chan,) + coord[1:]
                return (input_layer, in_coord, ia[in_coord])
            input_chan -= ia.shape[1]
        # The coordinate exceeded the total input layer sizes.
        assert False

    def major_maxpool_inputs(self, layer, coord, tolerance=1.0/128):
        """
        Given a max pooling layer and a specific unit, returns the top
        input unit which contributed to this unit, in the form [(coord, amount)].
        In the case of a near-tie, could return more then one coordinate.
        """
        pad = conv_padding(layer.pad, layer.pool_size)
        stride = conv_stride(layer.stride, layer.pool_size)
        channel = coord[0]
        sect = tuple(slice(c * s, c * s + f) for c, s, f in
                zip(coord[1:], stride, layer.pool_size))
        input_layer = layer.input_layer
        # Apply padding and extract the input seen in the input area.
        ia = self.acts[input_layer][channel]
        padded_input = np.pad(ia, pad, mode='constant',
                constant_values=-np.Inf)
        seen_input = padded_input[sect]
        # Find all the locations with a value within the tolerance of the max.
        cutoff = seen_input.max() * (1 - tolerance)
        top = seen_input.argwhere(seen_input >= cutoff)
        sort(top, key=lambda x: -seen_input[x])
        # Apply offsets to each position to gather final coordinates.
        result = [(
            input_layer,
            (channel, ) + tuple(t + c * s - p
                 for t, c, s, p in zip(pos, coord[1:], stride, pad)),
            seen_input[pos])
                for pos in top]

    def major_dense_inputs(self, layer, nocoord, num=10):
        """
        Given a dense layer and a specific unit, returns ths top N
        units in the input layer that contributed to this unit.
        """
        input_layer = layer.input_layer
        ia = self.acts[input_layer]
        weights = layer.W.get_value()
        contribs = ia.flatten() * weights
        top = contribs.argsort()[:num]
        coords = zip(np.unravel_index(top, ia.shape))
        return [(input_layer, c, contribs[t]) for c, t in zip(coords, top)]

    def major_conv_inputs(self, layer, coord, num=10):
        """
        Given a convolutional layer and a specific unit, returns the top
        N units in the input layer that contributed to this unit, in the form:
        [(coord, amount), (coord, amount)...].
        """
        # Get the contributions to the specific coordinate in this layer.
        inp, off, amounts = unroll_convolution(layer, coord)
        # Extract the top N contributions.
        top = amounts.argsort(axis=None)[:num]
        coords = zip(np.unravel_index(top, amounts.shape))
        result = [(tuple(c + o for c, o in zip(coord, off)), amounts[coord])
                  for coord in coords]
        # Clip the result to only include results within bounds.
        inp_shape = inp.output_shape[1:]
        return [(layer.input_layer, c, a)
                for c, a in result if self.inside_shape(c, inp_shape)]

    def unroll_convolution(self, layer, coord):
        """
        Given an RGBxhxw input image and one activation in a convolutional
        layer, returns the tensor of contributions of each weight to that
        specific unit in that specific situation.
        """
        pad = conv_padding(layer.pad, layer.filter_size)
        stride = conv_stride(layer.stride, layer.filter_size)
        sect = (slice(None),) + tuple(slice(c * s, c * s + f) for c, s, f in
                zip(coord[1:], stride, layer.filter_size))
        input_layer = layer.input_layer
        ia = self.acts[input_layer]
        padded_input = np.pad(ia, (0,) + pad, mode='constant')
        seen_input = padded_input[sect]
        weights = layer.W.get_value()
        offset = tuple(slice(c * s - p)
                for c, s, p in zip(coord[1:], stride, pad))
        return (input_layer, (0,) + offset, seen_input * weights)

