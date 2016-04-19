from periscope import Network
from periscope import load_from_checkpoint
import lasagne
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, ConcatLayer
from lasagne.layers import DenseLayer, InputLayer, Pool2DLayer, DropoutLayer
from PIL import Image
import numpy as np
import pickle
import copy
import os

def is_dendritic_layer(layer):
    """
    Identifies "thinking" layers that add up inputs.
    """
    return (isinstance(layer, Conv2DLayer) or
        isinstance(layer, DenseLayer) or
        isinstance(layer, InputLayer))

def is_simple_layer(layer):
    """
    Identifies "plumbing" layers.
    """
    return not is_dendritic_layer(layer)

def is_trivial_layer(layer):
    """
    Identifies scaling layers like batchnorm, nonlinearity, etc.
    """
    if (not is_simple_layer(layer) or isinstance(layer, Pool2DLayer) or
            isinstance(layer, ConcatLayer)):
        return False
    return (hasattr(layer, 'input_layer') and
            layer.input_shape == layer.output_shape)

def compute_out_layers(net):
    """
    Layers followed by normalization or nonlinearity layers should
    be considered together with those postprocessing layers.  This
    function computes a map from each layer to its deepest
    postprocessing layer with the same shape.
    """
    all_layers = net.all_layers()
    out_layer = {}
    for layer in reversed(all_layers):
        if layer not in out_layer:
            out_layer[layer] = layer
        child = out_layer[layer]
        if is_trivial_layer(layer) and layer.input_layer not in out_layer:
            out_layer[layer.input_layer] = child
    return out_layer

def layer_inputs(layer):
    if hasattr(layer, 'input_layers'):
        return layer.input_layers
    if hasattr(layer, 'input_layer'):
        return [layer.input_layer]
    return []

def compute_axial_maps(net):
    """
    Axial layers are the ones whose activations are directly picked up
    by dendritic layers.  Dropping an axon means dropping weights in
    dendritic layers fed by the axon, and dropping units in the dendritic
    layers that feed the axon.

    This computes two maps (axial_endpoint, axial_synapse).
      - axial_endpoint[layer] maps a layer to the axial endpoint layers
        to which it eventually outputs, plus an offset for each. If a
        ConcatLayer is in the axon, unit #n will be shifted to unit
        #n+offset at the endpoint.  In the case of multiple dependencies,
        there may be more than one axial endpoint.
      - axial_synapse[endpoint] maps to an array of dendritic layers
        which take their input from the given axial endpoint.
    """
    # Go through all layers, output first.
    all_layers = net.all_layers()
    axial_synapse = {}
    axial_endpoint = {}
    for layer in reversed(all_layers):
        if layer not in axial_endpoint:
            axial_endpoint[layer] = [{'layer': layer, 'offset': 0, 'par': []}]
            axial_synapse[layer] = []
        if is_dendritic_layer(layer):
            # If it's a dendritic layer, then its inputs are axial layers.
            for in_layer in layer_inputs(layer):
                # Remember axial info.
                if in_layer not in axial_synapse:
                    axial_synapse[in_layer] = []
                axial_synapse[in_layer].append(layer)
                # Remember axial layer
                if in_layer not in axial_endpoint:
                    axial_endpoint[in_layer] = []
                axial_endpoint[in_layer].append(
                    {'layer': in_layer, 'offset': 0, 'par': []})
        else:
            # Propagate axons backwords to the attached dendritic layer.
            # TODO: we must also track batchnorm layers along the way
            axons = axial_endpoint[layer]
            if len(layer.params):
               for a in axons:
                   a['par'].append(layer)
            offset = 0
            for in_layer in layer_inputs(layer):
                if in_layer not in axial_endpoint:
                    axial_endpoint[in_layer] = []
                axial_endpoint[in_layer].extend([
                    {'layer': a['layer'], 'offset': a['offset'] + offset,
                     'par': a['par']}
                    for a in axons])
                # If it's a concat layer, then figure its input offsets.
                if isinstance(layer, ConcatLayer):
                    offset += lasagne.layers.get_output_shape(in_layer)[1]
                # TODO: consider the effect of shape layers - not handled.
    return (axial_endpoint, axial_synapse)

def compute_next_layers(net):
    out_layer = compute_out_layers(net)

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
    # Padding shifts the visual field
    if hasattr(layer, 'pad'):
        return (calc_pad_input_area(layer, area), )
    # Other operations do not alter the spatial field.
    return ((layer.input_layer, area), )

def max_input_area(area1, area2):
    return tuple(
        slice(min(a1.start, a2.start), max(a1.stop, a2.stop))
        for a1, a2 in zip(area1, area2))

def calc_pad_input_area(layer, area):
    input_layer = layer.input_layer
    if len(area) == 0:
       return (input_layer,
               tuple(slice(0, m) for m in input_layer.output_shape[2:]))
    pad = layer.pad
    return (input_layer, tuple(
            slice(c.start - p, c.stop - p) for c, p in
            zip(area, pad)))

def calc_conv_input_area(layer, area):
    input_layer = layer.input_layer
    if len(area) == 0:
       return (input_layer,
               tuple(slice(0, m) for m in input_layer.output_shape[2:]))
    pad = conv_padding(layer.pad, layer.filter_size)
    stride = conv_stride(layer.stride, layer.filter_size)
    return (input_layer, tuple(
            slice(c.start * s - p, (c.stop - 1) * s + f - p) for c, s, f, p in
            zip(area, stride, layer.filter_size, pad)))

def calc_pool_input_area(layer, area):
    input_layer = layer.input_layer
    if len(area) == 0:
       return (input_layer,
               tuple(slice(0, m) for m in input_layer.output_shape[2:]))
    pad = conv_padding(layer.pad, layer.pool_size)
    stride = conv_stride(layer.stride, layer.pool_size)
    return (input_layer, tuple(
            slice(c.start * s - p, (c.stop - 1) * s + f - p) for c, s, f, p in
            zip(area, stride, layer.pool_size, pad)))

def padslice(arr, sect, fill=0):
    """
    Given a tuple of slices sect which may have out-of-bound ranges,
    returns an array of exactly the shape requested, with any data
    beyond the boundaries padded with a fill value, defaulting to zero.
    """
    size = arr.shape
    if (np.shape(fill) == arr.shape[0]):
        fill = fill[(slice(None),) + (None,) * (len(arr.shape) - 1)]
    src = tuple(slice(max(0, s.start), min(m, s.stop))
        for m, s in zip(size, sect))
    tar = tuple(slice(r.start - s.start, r.stop - s.start)
        for r, s in zip(src, sect))
    result = np.ones(tuple(s.stop - s.start for s in sect)) * fill
    result[tar] = arr[src]
    return result

def safe_unravel(index, shape):
    """
    Just like unravel index, but happy to return a degenerate zero-length
    location for a zero-dimensional shape.
    """
    if len(shape) == 0:
        return ()
    return np.unravel_index(index, shape)

class PurposeMapper:
    """
    Has the logic needed to create response visualizations for each unit
    of a network with respect to the images of a testing corpus, to answer
    the question "What for?"

    Can create, save, load, or visualize a top response database, which
    contains an array of the following:

       [(layer_index, prototype_indexes, prototype_locations)]

    layer_index - which layer, as in network.all_layers()[layer_index]
    prototype_indexes - numpy array(channels, N), to give the top N
        examples of each channel activation.  i = prototype_index[c, x]
        selects the xth example of a top activation for channel c, so
        that corpus.get(i) returns the example image.
    prototype_locations - numpy array(channels, N), to give the N
        locations corresponding to the prototype images above. These
        are flattened locations within the specific activation layer, so
        to get the original activation x, y, so for spatial layers we can
        recover location: xy = np.unravel_index(i, layer.output_shape[2:])
    """
    def __init__(self, network, corpus, kind='val', n=50):
        self.net = network
        self.network = network.network
        self.corpus = corpus
        self.kind = kind
        self.out_layer = compute_out_layers(network)
        self.n = n
        # Right now we collect only nonsimple layers
        self.collect = []
        for index, layer in enumerate(self.net.all_layers()):
            if index == 0:
                continue
            if not is_simple_layer(layer):
                self.collect.append((index, self.out_layer[layer]))
        self.prototypes = None

    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.net.checkpoint.directory, 'purpose.db')
        with open(filename, 'wb+') as f:
            f.seek(0)
            f.truncate()
            formatver = 1
            pickle.dump(formatver, f)
            pickle.dump(self.prototypes, f)

    def load(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.net.checkpoint.directory, 'purpose.db')
        with open(filename, 'rb') as f:
            f.seek(0)
            formatver = 1
            formatver = pickle.load(f)
            self.prototypes = pickle.load(f)

    def exists(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.net.checkpoint.directory, 'purpose.db')
        return os.path.isfile(filename)

    def extract_image_section(self,
            layer_index, prototype_index, prototype_loc,
            fill=0):
        layer = self.net.all_layers()[layer_index]
        img, label, name = self.corpus.get(
                self.kind, prototype_index, shape=self.net.crop_size)
        coord_loc = safe_unravel(prototype_loc, layer.output_shape[2:])
        sect = receptive_field(layer, tuple(slice(i, i+1) for i in coord_loc))
        return padslice(img, ((slice(0, img.shape[0]), ) + sect), fill=fill)

    def save_filmstrip_images(
            self, directory=None, blockheight=1, blockwidth=None,
            groupsize=32, pretty=None):
        if pretty:
            pretty.task('Computing {}x{} filmstrip images for {}'.format(
                blockheight, blockwidth or '', self.net.__class__.__name__))
        if directory is None:
            directory = os.path.join(
                self.net.checkpoint.directory,
                'purpose', 'f{}'.format(blockheight))
        os.makedirs(directory, exist_ok=True)
        if pretty:
            total = -sum((-len(im) // groupsize
                    for i, im, l in self.prototypes))
            p = pretty.progress(total)
            total = 0
        for index, im, loc in self.prototypes:
            for start in range(0, len(im), groupsize):
                stop = min(len(im), start + groupsize)
                pil_im = self.make_filmstrip(
                    index,
                    unit=range(start, min(len(im), start + groupsize)),
                    blockheight=blockheight, blockwidth=blockwidth)
                fname = "l{}_u{}.jpg".format(index, start)
                # Use lossy jpg for 10x image size savings, but
                # save small images at full quality, to avoid loss of
                # detailed color information for small convolutions.
                if pil_im.size[0] * pil_im.size[1] < 640 ** 2:
                    opts = { 'subsampling': 0, 'quality': 99 }
                else:
                    opts = {}
                pil_im.save(os.path.join(directory, fname), 'JPEG', **opts)
                if pretty:
                    total += 1
                    p.update(total)
        if pretty:
            p.finish()

    def make_filmstrip(self,
            layer, unit=None, blockwidth=None, blockheight=1,
            background='white', margin=1, fill=0):
        # Grab the prototypes array for the requested layer.
        all_layers = self.net.all_layers()
        if isinstance(layer, int):
            prot = [p for p in self.prototypes if p[0] == layer][0]
        elif isinstance(layer, tuple):
            prot = layer
        else:
            prot = [p for p in self.prototypes if all_layers[p[0]] == layer][0]
        layer_index, prototype_images, prototype_locations = prot
        layer = all_layers[layer_index]
        shape = layer.output_shape
        if unit is None:
            unit = range(shape[0])
        if isinstance(unit, int):
            unit = [unit]
        if blockwidth is None:
            blockwidth = self.n // blockheight
        # Compute a single example receptive field
        sect = receptive_field(
            layer, tuple(slice(0, 1) for i in shape[2:]))
        ri_shape = tuple(s.stop - s.start for s in sect)
        im_shape = self.net.crop_size
        if all(r >= i for r, i in zip(ri_shape, im_shape)):
            ri_shape = im_shape
        unitcount = len(unit)
        im = Image.new('RGB',
            ((ri_shape[1] + margin) * blockwidth - margin,
             (ri_shape[0] + margin) * blockheight * unitcount - margin),
            background)
        # Loop throught every selected unit, and paste a block of
        # top response areas from top response images.
        for i, u in enumerate(unit):
            index = 0
            for r in range(blockheight):
                for c in range(blockwidth):
                    pro_im = prototype_images[u, index]
                    pro_loc = prototype_locations[u, index]
                    if ri_shape == im_shape:
                        imarr, label, name = self.corpus.get(
                            self.kind, pro_im, shape=im_shape)
                    else:
                        imarr = self.extract_image_section(
                                layer_index, pro_im, pro_loc, fill=fill)
                    data = (imarr + 128).astype(
                            np.uint8).transpose((1, 2, 0)).tostring()
                    one_image = Image.frombytes('RGB',
                            (imarr.shape[2], imarr.shape[1]), data)
                    im.paste(one_image, (c * (ri_shape[1] + margin),
                            (r + (i * blockheight)) * (ri_shape[0] + margin)))
                    index += 1
        return im

    def compute(self, pretty=None):
        if pretty:
            pretty.task('Computing purpose database for {}'.format(
                self.net.__class__.__name__))
        layers = [layer for i, layer in self.collect]
        responses = {}
        responselocs = {}
        batch_size = 256
        crop_size = self.net.crop_size
        input_set = self.corpus.batches(
            self.kind,
            batch_size=batch_size,
            shape=crop_size)
        for layer in layers:
            sh = lasagne.layers.get_output_shape(layer)
            responses[layer] = np.zeros((sh[1], input_set.count()))
            responselocs[layer] = np.zeros(
                (sh[1], input_set.count()), dtype=np.int32)
        if pretty:
            pretty.subtask('Compiling debug function.')
        debug_fn = self.net.debug_fn(layers)
        # Now do the loop
        if pretty:
            p = pretty.progress(len(input_set))
        s = 0
        for i, (inp, lab, name) in enumerate(input_set):
            outs = debug_fn(inp)
            for j, layer in enumerate(layers):
                if len(outs[j].shape) == 4:
                    sh = outs[j].shape
                    # TODO: consider adding an option for a gaussian blur here
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
        # TODO: write this function.  It should generate filmstrips with
        # blacked out regions (or tranparency!) indicating where the
        # activations occurred.  It should run the network over the input
        # a second time, but avoid re-running the network over inputs.
        # collect together the set of all prototype inputs
        samples = set()
        layers = self.net.all_layers()
        layerindex = {}
        for (index, pro, loc) in self.prototypes:
            samples.update(pro.tolist())
            for p in pro:
                if p not in layerindex:
                    layerindex[p] = []
                # Accumulate jobs to do for input p:
                # create an image at location "loc" for the given layer.
                layerindex[p].append((self.out_layer[layers[index]], loc))
        samples = sorted(s)
        # Next step: compute receptive field shapes for each layer
        # Then iterate through every corpus sample in "samples"
        # For each sample, go through the list of jobs
        # For each job, grab the receptive field size and the ouput array.
        # For XY output:
        #   Map the activations back to input location centers.
        #   Apply a gaussian blur based on this RF size.
        #   Compute a cutoff.
        #   Clip out the rectangular receptive field (TODO: consider edging in)
        # Save the data away.
        pass

class ActivationSample:
    """
    Creates, saves, and loads an activation vector database for every
    axial endpoint unit, on a small sample of the training set.

    The self.activations database is an array
        [(layernum, activationmatrix), ...]

    layernum is the index of the relevant layer in the all_layers array.
    activationmatrix has 4096 (number of samples) rows and one column for
        each unit in the layer.

    One activation is collected per hidden unit per input image. Within
    convolutional layers, a single random X-Y coordinate is sampled (the
    location is different for each input image, but same for all units
    within a layer).

    The sampled activation is after nonlinearities and normalization:
    it is the number used as input to the next layer.  The purpose of
    the activation db is to compute approximations of behavior, and
    the observed behavior of a unit is how the next layer sees it.
    """
    def __init__(self, network, corpus,
                 kind='train', n=4096, empty=False, force=False, pretty=None):
        self.net = network
        self.network = network.network
        self.corpus = corpus
        self.kind = kind
        (axial_endpoints, axial_synapse) = compute_axial_maps(network)
        self.n = n
        # Right now we collect only nonsimple layers
        self.collect = []
        for layer in axial_synapse.keys():
            index = self.net.layer_number(layer)
            self.collect.append((index, layer))
        self.activations = None
        if empty:
            return
        if force or not self.exists():
            self.compute(pretty=pretty)
            self.save()
        else:
            self.load()

    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.net.checkpoint.directory, 'activation.db')
        with open(filename, 'wb+') as f:
            f.seek(0)
            f.truncate()
            formatver = 1
            pickle.dump(formatver, f)
            pickle.dump(self.activations, f)

    def load(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.net.checkpoint.directory, 'activation.db')
        with open(filename, 'rb') as f:
            f.seek(0)
            formatver = 1
            formatver = pickle.load(f)
            self.activations = pickle.load(f)

    def layernum(self, num):
        for (index, act) in self.activations:
            if index == num:
                return act
        return None

    def exists(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.net.checkpoint.directory, 'activation.db')
        return os.path.isfile(filename)

    def compute(self, pretty=None):
        if pretty:
            pretty.task('Computing activation database for {}'.format(
                self.net.__class__.__name__))
        layers = [layer for i, layer in self.collect]
        activations = {}
        rng = np.random.RandomState(1)
        batch_size = 256
        crop_size = self.net.crop_size
        input_set = self.corpus.batches(
            self.kind,
            batch_size=batch_size,
            shape=crop_size,
            limit=self.n)
        for layer in layers:
            sh = lasagne.layers.get_output_shape(layer)
            activations[layer] = np.zeros((self.n, sh[1]))
        if pretty:
            pretty.subtask('Compiling debug function.')
        debug_fn = self.net.debug_fn(layers)
        # Now do the loop
        if pretty:
            p = pretty.progress(len(input_set))
        s = 0
        for i, (inp, lab, name) in enumerate(input_set):
            outs = debug_fn(inp)
            for j, layer in enumerate(layers):
                if len(outs[j].shape) == 4:
                    # Remembering all the activations is too much data;
                    # so we just sample one geometric location per input.
                    # For each individual input, we will select a random
                    # location and sample that same location for all the
                    # units in this layer.
                    samp = (np.arange(len(inp)), slice(None)) + tuple(
                        rng.randint(0, m, len(inp)) for m in outs[j].shape[2:])
                    activations[layer][s:s+len(inp),:] = outs[j][samp]
                else:
                    activations[layer][s:s+len(inp),:] = outs[j]
            if pretty:
                p.update(i + 1)
            s += batch_size
        if pretty:
            p.finish()
        self.activations = []
        for index, layer in self.collect:
            self.activations.append((index, activations[layer]))

class NetworkMerger:
    def __init__(self, nets, corpus):
        self.nets = nets
        self.corpus = corpus

    def create_merged_network(self):
        # For now, we just combine full layers
        # Starting from the bottom, pile one network into the other
        # so we can run as a double-wide single network, summing results
        # right at the softmax/output layer.
        result = copy.deepcopy(self.nets[0])
        result.checkpoint = None
        layers = result.all_layers()
        # Do all the layers EXCEPT the last (output) layer
        plaininput = True
        for layernum, layer in enumerate(layers[:-1]):
            if hasattr(layer, 'W'):
                print("Merging dendrite layer", layernum, layer)
                self.merge_one_dendrite_layer(
                        result, layernum, share_input=plaininput)
                plaininput = False
            else:
                print("Merging simple layer", layernum, layer)
                self.merge_one_simple_layer(result, layernum)
        # The last layer should be the output layer with softmax
        print("Merging output layer", len(layers)- 1)
        self.merge_one_dendrite_layer(
                result, len(layers) - 1, share_output=True)
        return result

    def merge_one_simple_layer(self, result, layernum):
        print("Merging", layernum, result.layer_numbered(layernum))
        allparams = [result.layer_numbered(layernum).params] + [
            net.layer_numbered(layernum).params for net in self.nets]
        # Loop through all vector parameters
        for params in zip(*allparams):
            tparam = params[0]
            nparams = params[1:]
            # Accumulate parameter values in a vector.
            rvalue = np.zeros((0,), dtype='float32')
            for npar in nparams:
                npval = npar.get_value()
                # Don't know what to do with non-vector parameters
                assert len(npval.shape) == 1
                assert npval.shape[0] > 2
                rvalue = np.concatenate((rvalue, npval))
            print('Setting', tparam, rvalue.shape)
            tparam.set_value(rvalue)

    def merge_one_dendrite_layer(
            self, res, layernum, share_input=False, share_output=False):
        target = res.layer_numbered(layernum)
        print("Merging", layernum, target, "share_input", share_input,
                "share_output", share_output)
        if isinstance(target, DenseLayer):
            print("Using DenseLayer axis")
            inpd = 0
            outd = 1
        else:
            print("Using ConvLayer axis")
            inpd = 1
            outd = 0
        convshape = target.W.get_value().shape[2:]
        resw = np.zeros((0, 0) + convshape, dtype='float32')
        resb = np.zeros((0,), dtype='float32')
        outindex = inindex = 0
        hasbias = False
        for net in self.nets:
            layer = net.layer_numbered(layernum)
            netw = layer.W.get_value()
            if share_output:
                # When output is being shared, set output shape equal.
                if resw.shape[outd] != netw.shape[outd]:
                    rshape = list(resw.shape)
                    rshape[outd] = netw.shape[outd]
                    resw.shape = rshape
                # Concatenate along the input axis only.
                print('layernum',layernum,resw.shape, netw.shape,inpd)
                resw = np.concatenate((resw, netw), axis=inpd)
                print('result', resw.shape)
                # Bias aligns with output shape, so must be added.
                if hasattr(layer, 'b') and layer.b:
                    hasbias = True
                    netb = layer.b.get_value()
                    # If bias is present, add it!
                    if len(resb) == 0:
                        resb = netb
                    else:
                        resb = resb + netb
                continue
            if share_input:
                # When input is being shared, set input shape equal.
                if resw.shape[inpd] != netw.shape[inpd]:
                    rshape = list(resw.shape)
                    rshape[inpd] = netw.shape[inpd]
                    resw.shape = tuple(rshape)
            else:
                # Remember this number of units before we change it.
                toadd = netw.shape[inpd]
                # Concatenate zeroes to the left of the next network.
                inpzshape = list(netw.shape)
                inpzshape[inpd] = resw.shape[inpd]
                netw = np.concatenate(
                        (np.zeros(inpzshape, dtype='float32'), netw), axis=inpd)
                # Concatenate zeroes to the right of the accumulated network.
                inpzshape = list(resw.shape)
                inpzshape[inpd] = toadd # old netw.shape[inpd]
                resw = np.concatenate(
                        (resw, np.zeros(inpzshape, dtype='float32')), axis=inpd)
            # Concatentate the next network below the accumulated network.
            print('layernum',layernum,resw.shape, netw.shape,outd)
            resw = np.concatenate((resw, netw), axis=outd)
            print('result', resw.shape)
            if hasattr(layer, 'b') and layer.b:
                hasbias = True
                netb = layer.b.get_value()
                # If bias is present, concatenate it to the end
                resb = np.concatenate((resb, netb))
        # After everything has been concatenated, mutate the target network.
        print('final shape', resw.shape)
        target.W.set_value(resw)
        if hasbias:
            target.b.set_value(resb)

"""
    def merge_one_denselayer(self, result, layernum):
        outdims = [net.layer_numbered(layernum).W.get_value().shape[1]
                    for net in self.nets]
        indims = [net.layer_numbered(layernum).W.get_value().shape[0]
                    for net in self.nets]
        resultw = np.zeros((sum(indims), sum(outdims)))
        resultb = np.zeros(sum(outdims))
        outindex = inindex = 0
        hasbias = False
        for net in self.nets:
            layer = net.layer_numbered(layernum)
            netw = layer.W.get_value()
            resultw[inindex:inindex + netw.shape[0],
                    outindex:outindex + netw.shape[1]] = netw
            if hasattr(layer, 'b'):
                hasbias = True
                netb = net.layer_numbered(layernum).W.get_value()
                resultb[outindex:outindex + netb.shape[0]] = netb
            inindex += netw.shape[0]
            outindex += netw.shape[1]
        target = result.layer_numbered(layernum)
        target.W.set_value(resultw)
        if hasbias:
            target.b.set_value(resultb)

    def merge_one_convlayer(self, result, layernum, share_input=False):
        target = result.layer_numbered(layernum)
        convshape = target.W.get_value().shape[2:]
        outdims = [net.layer_numbered(layernum).W.get_value().shape[0]
                    for net in self.nets]
        indims = [net.layer_numbered(layernum).W.get_value().shape[1]
                    for net in self.nets]
        resultw = np.zeros((sum(outdims), sum(indims)) + convshape)
        resultb = np.zeros(sum(outdims))
        outindex = inindex = 0
        hasbias = False
        for net in self.nets:
            layer = net.layer_numbered(layernum)
            netw = layer.W.get_value()
            resultw[outoutdex:outoutdex + netw.shape[0],
                    inindex:inindex + netw.shape[1], :, :] = netw
            if hasattr(layer, 'b'):
                hasbias = True
                netb = net.layer_numbered(layernum).W.get_value()
                resultb[outindex:outindex + netb.shape[0]] = netb
            outindex += netw.shape[0]
            inindex += netw.shape[1]
        target.W.set_value(resultw)
        if hasbias:
            target.b.set_value(resultb)

    def merge_one_scalelayer(self, result, layernum):
        target = result.layer_numbered(layernum)
        allparams = [target.params] + [net.layer_number(layernum).params
                for net in self.nets]
        # Loop through all vector parameters
        for params in zip(allparams):
            tparam = params[0]
            nparams = params[1:]
            # Don't know what to do with non-vector parameters
            assert len(tparam.get_value().shape) == 1
            assert tparam.get_value().shape[0] > 2
            rvalue = np.zeros([np.get_value().shape[0] for np in nparams])
            pindex = 0
            for np in nparams:
                npval = np.get_value()
                rvalue[pindex:pindex + npval.shape[0]] = npval
                pindex += npval.shape[0]
            tparam.set_value(rvalue)

"""

class NetworkReducer:
    """
    Clones a network while hacking out and healing subsets of neurons.
    """
    def __init__(self, network, corpus):
        self.net = network
        self.corpus = corpus

    def create_reduced_network(self, sizes):
        """
        Sizes is a list of tuples (layer, integer), indicating the
        size of each layer to shrink.
        """
        # Load the ActivationSample for this network instance
        acts = ActivationSample(self.net, self.corpus)
        result = copy.deepcopy(self.net)
        result.checkpoint = None
        (axial_endpoints, axial_synapse) = compute_axial_maps(result)
        # Layers should be listed from closest-to-output first,
        # and then loop towards the beginning.
        # TODO: add a sort.
        for layer, numunits in sizes:
            # if layer is a string, look it up in net
            if isinstance(layer, str):
                layer = result.layer_named(layer)
            else:
                layer = result.layer_numbered(self.net(layer_number(layer)))
            numunits = int(numunits)
            # chop the layer's parameters
            weights = layer.W.get_value()
            if isinstance(layer, DenseLayer):
                # Dense layers organize weights as (in_units, out_units)
                layer.W.set_value(weights[:, :numunits])
                oldsize = weights.shape[1]
            else:
                # Conv layers organize weights as (out_units, in_units, x, y)
                layer.W.set_value(weights[(slice(0, numunits), ) + tuple(
                    (slice(None) for x in weights.shape[1:]))])
                oldsize = weights.shape[0]
            if hasattr(layer, 'b') and layer.b:
                biases = layer.b.get_value()
                layer.b.set_value(biases[:numunits])
            # also truncate batchnorm/relu or other parameterized layers
            # TODO: think about the case of more than one endpoint
            par = axial_endpoints[layer][0]['par']
            for parlayer in par:
                for param in parlayer.params:
                    val = param.get_value()
                    # assume that parameters of this dimension are
                    # one-per-channel and need to be truncated.
                    # TODO: maybe clean up this assumption.
                    if val.shape[0] == oldsize:
                        param.set_value(val[(slice(0, numunits), ) + tuple(
                            (slice(None) for x in val.shape[1:]))])
            # get the next layer's parameters
            # TODO: handle the case of more than one endpoint
            endpoint = axial_endpoints[layer][0]['layer']
            # solve least squares for the remaining units
            # TODO: handle the case when there are offsets
            activations = acts.layernum(result.layer_number(endpoint))
            # dimensions: (4096, size)
            A = activations[:,:numunits]
            # dimensions: (4096, removed)
            B = activations[:,numunits:]
            # Solve x = np.linagl.lstsq(A, B) for the original layer
            # X dimensions: (size, removed)
            (X, res, rank, s) = np.linalg.lstsq(A, B)
            # get the dendtritic layers
            # TODO: handle the case when the endpoint has more than one output
            syn = axial_synapse[endpoint][0]
            # TODO: think about b.
            # dimensions: (syn_units, in_units, 3, 3)
            syn_weights = syn.W.get_value()
            # Handle flattened input of dense layer case by unflattening
            flattened = None
            if isinstance(syn, DenseLayer):
                unflattened = endpoint.output_shape[1:]
                flattened = syn_weights.shape[0]
                syn_units = syn_weights.shape[1:]
                syn_weights.shape = (unflattened + syn_units)
                # transpose the weight matrix to conv2d orientation:
                # syn dimension first
                syn_weights = np.rollaxis(
                    syn_weights, len(syn_weights.shape) - 1)
            # dimensions: (syn_units, size, 3, 3)
            kept_weights = syn_weights[(slice(None), slice(0, numunits)) +
                tuple((slice(None) for s in syn_weights.shape[2:]))]
            # dimensions: (syn_units, removed, 3, 3)
            removed_weights = syn_weights[(slice(None), slice(numunits, None)) +
                tuple((slice(None) for s in syn_weights.shape[2:]))]
            # multiply x by the weights that are being removed
            # Adjusted weights. dimensions: (size, syn_units, 3, 3)
            adj_weights = np.tensordot(X, removed_weights, axes=([1], [1]))
            # Adjusted weights. dimensions: (syn_units, size, 3, 3)
            adj_weights = np.rollaxis(adj_weights, 1)
            # Incorporate adjustment
            sum_weights = kept_weights + adj_weights
            if flattened:
                # transpose the weight matrix to dense orientation:
                # syn dimension last (size, 3, 3, syn_units)
                sum_weights = np.rollaxis(
                    sum_weights, 0, len(adj_weights.shape))
                # now reflatten: (size*3*3, syn_units)
                sum_weights = sum_weights.reshape(
                    (flattened * numunits // oldsize,) + syn_units)
            # New weights.  Cast back down to float32.
            syn.W.set_value(sum_weights.astype(np.float32))
        # That's it.
        return result

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
        self.layers = network.all_layers()
        self.index_from_layer = dict(
            (layer, i) for i, layer in enumerate(self.layers))
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

    def layer_index(self, layer):
        return self.index_from_layer[layer]

    def layer(self, index):
        return self.layers[index]

    def major_nontrivial_inputs(self, layer, coord, num=10):
        parts = self.follow_nontrivial_inputs(layer, coord, num)
        results = []
        for layer, coord, amount in parts:
            while is_trivial_layer(layer):
                layer, coord = self.follow_trivial_inputs(layer, coord)
            results.append((layer, coord, amount))
        return results

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
        contribs = np.dot(ia.flatten(), weights)
        top = (-contribs).argsort()[:num]
        coords = zip(*np.unravel_index(top, ia.shape))
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

class DebuggerCache:
    def __init__(self, corpus):
        self.corpus = corpus
        self.model_cache = {}
        self.debugger_cache = {}
    def lookup(self, modelname, imgnum):
        if (modelname, imgnum) in self.debugger_cache:
            return self.debugger_cache[(modelname, imgnum)]
        if modelname not in self.model_cache:
             self.model_cache[modelname] = (
                 load_from_checkpoint(modelname))
        net = self.model_cache[modelname]
        img = corpus.get('val', img, shape=net.crop_size)
        dbg = Debugger(net, img)
        self.debugger_cache[(modelname, imgnum)] = dbg
        return dbg
