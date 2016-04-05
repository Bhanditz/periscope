import lasagne
import theano
import numpy as np

class ZeroGrayLayer(lasagne.layers.Layer):
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return input - 0.5

class ZeroPreluLayer(lasagne.layers.Layer):
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return (input - 0.2) * 1.25

class EdgeDecayLayer(lasagne.layers.Layer):
    """
    Adds padding to the input layer by reflecting the edge pixel values.
    """
    def __init__(self, incoming, decay=1/3.0, **kwargs):
        super(EdgeDecayLayer, self).__init__(incoming, **kwargs)
        self.pad = (1, 1)
        self.decay = decay

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2] + 2,
            input_shape[3] + 2)

    def get_output_for(self, input, **kwargs):
        # Decay an amount from top and bottom and smear it outward.
        top = input[:,:,:1,:]
        bot = input[:,:,-1:,:]
        middle = input[:,:,1:-1,:]
        grown = theano.tensor.concatenate((
            top * self.decay,
            top * (1 - self.decay),
            middle,
            bot * (1 - self.decay),
            bot * self.decay), axis=2)
        # Decay an amount from left and right and smear it outward.
        lef = grown[:,:,:,:1]
        rig = grown[:,:,:,-1:]
        middle = grown[:,:,:,1:-1]
        return theano.tensor.concatenate((
            lef * self.decay,
            lef * (1 - self.decay),
            middle,
            rig * (1 - self.decay),
            rig * self.decay), axis=3)

class ReflateLayer(lasagne.layers.Layer):
    """
    Adds padding to the input layer by reflecting the edge pixel values.
    """
    def __init__(self, incoming, pad=1, decay=1, **kwargs):
        super(ReflateLayer, self).__init__(incoming, **kwargs)
        if isinstance(pad, int):
            pad = (pad, pad)
        self.pad = pad
        self.decay = decay

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2] + self.pad[0] * 2,
            input_shape[3] + self.pad[1] * 2)

    def get_output_for(self, input, **kwargs):
        # Concatenate reflected top and bottom edges
        refltop = input[:,:,self.pad[0]-1::-1,:] * self.decay
        reflbot = input[:,:,:-self.pad[0]-1:-1,:] * self.decay
        grown = theano.tensor.concatenate((refltop, input, reflbot), axis=2)
        # Concatenate reflected left and right edges
        refllef = grown[:,:,:,self.pad[1]-1::-1] * self.decay
        reflrig = grown[:,:,:,:-self.pad[1]-1:-1] * self.decay
        return theano.tensor.concatenate((refllef, grown, reflrig), axis=3)

class LandmarkLayer(lasagne.layers.Layer):
    def __init__(self, incoming, kinds='vc', **kwargs):
        super(LandmarkLayer, self).__init__(incoming, **kwargs)
        dims = self.input_layer.output_shape[2:]
        # Bottom-to-top gradient
        vertical = (
            np.linspace(1, -1, dims[0], dtype=np.float32)[:,np.newaxis])
        vertical2 = vertical * vertical
        # Left-to-right gradient
        horizontal = (
            np.linspace(-1, 1, dims[1], dtype=np.float32)[np.newaxis,:])
        horizontal2 = horizontal * horizontal
        # Closeness to center
        center = 1 - (
            (vertical) ** 2 + (horizontal) ** 2) ** 0.5
        # Top edge
        top = np.eye(dims[0], 1, dtype=np.float32)
        top2 = top + np.roll(top, 1, axis=0)
        # Bottom edge
        bottom = top[::-1,:]
        bottom2 = top2[::-1,:]
        # Left edge
        left = np.eye(1, dims[1], dtype=np.float32)
        left2 = left + np.roll(left, 1, axis=1)
        # Right edge
        right = left[:,::-1]
        right2 = left2[:,::-1]
        # Middle point if odd (2 points if even)
        cpt = ((d - 1) // 2 for d in dims)
        middle = np.zeros(dims, dtype=np.float32)
        middle[tuple(slice(c, d - c) for c, d in zip(cpt, dims))] = 1
        # Middle 3 points if odd (4 points if even)
        middle3 = np.zeros(dims, dtype=np.float32)
        middle3[tuple(slice(c - 1, d - c + 1) for c, d in zip(cpt, dims))] = 1
        
        included = []
        for code, signal in zip(
            'VHvhctbrlmTBLRM',
            [vertical2, horizontal2,
             vertical, horizontal, center,
             top, bottom, left, right, middle,
             top2, bottom2, left2, right2, middle3]):
            if code in kinds:
                included.append(signal)
        # Verify all kinds are recognized
        assert len(kinds) == len(included), 'Unrecognized kind in %s' % kinds
        self.kinds = kinds
        # Create the theano constant
        gradient = np.zeros((1, len(kinds)) + dims, dtype=np.float32)
        for i, signal in enumerate(included):
            gradient[0, i, :, :] = signal
        # theano.shared is much faster than theano.tensor.constant:
        # it seems to force the value onto the GPU.
        self.gradient = theano.shared(gradient)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        shape[1] += len(self.kinds)
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        return theano.tensor.concatenate([
            input,
            theano.tensor.repeat(self.gradient, input.shape[0], axis=0)
        ], axis=1)

class QuickNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming,
                 axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem',
                 mean=lasagne.init.Constant(0),
                 inv_std=lasagne.init.Constant(1), **kwargs):
        super(QuickNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("QuickNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        self.mean = self.add_param(mean, shape, 'mean',
                                      trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = theano.tensor.inv(theano.tensor.sqrt(
                input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = kwargs.get('batch_norm_use_averages',
                                  deterministic)
        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        update_averages = kwargs.get('batch_norm_update_averages',
                                     not deterministic)
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * inv_std
        return normalized

class QuickMeanLayer(lasagne.layers.Layer):
    def __init__(self, incoming,
                 axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem',
                 mean=lasagne.init.Constant(0),
                 **kwargs):
        super(QuickMeanLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("QuickNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        self.mean = self.add_param(mean, shape, 'mean',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = kwargs.get('batch_norm_use_averages',
                                  deterministic)
        if use_averages:
            mean = self.mean
        else:
            mean = input_mean

        # Decide whether to update the stored averages
        update_averages = kwargs.get('batch_norm_update_averages',
                                     not deterministic)
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        mean = mean.dimshuffle(pattern)

        # normalize
        normalized = (input - mean)
        return normalized

class ZeroReluLayer(lasagne.layers.Layer):
    def __init__(self, incoming, alpha=0, **kwargs):
        super(ZeroReluLayer, self).__init__(incoming, **kwargs)
        self.mean = numpy.float32(numpy.sqrt(2 / numpy.pi) * (1 - alpha) / 2)
        self.invstd = numpy.float32(
            1/numpy.sqrt(0.5 * ((1 - 2/numpy.pi) * (1 + alpha) +
            1/numpy.pi * (1 + alpha)**2)))
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return (input - self.mean) * self.invstd

class ConstShiftLayer(lasagne.layers.Layer):
    def __init__(self, incoming, shift=0, scale=1, **kwargs):
        super(ConstShiftLayer, self).__init__(incoming, **kwargs)
        self.shift = shift
        self.scale = scale
    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_output_for(self, input, **kwargs):
        return (input + self.shift) * self.scale
