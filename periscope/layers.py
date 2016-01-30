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


# ***********
# ***********
# xx       xx
# xx       xx
# xx  ooo  xx
# xx  ooo  xx
# xx  ooo  xx
# xx       xx
# xx       xx
# ***********
# ***********
class LandmarkLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(LandmarkLayer, self).__init__(incoming, **kwargs)
        dims = self.input_layer.output_shape[2:]
        cy = dims[0] // 2
        cx = dims[1] // 2
        landmark = np.zeros((1, 5) + dims, dtype=np.float32)
        # Top edge
        landmark[0, 0, :2, :] = 1
        # Bottom edge
        landmark[0, 1, -2:, :] = 1
        # Left edge
        landmark[0, 2, :, :2] = 1
        # Right edge
        landmark[0, 3, :, -2:] = 1
        # Center
        landmark[0, 4, cy-1:cy+2, cx-1:cx+2] = 1
        self.landmark = theano.tensor.constant(landmark)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        shape[1] += 5
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        return theano.tensor.concatenate([
            input,
            theano.tensor.repeat(self.landmark, input.shape[0], axis=0)
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

