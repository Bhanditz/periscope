#!/usr/bin/env python3

from periscope.checkpoint import Checkpoint
import pickle
import numpy as np
import os
import os.path

class Network:
    def __init__(self, **kwargs):
        """
        Creates a convolutional network with some hidden topology.
        Arguments:
           model = '/exp-somedir' - loads state from the given directory
        """
        import theano, lasagne
        self.init_constants()

        self._train_fn = self._eval_fn = self._debug_fn = None

        self.input_var = theano.tensor.btensor4('X')
        self.target_var = theano.tensor.ivector('Y')
        self.epoch_var = theano.tensor.fscalar('epoch')
        self.acc = {'train': [], 'val': []}
        # rescale [-128...127] to [0...1]
        scaled_input = (theano.tensor.cast(
            self.input_var, 'float32') + 128) / 255

        # Build network from RGB crop_size images to output_size outupts,
        # with hidden layers between.
        network = lasagne.layers.InputLayer((None, 3) + self.crop_size,
            name="input", input_var=scaled_input)
        network = self.hidden_layers(network, **kwargs)
        network = lasagne.layers.DenseLayer(network, self.output_size,
            name="output", nonlinearity=lasagne.nonlinearities.softmax)
        self.network = network

        # Associate with checkpoint directory
        self.checkpoint = None
        self.epoch = -1
        if 'model' in kwargs:
            self.checkpoint = Checkpoint(kwargs['model'])
        data = kwargs.get('data', None)
        if data or (self.checkpoint and self.checkpoint.exists()):
            self.load_checkpoint(data)

    def init_constants(self):
        self.output_size = 100
        self.crop_size = (106, 106)
        self.batch_size = 256
        self.learning_rates = np.logspace(-2, -4.5, 30, dtype=np.float32)
        self.ramp_learning_rate = 2

    def hidden_layers(self, network, **kwargs):
        import lasagne
        # conv: 32x106x106
        network = lasagne.layers.Conv2DLayer(network, 32, (7, 7))
        # mp: 32x99x99
        network = lasagne.layers.MaxPool2DLayer(network, (3, 3), stride=2)
        # conv: 96x49x49
        network = lasagne.layers.Conv2DLayer(network, 96, (5, 5))
        # mp: 96x45x45
        network = lasagne.layers.MaxPool2DLayer(network, (15, 15), stride=15)
        # conv: 512x3x3
        network = lasagne.layers.Conv2DLayer(network, 512, (3, 3))
        return network

    def regularize_parameters(self):
        import lasagne
        return lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2) * 1e-2

    def training_loss(self):
        import lasagne
        training_prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(
                training_prediction, self.target_var).mean()
        loss += self.regularize_parameters()
        return loss

    def learning_rate(self):
        import theano, lasagne
        epoch_num = theano.tensor.cast(self.epoch_var, 'int32')
        learning_rates_array = theano.tensor.constant(
            self.learning_rates.astype(np.float32))
        learning_rate = learning_rates_array[epoch_num]
        if self.ramp_learning_rate:
            progress = theano.tensor.cast(self.epoch_var - epoch_num, 'float32')
            factor = self.ramp_learning_rate * (1 - progress) + progress
            learning_rate = learning_rate * factor
        return learning_rate

    def parameter_updates(self, training_loss):
        import lasagne
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        return lasagne.updates.nesterov_momentum(
                training_loss,
                params,
                learning_rate=self.learning_rate(),
                momentum=0.9)

    def all_layers(self):
        import lasagne
        return lasagne.layers.get_all_layers(self.network)

    def named_layers(self):
        return [layer for layer in self.all_layers() if layer.name]

    def layer_map(self, mapping):
        return dict([(layer, mapping[layer.name])
                   for layer in self.all_layers() if layer.name in mapping])

    def layer_named(self, name):
        for layer in self.all_layers():
            if layer.name == name:
                return layer

    def eval_fn(self):
        if not self._eval_fn:
            import theano, lasagne
            prediction = lasagne.layers.get_output(
                self.network, deterministic=True)
            self._eval_fn = theano.function(
                [self.input_var],
                prediction,
                allow_input_downcast=True)
        return self._eval_fn

    def train_fn(self):
        if not self._train_fn:
            import theano
            training_loss = self.training_loss()
            parameter_updates = self.parameter_updates(training_loss)
            self._train_fn = theano.function(
                [self.input_var, self.target_var, self.epoch_var],
                training_loss,
                updates=parameter_updates,
                allow_input_downcast=True)
        return self._train_fn

    def debug_fn(self, layers=None):
        import lasagne, theano
        if layers is None:
            if self._debug_fn:
                return self._debug_fn
            select = self.all_layers()
        else:
            select = layers
        outs = lasagne.layers.get_output(select, deterministic=True)
        result = theano.function(
            [self.input_var],
            outs,
            allow_input_downcast=True)
        if layers is None:
            self._debug_fn = result
        return result

    def conv_padding(self, pad, filter_size):
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

    def conv_stride(self, stride, filter_size):
        if isinstance(stride, int):
            return tuple(stride for s in filter_size)
        assert isinstance(stride, tuple)
        return stride

    def unroll_convolution(self, img, layer, coord):
        """
        Given an RGBxhxw input image and a specific activation in a convolutional
        layer, returns the tensor of contributions of each weight to that
        specific unit in that specific situation.
        """
        import lasagne
        debug_fn = self.debug_fn()
        activations = debug_fn(img[np.newaxis,:])
        acts = dict(zip(self.get_all_layers(), activations))
        pad = self.conv_padding(layer.pad, layer.filter_size)
        stride = self.conv_stride(layer.stride, layer.filter_size)
        sect = tuple(slice(c * s, c * s + f) for c, s, f in
                zip(coord, stride, layer.filter_size))
        input_layer = layer.input_layer
        ia = acts[input_layer]
        padded_input = np.pad(ia, (0,) + pad, mode='constant')
        seen_input = padded_input[(slice(None), ) + sect]
        weights = layer.W.get_value()
        return seen_input * weights

    def named_layers(self):
        import lasagne
        return [layer
                   for layer in lasagne.layers.get_all_layers(self.network)
                   if layer.name]

    def eval_1_5(self, kind, eval_fn, val_set, pretty=None):
        batch_count = min(len(val_set), 20)
        c_1 = c_5 = d = 0
        if pretty:
            pretty.subtask("Evaluating on {} batches of {} set".format(
                batch_count, kind))
            p = pretty.progress(batch_count)
        for i, val_batch in enumerate(val_set):
            if i >= batch_count:
                break
            images, labels, names = val_batch
            results = eval_fn(images)
            preds = np.argsort(-results, axis=1)
            labels = np.expand_dims(labels, 1)
            ranks = abs(preds - labels).argmin(axis=1)
            c_1 += sum(ranks == 0)
            c_5 += sum(ranks < 5)
            d += len(ranks)
            if pretty:
                p.update(i + 1)
        if pretty:
            p.finish()
        return (c_1 / float(d), c_5 / float(d))

    def train(self, corpus, pretty=None):
        # Build training function
        if pretty:
            import lasagne
            pretty.subtask("Compiling training function: %d params." %
                lasagne.layers.count_params(self.network))
        train_fn = self.train_fn()
        if pretty:
            eval_fn = self.eval_fn()
        for epoch in range(len(self.learning_rates)):
            if epoch <= self.epoch:
                continue
            # The main training loop
            training_set = corpus.batches(
                'train',
                batch_size=self.batch_size,
                shape=self.crop_size,
                randomize=True)
            batch_count = len(training_set)
            if pretty:
                pretty.task("Starting epoch {} for network {}".format(
                    epoch, self.__class__.__name__))
                p = pretty.progress(batch_count)
            for i, training_batch in enumerate(training_set):
                epoch_f = epoch + i / float(batch_count)
                images, labels, names = training_batch
                train_fn(images, labels, epoch_f)
                if pretty:
                    p.update(i+1)
            p.finish()
            # Evaluation
            for kind in ['train', 'val']:
                val_set = corpus.batches(kind, shape=self.crop_size)
                acc1, acc5 = self.eval_1_5(kind, eval_fn, val_set)
                self.acc[kind].append((0, acc1, acc5))
            if pretty:
                pretty.subtask(("Epoch {} results:" +
                        " {:.2f}%/{:.2f}% (t1acc, v1acc)" +
                        " {:.2f}%/{:.2f}% (t5acc, v5acc)").format(
                        epoch,
                        self.acc['train'][-1][1] * 100,
                        self.acc['val'][-1][1] * 100,
                        self.acc['train'][-1][2] * 100,
                        self.acc['val'][-1][2] * 100,
                ))
            # Checkpointing
            self.epoch = epoch
            if self.checkpoint:
                self.save_checkpoint()

    def load_checkpoint(self, data):
        if not data:
            data = self.checkpoint.load()
        (state, epoch, train_acc, val_acc) = data[:4]
        self.epoch = epoch
        self.acc['train'] = train_acc
        self.acc['val'] = val_acc 
        import lasagne
        saveparams = lasagne.layers.get_all_params(self.network)
        assert len(saveparams) == len(state)
        for p, v in zip(saveparams, state):
            p.set_value(v)

    def save_checkpoint(self):
        import lasagne
        saveparams = lasagne.layers.get_all_params(self.network)
        state = [p.get_value() for p in saveparams]
        self.checkpoint.save(
            (state, self.epoch, self.acc['train'], self.acc['val'],
             self.__class__.__module__ + '.' + self.__class__.__qualname__))

