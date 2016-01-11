#!/usr/bin/env python3

import experiment
import lasagne
import theano
import theano.tensor as T
import pickle
import numpy
import os
import os.path

class Model:
    def __init__(self, network, filename=None, batchsize=None, cats=100):
        if network not in experiment.__dict__:
            raise LookupError("No network {} found.".format(network))
        self.network_fn = experiment.__dict__[network]
        self.cropsz = 117
        self.batchsize = 256
        self.l2reg = 1e-3
        self.cats = cats
        self.learning_rates = numpy.logspace(-1.5, -4, 30, dtype=numpy.float32)
        self._eval_fn = None
        self._train_fn = None
        self._acc_fn = None
        self._debug_fn = None
        if hasattr(self.network_fn, 'cropsz'):
            self.cropsz = self.network_fn.cropsz
        if hasattr(self.network_fn, 'l2reg'):
            self.l2reg = self.network_fn.l2reg
        if hasattr(self.network_fn, 'learning_rates'):
            self.learning_rates = self.network_fn.learning_rates
        if batchsize is not None:
            self.batchsize = batchsize
        self.state = None
        if filename:
            lfile = open(filename, 'rb')
            formatver = pickle.load(lfile)
            self.state = pickle.load(lfile)
            lfile.close()
        self.center = numpy.zeros((2,), dtype=numpy.int32)
        self.center.fill(numpy.floor((128 - self.cropsz)/2))

        # parameters
        self.input_var = T.tensor4('X')
        self.flip_var = T.iscalar('f')
        self.crop_var = T.ivector('c') # ycrop, xcrop
        
        # crop+flip
        top = self.crop_var[0]
        left = self.crop_var[1]
        cropped = self.input_var[:,:,top:top+self.cropsz, left:left+self.cropsz]
        prepared = cropped[:,:,:,::self.flip_var]
        
        # input layer is always the same
        network = lasagne.layers.InputLayer(
                (self.batchsize, 3, self.cropsz, self.cropsz), prepared,
                name="input")
        network = self.network_fn(network, self.cropsz, self.batchsize)

        # Last softmax layer is always the same
        from lasagne.nonlinearities import softmax
        network = lasagne.layers.DenseLayer(network, cats, name="softmax",
                nonlinearity=softmax)
        self.network = network
        self.prediction = lasagne.layers.get_output(network, deterministic=True)

        # initialize params based on saved data
        if self.state:
            saveparams = lasagne.layers.get_all_params(network)
            assert len(saveparams) == len(self.state)
            for p, v in zip(saveparams, self.state):
                p.set_value(v)

        # accuracy setup
        self.target_var = T.ivector('y')
        self.loss = lasagne.objectives.categorical_crossentropy(
                self.prediction, self.target_var)

        # training setup
        self.learning_rate_var = T.scalar('l')
        train_prediction = lasagne.layers.get_output(network)
        # create loss function
        from lasagne.regularization import regularize_network_params, l2
        loss = lasagne.objectives.categorical_crossentropy(
                train_prediction, self.target_var).mean()
        loss += regularize_network_params(network, l2) * self.l2reg
        self.train_loss = loss
        # create parameter update expressions
        params = lasagne.layers.get_all_params(network, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
                loss,
                params,
                learning_rate=self.learning_rate_var,
                momentum=0.9)
   
        # As a bonus, create an expression for the classification accuracy:
        self.test_1_acc = T.mean(lasagne.objectives.categorical_accuracy(
                self.prediction, self.target_var, top_k=1))
        self.test_5_acc = T.mean(lasagne.objectives.categorical_accuracy(
                self.prediction, self.target_var, top_k=5))

    def named_layers(self):
        return [layer
                   for layer in lasagne.layers.get_all_layers(self.network)
                   if layer.name]

    def layer_names(self):
        return [layer.name for layer in self.named_layers()]

    def eval_fn(self):
        if not self._eval_fn:
            self._eval_fn = theano.function([
                self.input_var,
                theano.Param(self.flip_var, default=1),
                theano.Param(self.crop_var, default=self.center)],
                self.prediction,
                allow_input_downcast=True)
        return self._eval_fn

    def acc_fn(self):
        if not self._acc_fn:
            self._acc_fn = theano.function([
                self.input_var,
                self.target_var,
                theano.Param(self.flip_var, default=1),
                theano.Param(self.crop_var, default=self.center)],
                [self.loss.mean(), self.test_1_acc, self.test_5_acc],
                allow_input_downcast=True)
        return self._acc_fn

    def train_fn(self):
        if not self._train_fn:
            self._train_fn = theano.function([
                self.input_var,
                self.target_var,
                self.learning_rate_var,
                theano.Param(self.flip_var, default=1),
                theano.Param(self.crop_var, default=self.center)],
                self.train_loss,
                updates=self.updates,
                allow_input_downcast=True)
        return self._train_fn

    def debug_fn(self):
        if not self._debug_fn:
            named = self.named_layers()
            outs = lasagne.layers.get_output(named, deterministic=True)
            self._debug_fn = theano.function([
                self.input_var,
                theano.Param(self.flip_var, default=1),
                theano.Param(self.crop_var, default=self.center)],
                outs,
                allow_input_downcast=True)
        return self._debug_fn
