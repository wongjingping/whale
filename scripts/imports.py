

from glob import glob
from os import remove
from cPickle import dump, load
from time import time
from re import search
from PIL import Image

import matplotlib.pyplot as plt
import pylab

from random import sample, random
from numpy import empty, asarray, ones, zeros, where, concatenate, \
	int32, mean, savetxt, sqrt
from numpy.random import RandomState
import pandas as pd

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams

