from torch.autograd import Variable
import torch.onnx
import torchvision
import cv2

import numpy as np

import nn_models

import data_loading.image_loading as il

import nn_models.Models as models

import data_loading.data_loaders_old as loaders

import numpy.random

import torch, random

import torch.nn as nn 

import torch.optim as optim

from tqdm import tqdm as tqdm

import pickle

from datetime import datetime

import os

import string

import argparse

import torchvision.transforms as transforms
import onnx



dummy_input = torch.randn(1, 10, 3, 66, 200)#.cuda(0)

rnn_cell_type = 'lstm'
gpu = -1
model = models.AdmiralNet(cell=rnn_cell_type, context_length = 10, sequence_length=1, hidden_dim = 100, use_float32 = True, gpu = gpu)
#model = model.cuda(gpu)
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "images_in"]
output_names = [ "control_out" ]
torch.onnx.export(model, dummy_input, "admiralnet.onnx", input_names=input_names, output_names=output_names,verbose=False)


