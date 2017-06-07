#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:47:39 2017
Obtendo a representação tempo - frequência do sinal utilizando a Transformada 
Wavelet Contínua

@author: renato
"""
import numpy as np
import pywt 
import matplotlib.pyplot as plt
from scipy.io import loadmat

# carregando o dataset contendo os dados de eeg

data = loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/dataset_kolodziej.mat')

