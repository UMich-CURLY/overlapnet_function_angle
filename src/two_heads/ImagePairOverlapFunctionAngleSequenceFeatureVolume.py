#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: A keras generator which generates batches out of given feature volumes
from tensorflow.keras.utils import Sequence
import numpy as np


class ImagePairOverlapFunctionAngleSequenceFeatureVolume(Sequence):
    """ This class is responsible for separating feature volumes into batches. It
        can be used as keras generator object in e.g. model.fit_generator.
    """
    
    def __init__(self, pairs, overlap, function_angle, batch_size, feature_volumes):
      """ Initialize the dataset.
          Args:
            pairs  : a nx2 array of indizes (in array feature_volumes) of pairs
            overlap: a nx1 numpy array with the overlap (0..1). Same length as
                     pairs
            functiona ngle: a nx1 numpy array with the function angle (0..1). Same length as
                     pairs
            batch_size: size of a batch                   
            feature_volumes: all feature volumes of all image sets:
                             a numpy array with dimension n x w x h x chans
      """
      self.pairs=pairs                     
      self.batch_size = batch_size
      self.overlap = overlap
      self.function_angle = function_angle
      self.feature_volumes=feature_volumes
      # number of pairs
      self.n = overlap.size

    def __len__(self):
      """ Returns number of batches in the sequence. (overwritten method)
      """
      return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, idx):
      """ Get a batch. (overwritten method)
      """
      maxidx=(idx + 1) * self.batch_size
      if maxidx>self.n:
          maxidx=self.n
            
      batch = self.pairs[idx * self.batch_size : maxidx, :]
      x1=self.feature_volumes[batch[:,0],:,:,:]
      x2=self.feature_volumes[batch[:,1],:,:,:]
      y_overlap  = self.overlap[idx * self.batch_size : maxidx]
      y_function_angle  = self.function_angle[idx * self.batch_size : maxidx]
      
      return ( [x1,x2], [np.hstack((y_overlap.reshape(-1,1), y_function_angle.reshape(-1,1)))] )

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          'pairs': self.pairs,
          'batch_size': self.batch_size,
          'overlap': self.overlap,
          'function_angle': self.function_angle,
          'feature_volunb': self.feature_volumns,
          'n': self.n,
          })
      return config