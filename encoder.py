"""
Encoder:
    text
    |
    V
    embedding
    |
    V
    Prenet
    |
    V
    CBHG

"""
import tensorflow as tf
import numpy as np

from network_module import prenet, cbhg




"""
        encoder contains:
        
        prenet  -> 256 neurons (dropout 0.5) 
                -> 128 neurons (dropout 0.5)
                
        cbhg
        
        return the output
"""
def encoder(inputs, is_training=True):
    #prenet
    outputs = prenet(inputs, is_training=is_training, scope='encoder_prenet')
    #cbhg
    outputs = cbhg(outputs, 16, projections=[128, 128], scope='encoder_cbhg', is_training=is_training) #16 refers to the K filter for the conv bank

    return outputs







#  toy data
data = np.array(np.random.rand(100, 26), dtype=np.float32)
data = tf.convert_to_tensor(data, dtype=tf.float32)
prenet(data, True)