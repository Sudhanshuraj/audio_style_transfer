import sys
import tensorflow as tf
import librosa
import numpy as np

from feature_extractor import EmbeddingsFromVGGish,vgg

def perceptual_loss_fn(x1, x2, sr):
    # a = tf.zeros_like(x2)
    # a[:N_CHANNELS,:] = tf.exp(tf.transpose(x1[0,0])) - 1
    # x1 = librosa.istft(a)
    resdict1 = EmbeddingsFromVGGish(vgg, x1, sr)
    embedding1=resdict1["embedding"]
    del resdict1
    resdict2 = EmbeddingsFromVGGish(vgg, x2, sr)
    embedding2=resdict2["embedding"]
    del resdict2
    return tf.nn.l2_loss(embedding1-resdict2["embedding"])

# print(perceptual_loss(sys.argv[1],sys.argv[2]))