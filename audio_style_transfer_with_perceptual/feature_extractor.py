import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess
import sys
import numpy as np
from vggish_smoke_test import *
def CreateVGGishNetwork(hop_size=0.96):   # Hop size is in seconds.
  """Define VGGish model, load the checkpoint, and return a dictionary that points
  to the different tensors defined by the model.
  """
  vggish_slim.define_vggish_slim()
  checkpoint_path = 'vggish_model.ckpt'
  vggish_params.EXAMPLE_HOP_SECONDS = hop_size
  
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)

  layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
  g = tf.get_default_graph()
  for k in layers:
    layers[k] = g.get_tensor_by_name( layers[k] + ':0')
    
  return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         }

# def ProcessWithVGGish(vgg, x, sr):
#       '''Run the VGGish model, starting with a sound (x) at sample rate
#   (sr). Return a whitened version of the embeddings. Sound must be scaled to be
#   floats between -1 and +1.'''

#   # Produce a batch of log mel spectrogram examples.
#   input_batch = vggish_input.waveform_to_examples(x,sr)
#   # print('Log Mel Spectrogram example: ', input_batch[0])
#   [embedding_batch] = sess.run([vgg['embedding']],
#                                feed_dict={vgg['features']: input_batch})

#   # Postprocess the results to produce whitened quantized embeddings.
#   pca_params_path = 'vggish_pca_params.npz'

#   pproc = vggish_postprocess.Postprocessor(pca_params_path)
#   postprocessed_batch = pproc.postprocess(embedding_batch)
#   # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
#   return postprocessed_batch[0]

def ProcessWithVGGish_file(vgg, file):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a whitened version of the embeddings. Sound must be scaled to be
  floats between -1 and +1.'''

  # # Produce a batch of log mel spectrogram examples.
  input_batch = vggish_input.wavfile_to_examples(file)
  # # print('Log Mel Spectrogram example: ', input_batch[0])

  [embedding_batch] = sess.run([vgg['embedding']],
                               feed_dict={vgg['features']: input_batch})

  # Postprocess the results to produce whitened quantized embeddings.
  pca_params_path = 'vggish_pca_params.npz'

  pproc = vggish_postprocess.Postprocessor(pca_params_path)
  postprocessed_batch = pproc.postprocess(embedding_batch)
  # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
  return postprocessed_batch[0]


# Test these new functions with the original test.
import tensorflow as tf
tf.reset_default_graph()
sess = tf.Session()

vgg = CreateVGGishNetwork(0.01)

# Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
# to test resampling to 16 kHz during feature extraction).
num_secs = 3
freq = 1000
sr = 44100
t = np.linspace(0, num_secs, int(num_secs * sr))
x = np.sin(2 * np.pi * freq * t)  # Unit amplitude input signal
# file=sys.argv[1]
# print(file)
# postprocessed_batch = ProcessWithVGGish_file(vgg, file)

# print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
# expected_postprocessed_mean = 123.0
# expected_postprocessed_std = 75.0
# np.testing.assert_allclose(
#     [np.mean(postprocessed_batch), np.std(postprocessed_batch)],
#     [expected_postprocessed_mean, expected_postprocessed_std],
#     rtol=rel_error)


def EmbeddingsFromVGGish(vgg, x, sr):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a dictionary of embeddings from the different layers
  of the model.'''
  # Produce a batch of log mel spectrogram examples.
  input_batch = vggish_input.waveform_to_examples(x, sr)
  # print('Log Mel Spectrogram example: ', input_batch[0])

  layer_names = vgg['layers'].keys()
  tensors = [vgg['layers'][k] for k in layer_names]
  
  results = sess.run(tensors,
                     feed_dict={vgg['features']: input_batch})

  resdict = {}
  for i, k in enumerate(layer_names):
    resdict[k] = results[i]
    
  return resdict


def EmbeddingsFromVGGish_file(vgg, path):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a dictionary of embeddings from the different layers
  of the model.'''
  # Produce a batch of log mel spectrogram examples.
  input_batch = vggish_input.wavfile_to_examples(path)
  # print('Log Mel Spectrogram example: ', input_batch[0])

  layer_names = vgg['layers'].keys()
  tensors = [vgg['layers'][k] for k in layer_names]
  
  results = sess.run(tensors,
                     feed_dict={vgg['features']: input_batch})

  resdict = {}
  for i, k in enumerate(layer_names):
    resdict[k] = results[i]
    
  return resdict


# resdict = EmbeddingsFromVGGish_file(vgg, file)

# print(resdict["embedding"])

# for k in resdict:
#   print(k, resdict[k].shape)
  # print(resdict[k])

