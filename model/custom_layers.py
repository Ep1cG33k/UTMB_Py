import keras.backend as K
from tensorflow.keras.layers import UpSampling2D

class DePool2D(UpSampling2D):
  '''Simplar to UpSample, yet traverse only maxpooled elements
  # Input shape
      4D tensor with shape:
      `(samples, channels, rows, cols)` if dim_ordering='th'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if dim_ordering='tf'.
  # Output shape
      4D tensor with shape:
      `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
      or 4D tensor with shape:
      `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
  # Arguments
      size: tuple of 2 integers. The upsampling factors for rows and columns.
      dim_ordering: 'th' or 'tf'.
          In 'th' mode, the channels dimension (the depth)
          is at index 1, in 'tf' mode is it at index 3.
  '''
  input_ndim = 4

  def __init__(self, pool2d_layer=0, *args, **kwargs):
      self._pool2d_layer = pool2d_layer
      super().__init__(*args, **kwargs)

  def get_config(self):
      config = super().get_config()
      config['pool2d_layer'] =  self._pool2d_layer# say self. _localization_net  if you store the argument in __init__
        # say self. _output_size  if you store the argument in __init__
      return config
  
  def get_output(self, train=False):
      X = self.get_input(train)
      if self.dim_ordering == 'th':
          output = K.repeat_elements(X, self.size[0], axis=2)
          output = K.repeat_elements(output, self.size[1], axis=3)
      elif self.dim_ordering == 'tf':
          output = K.repeat_elements(X, self.size[0], axis=1)
          output = K.repeat_elements(output, self.size[1], axis=2)
      else:
          raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

      f = K.grad(K.sum(self._pool2d_layer.get_output(train)), wrt=self._pool2d_layer.get_input(train)) * output

      return f