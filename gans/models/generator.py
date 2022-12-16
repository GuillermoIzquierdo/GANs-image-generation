from tensorflow.keras.layers import Conv2DTranspose, Reshape, Conv2D, BatchNormalization, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU

def build_generator(start_filters, filter_size, latent_dim):

  # function for building a CNN block for upsampling the image
  def add_generator_block(x, filters, filter_size):
      x = Conv2DTranspose(filters, filter_size, strides=2, padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.3)(x)
      return x

  # input is a noise vector
  inp = Input(shape=(latent_dim,))

  # projection of the noise vector into a tensor with
  # same shape as last conv layer in discriminator
  x = Dense(4 * 4 * (start_filters * 8), input_dim=latent_dim)(inp)
  x = BatchNormalization()(x)
  x = Reshape(target_shape=(4, 4, start_filters * 8))(x)

  # design the generator to upsample the image 4x
  x = add_generator_block(x, start_filters * 4, filter_size)
  x = add_generator_block(x, start_filters * 2, filter_size)
  x = add_generator_block(x, start_filters, filter_size)
  x = add_generator_block(x, start_filters, filter_size)

  # turn the output into a 2D tensor, an image with 3 channels
  x = Conv2D(1, kernel_size=5, padding='same', activation='tanh')(x)

  return Model(inputs=inp, outputs=x)

def build_generator_test(start_filters, filter_size, latent_dim):

    def add_generator_block(x, filters, filter_size, strides=2):
        x = Conv2DTranspose(filters, filter_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        return x

    inp = Input(shape=(latent_dim,))
    print(inp.shape)

    x = Dense(7 * 7 * (start_filters * 8), use_bias=False, input_shape=inp.shape)(inp)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(7, 7, start_filters * 8))(x)

    # design the generator to upsample the image 4x
    x = add_generator_block(x, start_filters * 4, filter_size, strides=1)
    x = add_generator_block(x, start_filters * 2, filter_size, strides=1)
    x = add_generator_block(x, start_filters, filter_size)
    x = add_generator_block(x, start_filters, filter_size)

    # turn the output into a 2D tensor, an image with 3 channels
    x = Conv2D(1, kernel_size=5, padding='same', activation='tanh')(x)

    return Model(inputs=inp, outputs=x)
