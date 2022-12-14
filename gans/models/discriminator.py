from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU

# function for building the discriminator layers
def build_discriminator(start_filters, spatial_dim, filter_size):

    # function for building a CNN block for downsampling the image
    def add_discriminator_block(x, filters, filter_size):
      x = Conv2D(filters, filter_size, padding='same')(x)
      x = BatchNormalization()(x)
      x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.3)(x)
      return x

    # input is an image with shape spatial_dim x spatial_dim and 1 channel
    inp = Input(shape=(spatial_dim, spatial_dim, 1))

    # design the discrimitor to downsample the image 4x
    x = add_discriminator_block(inp, start_filters, filter_size)
    x = add_discriminator_block(x, start_filters * 2, filter_size)
    x = add_discriminator_block(x, start_filters * 4, filter_size)
    x = add_discriminator_block(x, start_filters * 8, filter_size)

    # average and return a binary output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inp, outputs=x)

def build_discriminator_test(start_filters, spatial_dim, filter_size):

    # function for building a CNN block for downsampling the image
    def add_discriminator_block(x, filters, filter_size):
      x = Conv2D(filters, filter_size, padding='same')(x)
      x = BatchNormalization()(x)
      print(x.shape)
      x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
      x = BatchNormalization()(x)
      print(x.shape)
      x = LeakyReLU(0.3)(x)
      return x

    # input is an image with shape spatial_dim x spatial_dim and 1 channel
    inp = Input(shape=(28, 28, 1))
    print(f'Input Shape: {inp.shape}')

    # design the discrimitor to downsample the image 4x
    x = add_discriminator_block(inp, start_filters, filter_size)
    x = add_discriminator_block(x, start_filters * 2, filter_size)
    x = add_discriminator_block(x, start_filters * 4, filter_size)
    x = add_discriminator_block(x, start_filters * 8, filter_size)

    # average and return a binary output
    x = GlobalAveragePooling2D()(x)
    print(f'Average Pooling: {x.shape}')
    x = Dense(1, activation='sigmoid')(x)
    print(f'Output Shape: {x.shape}')

    return Model(inputs=inp, outputs=x)
