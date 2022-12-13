from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from gans.models.discriminator import build_discriminator
from gans.models.generator import build_generator
from gans.utils.params import NET_CAPACITY, SPATIAL_DIM, FILTER_SIZE, LATENT_DIM_GAN

def construct_models(verbose=False):
    ### discriminator
    discriminator = build_discriminator(NET_CAPACITY, SPATIAL_DIM, FILTER_SIZE)
    # compile discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])

    ### generator
    # do not compile generator
    generator = build_generator(NET_CAPACITY, FILTER_SIZE, LATENT_DIM_GAN)

    ### DCGAN
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])

    if verbose:
        generator.summary()
        discriminator.summary()
        gan.summary()

    return generator, discriminator, gan
