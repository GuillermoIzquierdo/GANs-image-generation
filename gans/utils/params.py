import os
import pandas as pd

SPATIAL_DIM = int(os.environ.get('SPATIAL_DIM'))
LATENT_DIM_GAN = int(os.environ.get('LATENT_DIM_GAN'))
FILTER_SIZE = int(os.environ.get('FILTER_SIZE'))
NET_CAPACITY = int(os.environ.get('NET_CAPACITY'))
BATCH_SIZE_GAN = int(os.environ.get('BATCH_SIZE_GAN'))
PROGRESS_INTERVAL = int(os.environ.get('PROGRESS_INTERVAL'))
ROOT_DIR = os.environ.get('ROOT_DIR')
LOCAL_DATA_PATH = os.environ.get('LOCAL_DATA_PATH')
