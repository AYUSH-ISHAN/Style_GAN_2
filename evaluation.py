
import argparse
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

###   Some global parameters declaration.

number_of_images = 1     ##  Number of images to be generated in one call.
multi_folds_channel = 1     # (may change it to 2, on basis of results)
size_of_image = 32      # chage it to 32 if some error occurs
TRUNCATION_MEAN = 4096      # number of vectors to calculate mean for the truncation
TRUNCATION = 1      # truncation ratio
SAMPLE_IMAGES = 1    # number of samples to be generated for each image

###  The evaluation function mainly responsible for generating images from trained model.
  
device = "cuda"

LATENT = 512
N_MLP = 8


def evaluation(path, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(number_of_images)):
            sample_z = torch.randn(SAMPLE_IMAGES, LATENT, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=TRUNCATION, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def final_evaluation(path):

    g_ema = Generator(
        size_of_image, LATENT, N_MLP, channel_multiplier=multi_folds_channel
    ).to(device)
    checkpoint = torch.load(path)

    g_ema.load_state_dict(checkpoint["g_ema"])

    mean_latent = None

    evaluation(path, g_ema, device, mean_latent)
