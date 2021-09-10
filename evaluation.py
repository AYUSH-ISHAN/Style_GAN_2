
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

def evaluation(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(number_of_images)):
            sample_z = torch.randn(SAMPLE_IMAGES, args.latent, device=device)

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


if __name__ == "__main__":
    
    device = "cuda"
    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        size_of_image, args.latent, args.n_mlp, channel_multiplier=multi_folds_channel
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    mean_latent = None

    evaluation(args, g_ema, device, mean_latent)
