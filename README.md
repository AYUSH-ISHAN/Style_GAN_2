<h1> Style_GAN_2 </h1>

# Introduction:

This is the simple implementation of Style GANS 2 paper (link - <a href = "https://arxiv.org/pdf/1912.04958.pdf">Style GAN 2</a>). <br>

<h4> Abstract by NVIDIA Team in there paper :</h4>

The style-based GAN architecture (StyleGAN) yields
state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of
its characteristic artifacts, and propose changes in both
model architecture and training methods to address them.
In particular, we redesign the generator normalization, revisit progressive growing, and regularize the generator to
encourage good conditioning in the mapping from latent
codes to images. In addition to improving image quality,
this path length regularizer yields the additional benefit that
the generator becomes significantly easier to invert. This
makes it possible to reliably attribute a generated image to
a particular network. We furthermore visualize how well
the generator utilizes its output resolution, and identify a
capacity problem, motivating us to train larger models for
additional quality improvements. Overall, our improved
model redefines the state of the art in unconditional image
modeling, both in terms of existing distribution quality metrics as well as perceived image quality.

# Generator and Descriminator Model :

