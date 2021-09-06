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

# Discriminator and Generator Model :

<h3><B>1. Discriminator Model:</B></h3>

                                    Discriminator  Parameters  Buffers  Output shape      Datatype
                                    ---            ---         ---      ---               ---     
                                    b32.fromrgb    2048        16       [4, 512, 32, 32]  float16 
                                    b32.skip       262144      16       [4, 512, 16, 16]  float16 
                                    b32.conv0      2359808     16       [4, 512, 32, 32]  float16 
                                    b32.conv1      2359808     16       [4, 512, 16, 16]  float16 
                                    b32            -           16       [4, 512, 16, 16]  float16 
                                    b16.skip       262144      16       [4, 512, 8, 8]    float16 
                                    b16.conv0      2359808     16       [4, 512, 16, 16]  float16 
                                    b16.conv1      2359808     16       [4, 512, 8, 8]    float16 
                                    b16            -           16       [4, 512, 8, 8]    float16 
                                    b8.skip        262144      16       [4, 512, 4, 4]    float16 
                                    b8.conv0       2359808     16       [4, 512, 8, 8]    float16 
                                    b8.conv1       2359808     16       [4, 512, 4, 4]    float16 
                                    b8             -           16       [4, 512, 4, 4]    float16 
                                    b4.mbstd       -           -        [4, 513, 4, 4]    float32 
                                    b4.conv        2364416     16       [4, 512, 4, 4]    float32 
                                    b4.fc          4194816     -        [4, 512]          float32 
                                    b4.out         513         -        [4, 1]            float32 
                                    ---            ---         ---      ---               ---     
                                    Total          21507073    224      -                 -      
                                    
                                    
<h3><B>2. Generator Model:</B></h3>
                                    
                                    Generator            Parameters  Buffers  Output shape      Datatype
                                    ---                  ---         ---      ---               ---     
                                    mapping.fc0          262656      -        [4, 512]          float32 
                                    mapping.fc1          262656      -        [4, 512]          float32 
                                    mapping.fc2          262656      -        [4, 512]          float32 
                                    mapping.fc3          262656      -        [4, 512]          float32 
                                    mapping.fc4          262656      -        [4, 512]          float32 
                                    mapping.fc5          262656      -        [4, 512]          float32 
                                    mapping.fc6          262656      -        [4, 512]          float32 
                                    mapping.fc7          262656      -        [4, 512]          float32 
                                    mapping              -           512      [4, 8, 512]       float32 
                                    synthesis.b4.conv1   2622465     32       [4, 512, 4, 4]    float32 
                                    synthesis.b4.torgb   264195      -        [4, 3, 4, 4]      float32 
                                    synthesis.b4:0       8192        16       [4, 512, 4, 4]    float32 
                                    synthesis.b4:1       -           -        [4, 512, 4, 4]    float32 
                                    synthesis.b8.conv0   2622465     80       [4, 512, 8, 8]    float16 
                                    synthesis.b8.conv1   2622465     80       [4, 512, 8, 8]    float16 
                                    synthesis.b8.torgb   264195      -        [4, 3, 8, 8]      float16 
                                    synthesis.b8:0       -           16       [4, 512, 8, 8]    float16 
                                    synthesis.b8:1       -           -        [4, 512, 8, 8]    float32 
                                    synthesis.b16.conv0  2622465     272      [4, 512, 16, 16]  float16 
                                    synthesis.b16.conv1  2622465     272      [4, 512, 16, 16]  float16 
                                    synthesis.b16.torgb  264195      -        [4, 3, 16, 16]    float16 
                                    synthesis.b16:0      -           16       [4, 512, 16, 16]  float16 
                                    synthesis.b16:1      -           -        [4, 512, 16, 16]  float32 
                                    synthesis.b32.conv0  2622465     1040     [4, 512, 32, 32]  float16 
                                    synthesis.b32.conv1  2622465     1040     [4, 512, 32, 32]  float16 
                                    synthesis.b32.torgb  264195      -        [4, 3, 32, 32]    float16 
                                    synthesis.b32:0      -           16       [4, 512, 32, 32]  float16 
                                    synthesis.b32:1      -           -        [4, 512, 32, 32]  float32 
                                    ---                  ---         ---      ---               ---     
                                    Total                21523475    3392     -                 -       


