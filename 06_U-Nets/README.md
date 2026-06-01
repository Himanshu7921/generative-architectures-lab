# Note

U-Net is **not a generative model**.

The original U-Net architecture was introduced for **image segmentation**, where the goal is to assign a class label to every pixel in an image. It was designed to accurately localize objects while preserving fine spatial details through its encoder-decoder structure and skip connections.

So why is U-Net included in this generative models repository?

Although U-Net itself is not generative, many modern generative AI systems use architectures that are heavily inspired by or built upon U-Net. In particular, diffusion-based image generation models use advanced variants of U-Net as their core neural network.

The reason is that U-Nets are excellent at:

- Capturing high-level semantic information through downsampling.
- Preserving fine-grained spatial details through skip connections.
- Reconstructing high-resolution outputs through upsampling.
- Combining global context with local image features.

These properties make U-Net an ideal backbone for tasks such as image denoising, image restoration, and diffusion-based generation.

I am studying U-Net in this repository because understanding its architecture provides the foundation for understanding more advanced generative models, including:

- Denoising Diffusion Probabilistic Models (DDPMs)
- Stable Diffusion
- Latent Diffusion Models
- Modern image generation systems based on diffusion

This section focuses on understanding the original U-Net paper, implementing the architecture from scratch, and building the intuition required for studying modern generative AI architectures.