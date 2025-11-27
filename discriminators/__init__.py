from .stylegan3_d import StyleGAN3Discriminator
from .stylegan2ada_d import StyleGAN2ADADiscriminator
from .diffusion_d import DiffusionStyleGAN2Discriminator
from .dummy_discriminator import DummyDiscriminator  # optional fallback

def get_all_discriminators():
    return [
        StyleGAN3Discriminator("models/stylegan3-t-ffhq-1024x1024.pkl"),
        StyleGAN2ADADiscriminator("models/ffhq-1024-stylegan2-ada.pkl"),
        DiffusionStyleGAN2Discriminator("models/diffusion-stylegan2-ffhq.pkl"),
        DummyDiscriminator(),  # OPTIONAL so system never breaks
    ]
