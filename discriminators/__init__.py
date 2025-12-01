import os
import sys

# Make the bundled StyleGAN code discoverable so `import legacy` works when loading checkpoints.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_STYLEGAN_DIR = os.path.join(_PROJECT_ROOT, "external", "stylegan2-ada-pytorch")
if _STYLEGAN_DIR not in sys.path:
    sys.path.insert(0, _STYLEGAN_DIR)

from .stylegan3_d import StyleGAN3Discriminator
from .stylegan2ada_d import StyleGAN2ADADiscriminator
# from .diffusion_d import DiffusionGANDiscriminator

def get_all_discriminators():
    return [
        StyleGAN3Discriminator(r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\models\stylegan3-t-ffhq-1024x1024.pkl"),
        StyleGAN2ADADiscriminator(r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\models\ffhq-1024-stylegan2-ada.pkl"),
       # DiffusionGANDiscriminator(r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\models\diffusion-stylegan2-ffhq.pkl"),
       # DummyDiscriminator(),  # OPTIONAL so system never breaks
    ]
