from .base_discriminator import BaseDiscriminator
import numpy as np

class DummyDiscriminator(BaseDiscriminator):
    def __init__(self):
        super().__init__("DummyD")

    def score(self, pil_image):
        # just returns random score for testing
        return float(np.random.uniform(-3, 3))
