from discriminators.base_discriminator import BaseDiscriminator

class StyleGAN3D(BaseDiscriminator):
    def __init__(self, checkpoint_path):
        super().__init__("StyleGAN3-D")
        self.device = "cuda"
        self.checkpoint = checkpoint_path
        self.load_model()

    def load_model(self):
        import torch
        import legacy
        with open(self.checkpoint, 'rb') as f:
            self.model = legacy.load_network_pkl(f)['D']
        self.model.to(self.device).eval()

    def preprocess(self, img):
        import torch
        img = img.resize((1024, 1024))
        img = torch.from_numpy(np.asarray(img)).permute(2,0,1)
        img = img.unsqueeze(0).to(self.device) / 127.5 - 1
        return img

    def forward(self, img):
        x = self.preprocess(img)
        out = self.model(x)
        score = out.item()
        return score, None
