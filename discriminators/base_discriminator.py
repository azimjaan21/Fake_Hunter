class BaseDiscriminator:
    def __init__(self, name):
        self.name = name

    def load_model(self):
        raise NotImplementedError

    def preprocess(self, img):
        raise NotImplementedError

    def forward(self, img):
        """
        Returns:
            score (float) - real vs fake logit
            embedding (np.array) - optional
        """
        raise NotImplementedError
