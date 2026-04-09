class FeatureExtractor:
    def extract(self, state, action):
        raise NotImplementedError
    
    def num_features(self):
        raise NotImplementedError