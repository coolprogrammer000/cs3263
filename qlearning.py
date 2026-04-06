import json


class LinearQFunction:
    def __init__(self, feature_extractor, alpha=0.1, weights=None, default_q_value=0.0):
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.num_features = feature_extractor.num_features()

        if weights is None:
            self.weights = [default_q_value for _ in range(self.num_features)]
        else:
            if len(weights) != self.num_features:
                raise ValueError(
                    f"weights length {len(weights)} does not match "
                    f"num_features {self.num_features}"
                )
            self.weights = list(weights)

    def get_q_value(self, state, action):
        feature_values = self.feature_extractor.extract(state, action)

        if len(feature_values) != self.num_features:
            raise ValueError(
                f"Feature extractor returned {len(feature_values)} features, "
                f"expected {self.num_features}"
            )

        q_value = 0.0
        for i in range(self.num_features):
            q_value += self.weights[i] * feature_values[i]
        return q_value

    def update(self, state, action, delta):
        feature_values = self.feature_extractor.extract(state, action)

        if len(feature_values) != self.num_features:
            raise ValueError(
                f"Feature extractor returned {len(feature_values)} features, "
                f"expected {self.num_features}"
            )

        for i in range(self.num_features):
            self.weights[i] += self.alpha * delta * feature_values[i]

    def get_weights(self):
        return list(self.weights)

    def save_weights(self, filepath):
        data = {
            "num_features": self.num_features,
            "alpha": self.alpha,
            "weights": self.weights,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_weights(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_num_features = data["num_features"]
        loaded_weights = data["weights"]

        if loaded_num_features != self.num_features:
            raise ValueError(
                f"Saved weights expect {loaded_num_features} features, "
                f"but current feature extractor has {self.num_features}"
            )

        if len(loaded_weights) != self.num_features:
            raise ValueError(
                f"Saved weights length {len(loaded_weights)} does not match "
                f"num_features {self.num_features}"
            )

        self.alpha = data.get("alpha", self.alpha)
        self.weights = list(loaded_weights)