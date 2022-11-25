class BaseModel():
    """Abstract Class, can be refactored."""

    def train(self, data):
        raise NotImplementedError()

    def test(self, model, data):
        raise NotImplementedError()