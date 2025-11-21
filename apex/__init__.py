class DummyScaleLoss:
    def __init__(self, loss):
        self.loss = loss
    def __enter__(self):
        return self.loss
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyAmp:
    def scale_loss(self, loss, optimizer):
        # No-op wrapper for loss
        return DummyScaleLoss(loss)
    def initialize(self, model, optimizer, **kwargs):
        # Devuelve el modelo y el optimizador tal cual
        return model, optimizer
    def float_function(self, func):
        # Devuelve la función sin modificar
        return func

amp = DummyAmp()
