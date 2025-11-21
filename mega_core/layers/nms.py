# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from mega_core import _C

try:
    from apex import amp
except ImportError:
    # Fallback cuando apex no está instalado: no usamos mixed precision
    class _DummyScaleLoss(object):
        def __init__(self, loss):
            self.loss = loss
        def __enter__(self):
            return self.loss
        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyAmp(object):
        def float_function(self, func):
            return func
        def scale_loss(self, loss, optimizer):
            # Devuelve un context manager que no hace nada especial
            return _DummyScaleLoss(loss)
        def initialize(self, model, optimizer, *args, **kwargs):
            # Devuelve modelo y optimizador tal cual
            return model, optimizer

    amp = _DummyAmp()

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
