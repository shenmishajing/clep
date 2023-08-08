from .clep import CLEP
from .ecg_conv_transformer import ECGConvTransformer


class ConvCLEP(ECGConvTransformer, CLEP):
    pass
