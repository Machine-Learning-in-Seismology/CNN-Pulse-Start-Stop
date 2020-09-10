from MaLeConvSeq import MaLeConvSeq
from MaLeSimple import MaLeSimple


class MaLeNeuralNetworkFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_network(n_type,  xshape, neuron, optmizer):
        __network_classes = {
            'MaLeConvSeq': MaLeConvSeq,
            'MaLeSimple': MaLeSimple
        }
        return __network_classes[n_type](xshape, neuron, optmizer)