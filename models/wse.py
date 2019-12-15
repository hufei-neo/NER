from mxnet import nd
from mxnet.gluon import nn


class WeightedSoftmaxCE(nn.Block):
    def __init__(self, sparse_label=True, from_logits=False, **kwargs):
        super(WeightedSoftmaxCE, self).__init__(**kwargs)
        with self.name_scope():
            self.sparse_label = sparse_label
            self.from_logits = from_logits

    def forward(self, pred, label, class_weight, depth=None):
        if self.sparse_label:
            label = nd.reshape(label, shape=(-1,))
            label = nd.one_hot(label, depth)
        if not self.from_logits:
            pred = nd.log_softmax(pred, -1)
        weight_label = nd.broadcast_mul(label, class_weight)
        loss = -nd.sum(pred * weight_label, axis=-1)
        return loss
