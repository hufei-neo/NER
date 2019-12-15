from gluonnlp.model.attention_cell import DotProductAttentionCell
from gluonnlp.model.attention_cell import MultiHeadAttentionCell
from mxnet.gluon import rnn, nn


class PositionTagging(nn.Block):
    def __init__(self, embed_size, vocab, hidden, dense, unit=0, headers=0):
        """
        Position Tagging with MHA
        :param embed_size: Int
        Number of embedding dimension
        :param vocab: Int
        Number of token index size
        :param hidden: Int
        Hidden dimension of biLSTM
        :param dense: List of Int
        List of int to create
        :param unit:
        :param headers:
        """
        super(PositionTagging, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(vocab, embed_size)
            # bidirectional LSTM
            self.biLSTM = rnn.BidirectionalCell(
                rnn.LSTMCell(hidden_size=hidden // 2),
                rnn.LSTMCell(hidden_size=hidden // 2)
            )
            if headers > 0:
                cell = DotProductAttentionCell(scaled=True, dropout=0.2)
                cell = MultiHeadAttentionCell(base_cell=cell, use_bias=False,
                                              query_units=unit, key_units=unit,
                                              value_units=unit,
                                              num_heads=headers)
                self.att = cell
            else:
                self.att = None
            self.dense = nn.Sequential()
            for each in dense[:-1]:
                self.dense.add(nn.Dense(each, activation="softrelu"))
                self.dense.add(nn.BatchNorm())
            self.dense.add(nn.Dense(dense[-1], activation="sigmoid"))

    def begin_state(self, func, **kwargs):
        self.biLSTM.begin_state(func=func, **kwargs)

    def forward(self, x, mask=None, mem_mask=None):
        """
        Use biLSTM-MHA to predict results based on input sentence
        :param x: NDArray
        Shape = (batch size, sentence length)
        :param mask: NDArray
        shape (batch_size, seq_len)
        :param mem_mask: NDArray
        shape (batch_size, seq_len, seq_len)
        :return: NDArray:
        the matrix of tagging score, shape (batch_size, seq_len, dense[-1] unit)
        """
        batch_size, seq_len = x.shape
        x = self.embedding(x)
        x = x * mask.expand_dims(-1)
        # batch size * time step size * embedding dimension
        outputs, hidden = self.biLSTM.unroll(length=x.shape[1], inputs=x,
                                             layout='NTC', merge_outputs=True)
        if self.att is not None:
            outputs, att = self.att(outputs, outputs, outputs, mem_mask)
        outputs = outputs.reshape((batch_size * seq_len, -1))
        outputs = self.dense(outputs)
        return outputs.reshape((batch_size, seq_len, -1))
