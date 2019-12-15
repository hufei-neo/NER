import pandas as pd


class Config(object):
    """
    structured self attention based sentence analysis parameters
    """

    def __init__(self):
        self.max_seq = 40
        # model file info
        # model file
        # encoder
        self.mask = False
        self.embed_size = 100
        self.vocab_size = 1
        self.hidden = 2
        # decoder
        self.tag2idx = {
            0: 0,
            1: 1,
        }
        self.headers = 10
        self.unit = 100
        self.dense = len(self.tag2idx)
        # model self
        self.save_path = './checkpoints/'
        self.model_file = "./P_T.params"
        self.vocab_file = "./P_T.vocab"
        # token
        self.default = "<unk>"
        self.pad = "<pad>"
        self.eos = "<eos>"

    def to_json(self, file_name):
        f = []
        for each in self.__dict__:
            f.append({"key": each, "value": self.__getattribute__(each)})
        f = pd.DataFrame(f)
        f.to_json(file_name, orient='index')

    def from_json(self, file_name):
        _att = pd.read_json(file_name, orient='index')
        _att.index = _att["key"]
        for each in _att.index:
            self.__setattr__(each, _att.loc[each, "value"])
