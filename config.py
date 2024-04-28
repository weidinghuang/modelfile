import json
import tensorflow as tf
import six
class PARAMS(object):
    def __init__(self,
                 BatchSize=32,
                 VocabSize=130,
                 HiddenSize=768,
                 MaxSeqLength=60,
                 DropoutRate=0.1,
                 AttenDropoutRate=0.1,
                 AttentionHiddenSize=1248,
                 IntermediateSize=3072,
                 HeadNum=12,
                 NumTokenType=2,
                 PosMaxLength=512,
                 NumBlocks=12,
                 DomainLabelSize=4,
                 IntentLabelSize=10,
                 SemanticLabelSize=20,
                 VocabFilePath="/repos/model/uncased_L-12_H-768_A-12/vocab.txt",
                 EnableSwapTag=True,
                 DoLowerCase=True,
                 OutputAribitraryTargetsForInfeasibleExamples=True):
        self.BatchSize=BatchSize
        self.VocabSize = VocabSize
        self.HiddenSize=HiddenSize
        self.DropoutRate=DropoutRate
        self.MaxSeqLength=MaxSeqLength
        self.AttenDropoutRate=AttenDropoutRate
        self.AttentionHiddenSize=AttentionHiddenSize
        self.IntermediateSize=IntermediateSize
        self.HeadNum=HeadNum
        self.NumTokenType=NumTokenType
        self.NumBlocks=NumBlocks
        self.DomainLabelSize=DomainLabelSize
        self.IntentLabelSize=IntentLabelSize
        self.SemanticLabelSize=SemanticLabelSize
        self.VocabFilePath=VocabFilePath
        self.EnableSwapTag=EnableSwapTag
        self.DoLowerCase=DoLowerCase
        self.OutputAribitraryTargetsForInfeasibleExamples=OutputAribitraryTargetsForInfeasibleExamples
        self.PosMaxLength=PosMaxLength

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = PARAMS(VocabSize=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def update(self,BatchSize=8,
                 VocabSize=21128,
                 HiddenSize=768,
                 MaxSeqLength=60,
                 DropoutRate=0.1,
                 AttenDropoutRate=0.1,
                 AttentionHiddenSize=1248,
                 IntermediateSize=3072,
                 HeadNum=12,
                 NumTokenType=2,
                 PosMaxLength=512,
                 NumBlocks=12,
                 DomainLabelSize=17+2,
                 IntentLabelSize = 17,
                 VocabFilePath="../bert_base/RoBERTa-tiny-clue/vocab.txt",
                 EnableSwapTag=True,
                 DoLowerCase=True,
                 OutputAribitraryTargetsForInfeasibleExamples=False):
        self.BatchSize = BatchSize
        self.VocabSize = VocabSize
        self.HiddenSize = HiddenSize
        self.DropoutRate = DropoutRate
        self.MaxSeqLength = MaxSeqLength
        self.AttenDropoutRate = AttenDropoutRate
        self.AttentionHiddenSize = AttentionHiddenSize
        self.IntermediateSize = IntermediateSize
        self.HeadNum = HeadNum
        self.NumTokenType = NumTokenType
        self.NumBlocks = NumBlocks
        self.DomainLabelSize = DomainLabelSize
        self.IntentLabelSize = IntentLabelSize
        self.VocabFilePath = VocabFilePath
        self.EnableSwapTag = EnableSwapTag
        self.DoLowerCase = DoLowerCase
        self.OutputAribitraryTargetsForInfeasibleExamples = OutputAribitraryTargetsForInfeasibleExamples
        self.PosMaxLength = PosMaxLength


CONFIG = PARAMS()
print(CONFIG.DomainLabelSize)
