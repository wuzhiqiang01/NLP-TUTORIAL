import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers
from tokenizers import trainers, pre_tokenizers, processors, decoders


# 参考
# https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt#building-a-unigram-tokenizer-from-scratch
# https://huggingface.co/docs/tokenizers/components

def train_tokenizer():

    train_en_df = pd.read_csv("dataset/rawdata/ted2020/train.clean.en.csv")
    train_zh_df = pd.read_csv("dataset/rawdata/ted2020/train.clean.zh.csv")
    valid_en_df = pd.read_csv("dataset/rawdata/ted2020/valid.clean.en.csv")
    valid_zh_df = pd.read_csv("dataset/rawdata/ted2020/valid.clean.zh.csv")

    all_df = pd.concat([train_en_df[['sentence']],
                        train_zh_df[['sentence']],
                        valid_en_df[['sentence']],
                        valid_zh_df[['sentence']]])    
    dataset = Dataset.from_pandas(all_df)
    def train_corp_iter(): 
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["sentence"]


    # corpus_generator = train_corp_iter()

    # 为了演示，我们可以获取生成器的第一个批次并展示前几个元素
    # first_batch = next(corpus_generator)
    # print(first_batch[:100]) 

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.normalizer = normalizers.Sequence(
        [   
            normalizers.Nmt(),
            normalizers.NFD(),
            normalizers.Lowercase(),
        ]

    )
    # tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    # print(tokenizer.normalizer.normalize_str("我是由衷的想這麼說 , 有部份原因是因為我真的有需要"))
    # print(len(tokenizer.normalizer.normalize_str("我是由衷的想這麼說 , 有部份原因是因為我真的有需要")))

    vocab_size = 8000
    # special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
    special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>"]
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        # shrinking_factor=1.0,
        shrinking_factor=0.8,
        n_sub_iterations=3,
    )
    tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)


    for text in train_en_df['sentence'].tolist()[:10]:
        encoding = tokenizer.encode(text)
        print(encoding.tokens)
    for text in train_zh_df['sentence'].tolist()[:10]:
        encoding = tokenizer.encode(text)
        print(encoding.tokens)

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    print(bos_token_id, eos_token_id)


    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        special_tokens=[("<bos>", bos_token_id), ("<eos>", eos_token_id)],
    )

    tokenizer.decoder = decoders.Metaspace()

    for text in train_en_df['sentence'].tolist()[:10]:
        encoding = tokenizer.encode(text)
        print(encoding.tokens)
        print(encoding.ids)
        print(tokenizer.decode(encoding.ids))
    for text in train_zh_df['sentence'].tolist()[:10]:
        encoding = tokenizer.encode(text)
        print(encoding.tokens)
        print(encoding.ids)
        print(tokenizer.decode(encoding.ids))

    tokenizer.save("tokenizer.json")

    new_tokenizer = Tokenizer.from_file("tokenizer.json")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=new_tokenizer,
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    for text in train_en_df['sentence'].tolist()[:10]:
        encoding = wrapped_tokenizer(text)
        print(encoding)
    for text in train_zh_df['sentence'].tolist()[:10]:
        encoding = wrapped_tokenizer(text)
        print(encoding)


def inference():
    train_en_df = pd.read_csv("dataset/rawdata/ted2020/train.clean.en.csv")
    train_zh_df = pd.read_csv("dataset/rawdata/ted2020/train.clean.zh.csv")
    new_tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=new_tokenizer,
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    for text in train_zh_df['sentence'].tolist()[:5]:
        encoding = tokenizer.tokenize(text)
        print(encoding)
    for text in train_en_df['sentence'].tolist()[:5]:
        encoding = tokenizer.tokenize(text)
        print(encoding)


if __name__ == "__main__":
    # train_tokenizer()
    inference()
