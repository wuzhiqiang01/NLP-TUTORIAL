import os
import spacy
import torch
import time
import GPUtil
import torch.distributed as dist
from model import make_model
import torch.nn.functional as F
import torch.utils.data as data
from utils import LabelSmoothing
import torch.multiprocessing as mp
import torchtext.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torch.nn.parallel import DistributedDataParallel as DDP


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])  # 用分词器对句子进行分词

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(root="./dataset", language_pair=("de", "en"))

    # 生成词表
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(root="./dataset", language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    # 设置默认的词表，找不到的默认返回为<unk>
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not os.path.exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            F.pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            F.pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=False,
):

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )
    
    train_iter, valid_iter, _ = datasets.Multi30k(
        root="./dataset", language_pair=("de", "en") 
    )

    train_iter_map = to_map_style_dataset(train_iter)
    valid_iter_map = to_map_style_dataset(valid_iter)

    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = data.DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    valid_dataloader = data.DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn
    )
    return train_dataloader, valid_dataloader


class CFG(): 
    distributed = False
    num_epochs = 8
    accum_iter = 10
    base_lr = 1.0
    max_padding = 72
    warmup = 3000
    file_prefix = "multi30k_model_"

    ngpus_per_node = 2
    batch_size = 100

    # train
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def run_epoch(
    data_iter,
    model,
    loss_compute=None,
    optimizer=None,
    scheduler=None,
    mode="train",
    accum_iter=1,
    pad_idx="",
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, b in enumerate(data_iter):
        batch = Batch(b[0], b[1], pad_idx)
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            CFG.step += 1
            CFG.samples += batch.src.shape[0]
            CFG.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                CFG.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

def train(rank, vocab_src, vocab_tgt, spacy_de, spacy_en):
    print(f"Train worker process using GPU: {rank} for training", flush=True)
    torch.cuda.set_device(rank)

    pad_idx = vocab_tgt['<blank>']
    d_model = 512
    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=6)
    model.cuda(rank)
    module = model

    is_main_process = True
    if CFG.distributed:
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=CFG.ngpus_per_node,
        )
        model = DDP(model, device_ids=[rank])
        module = model.module
        is_main_process = rank == 0
        
    criterion = LabelSmoothing(
        size=len(vocab_tgt),
        padding_idx=pad_idx,
        smoothing=0.1
    )
    criterion.cuda(rank)
    
    # print(CFG.batch_size)
    train_dataloader, valid_dataloader = create_dataloaders(
        rank,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        CFG.batch_size,
        CFG.max_padding,
        CFG.distributed
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = CFG.base_lr,
        betas = (0.9, 0.98),
        eps = 1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=CFG.warmup
        ),
    )

    for epoch in range(CFG.num_epochs):
        if CFG.distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{rank}] Epoch {epoch} Training ====", flush=True)

        # for i, b in enumerate(valid_dataloader):
        #     batch = Batch(b[0], b[1], pad_idx)
        #     print(i)
        # for b in train_dataloader:
        #     print(b)


        _ = run_epoch(
            train_dataloader,
            model,
            SimpleLossCompute(module.classifier, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=CFG.accum_iter,
            pad_idx=pad_idx
        )
        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (CFG.file_prefix, epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{rank}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            valid_dataloader,
            model,
            SimpleLossCompute(module.classifier, criterion),
            mode="eval",
            pad_idx=pad_idx,
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % CFG.file_prefix
        torch.save(module.state_dict(), file_path)

def main():

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    if CFG.distributed:
        ngpus = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        print(f"Number of GPUs detected: {ngpus}")
        print("Spawning training processes ...")
        mp.spawn(
            train,
            nprocs=ngpus,
            args=(vocab_src, vocab_tgt, spacy_de, spacy_en)
        )
    else:
        train(
            0,
            vocab_src,
            vocab_tgt,
            spacy_de,
            spacy_en
        )


if __name__ == "__main__":
    # spacy_de, spacy_en = load_tokenizers()
    # vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    main()