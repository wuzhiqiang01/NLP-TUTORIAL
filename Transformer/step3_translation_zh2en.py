import os
import torch
import time
import GPUtil
import numpy as np
import pandas as pd
from model import make_model
import torch.utils.data as data
from tokenizers import Tokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import LabelSmoothing, rate, subsequent_mask, SimpleLossCompute
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast
from torch.nn.parallel import DistributedDataParallel as DDP


class TransalationDataset(data.Dataset):
    def __init__(self, src_path, tgt_path, tokenizer):
        super().__init__()
        train_en_df = pd.read_csv(src_path)
        train_zh_df = pd.read_csv(tgt_path)
        train_en_df = train_en_df.rename(columns={'sentence': 'src'})
        train_zh_df = train_zh_df.rename(columns={'sentence': 'tgt'})
        self.df = pd.concat([train_en_df[['src']],
                        train_zh_df[['tgt']]],
                        axis=1,
                        )
        self.tokenizer = tokenizer

    
    def __getitem__(self, index):
        train_pair = self.df.iloc[index]
        src_sentence = train_pair['src']
        tgt_sentence = train_pair['tgt']

        src_token = self.tokenizer(src_sentence, max_length=128, padding="max_length", truncation=True, return_tensors="np")
        tgt_token = self.tokenizer(tgt_sentence, max_length=128, padding="max_length", truncation=True, return_tensors="np")
        return {"src": src_token, "tgt": tgt_token}
    
    def __len__(self):
        return self.df.shape[0]


def collate_batch(batch, device):
    src_input_ids = []
    src_attention_mask = []
    tgt_input_ids = []
    tgt_attention_mask = []
    for data_dict in batch:
        src_token = data_dict['src']
        tgt_token = data_dict['tgt']

        src_input_ids.append(src_token['input_ids'])
        src_attention_mask.append(src_token["attention_mask"])

        tgt_input_ids.append(tgt_token["input_ids"])
        tgt_attention_mask.append(tgt_token["attention_mask"])
    
    src_input_ids = np.concatenate(src_input_ids, axis=0)
    src_attention_mask = np.concatenate(src_attention_mask, axis=0)
    tgt_input_ids = np.concatenate(tgt_input_ids, axis=0)
    tgt_attention_mask = np.concatenate(tgt_attention_mask, axis=0)

    src_input_ids_tensor = torch.from_numpy(src_input_ids).contiguous().to(device)
    src_attention_mask_tensor = torch.from_numpy(src_attention_mask).contiguous().to(device)
    tgt_input_ids_tensor = torch.from_numpy(tgt_input_ids).contiguous().to(device)
    tgt_attention_mask_tensor = torch.from_numpy(tgt_attention_mask).contiguous().to(device)

    return{
        "src_input_ids": src_input_ids_tensor,
        "src_attention_mask": src_attention_mask_tensor,
        "tgt_input_ids": tgt_input_ids_tensor,
        "tgt_attention_mask": tgt_attention_mask_tensor,
    }

def create_dataloaders(
    device,
    tokenizer,
    is_distributed,
):


    def collate_fn(batch):
        return collate_batch(
            batch,
            device,
        )

    train_dataset = TransalationDataset(
        CFG.train_src_path,
        CFG.train_tgt_path,
        tokenizer
    )
    valid_dataset = TransalationDataset(
        CFG.valid_src_path,
        CFG.valid_tgt_path,
        tokenizer
    )

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    valid_dataloader = data.DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
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
    checkpoint_path = "checkpoint_en2zh"
    file_prefix = "en2zh_model_"

    ngpus_per_node = 2
    batch_size = 32

    # train
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    train_src_path = "dataset/rawdata/ted2020/train.clean.en.csv"
    train_tgt_path = "dataset/rawdata/ted2020/train.clean.zh.csv"

    valid_src_path = "dataset/rawdata/ted2020/valid.clean.en.csv"
    valid_tgt_path = "dataset/rawdata/ted2020/valid.clean.zh.csv"


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
        batch = Batch(b, pad_idx, )
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


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, batch, pad):  # 2 = <blank>
        self.src = batch["src_input_ids"]
        src_mask = batch["src_attention_mask"]

        tgt = batch["tgt_input_ids"]

        self.src_mask = src_mask.unsqueeze(-2)
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

def train(rank):
    print(f"Train worker process using GPU: {rank} for training", flush=True)
    torch.cuda.set_device(rank)

    new_tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=new_tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    d_model = 512
    model = make_model(
        vocab_size,
        vocab_size,
        N=6,
        d_model=d_model)
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
        size=8000,
        padding_idx=pad_idx,
        smoothing=0.1
    )
    criterion.cuda(rank)
    
    # print(CFG.batch_size)

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
    
    train_dataloader, valid_dataloader = create_dataloaders(
        device=rank,
        tokenizer=tokenizer,
        is_distributed=CFG.distributed
    )

    for epoch in range(CFG.num_epochs):
        if CFG.distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{rank}] Epoch {epoch} Training ====", flush=True)

        _ = run_epoch(
            train_dataloader,
            model,
            SimpleLossCompute(module.classifier, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=CFG.accum_iter,
            pad_idx=pad_idx,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "{}/{}{}.pth".format(CFG.checkpoint_path,

                                             CFG.file_prefix, epoch)
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
        file_path = "{}/{}final.pth".format(CFG.checkpoint_path,
                                        CFG.file_prefix)
        torch.save(module.state_dict(), file_path)


# def test_dataloader():
#     dataset = TransalationDataset()
#     print(dataset[0])

#     train_dataloader = data.DataLoader(
#         dataset,
#         batch_size=128,
#         collate_fn=collate_fn
#     )

#     for i, batch in enumerate(train_dataloader):
#         print(i)
#         print(batch)


def main():
    if CFG.distributed:
        ngpus = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        print(f"Number of GPUs detected: {ngpus}")
        print("Spawning training processes ...")
        mp.spawn(
            train,
            nprocs=ngpus,
            args=()
        )
    else:
        train(
            0,
            )

if __name__ == "__main__":
    main()