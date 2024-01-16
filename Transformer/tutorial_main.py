import torch
import time
from model import make_model
import torch.optim as optim
from utils import LabelSmoothing


class CFG:
    vocab_size = 11
    batch_size = 80
    nbatches = 20

    # label smoothing
    padding_idx = 0
    smoothing = 0.0

    # model
    N = 2

    # train
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class Batch():
    # 将随机组成的数据构成batch
    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
            
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


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


def data_gen(vocab_size, batch_size, nbatches):
    # 随机生成数据
    for i in range(nbatches):
        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


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
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
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


def test_simple_model():
    criterion = LabelSmoothing(
        CFG.vocab_size,
        CFG.padding_idx,
        CFG.smoothing
        )
    model = make_model(
        src_vocab=CFG.vocab_size,
        tgt_vocab=CFG.vocab_size,
        N=CFG.N)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.5, 
        betas = (0.9, 0.98),
        eps = 1e-9
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step,
                                    model_size=model.src_embed.d_model,
                                    factor=1.0, warmup=400)
    )
    
    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(CFG.vocab_size, batch_size, 20),
            model,
            SimpleLossCompute(model.classifier, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
    

def main():
    # data = data_gen(CFG.vocab_size, CFG.batch_size, CFG.nbatches)
    # for i, batch in enumerate(data):
    #     print(batch.src)
    test_simple_model()


if __name__ == "__main__":
    main()