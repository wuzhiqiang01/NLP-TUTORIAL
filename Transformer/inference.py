import torch
from translation_ge2en import load_tokenizers, load_vocab
from translation_ge2en import create_dataloaders, Batch, make_model, subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            ys, memory, subsequent_mask(ys.size(1)).type_as(src.data), src_mask
        )
        prob = model.classifier(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def main():
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    pad_idx = vocab_tgt['<blank>']
    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False
    )

    print("Loading model ...")


    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=6)
    
    model.load_state_dict(
        torch.load("checkpoint/multi30k_model_final.pth", map_location=torch.device("cpu"))
    )
    # print(model)
    # for i, b in enumerate(valid_dataloader):
    #     batch = Batch(b[0], b[1], pad_idx)
    #     # print(batch)
    #     break
    check_outputs(valid_dataloader,
                  model,
                  vocab_src,
                  vocab_src)


if __name__ == "__main__":
    main()