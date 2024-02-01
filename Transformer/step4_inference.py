import torch
import torch.utils.data as data
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from step3_translation_zh2en import Batch, make_model, subsequent_mask, TransalationDataset, collate_batch



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
    tokenizer,
    n_examples=15,
    pad_idx=2,
    eos_string="<eos>",
):  
    iter_data = iter(valid_dataloader)
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter_data)
        rb = Batch(b, pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_ids = rb.src[0]
        src_ids = src_ids[src_ids != pad_idx]
        src_tokens = tokenizer.decode(src_ids)

        tgt_ids = rb.tgt[0]
        tgt_ids = tgt_ids[tgt_ids != pad_idx]
        tgt_tokens = tokenizer.decode(tgt_ids)

        print(
            "Source Text (Input)        : "
            + "".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + "".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]

        model_out = model_out[model_out != pad_idx]
        model_txt = (
            "".join(
                [tokenizer.decode(model_out)]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def main():

    new_tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=new_tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    valid_src_path = "dataset/rawdata/ted2020/valid.clean.en.csv"
    valid_tgt_path = "dataset/rawdata/ted2020/valid.clean.zh.csv"


    valid_src_path = "dataset/rawdata/ted2020/train.clean.en.csv"
    valid_tgt_path = "dataset/rawdata/ted2020/train.clean.zh.csv"

    batch_size = 1
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id

    print("Preparing Data ...")
    valid_dataset = TransalationDataset(
        valid_src_path,
        valid_tgt_path,
        tokenizer
    )
    def collate_fn(batch):
        return collate_batch(
            batch,
            torch.device("cpu"),
        )

    valid_dataloader = data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Loading model ...")


    model = make_model(
        vocab_size,
        vocab_size,
        N=6)
    
    model.load_state_dict(
        torch.load("checkpoint_en2zh/en2zh_model_4.pth", map_location=torch.device("cpu"))
    )
    # print(model)
    # for i, b in enumerate(valid_dataloader):
    #     batch = Batch(b[0], b[1], pad_idx)
    #     # print(batch)
    #     break
    check_outputs(valid_dataloader,
                  model,
                  tokenizer,)


if __name__ == "__main__":
    main()