import re
import os
import random
import pandas as pd
from pathlib import Path


def strQ2B(ustring):
    """Full width -> half width"""
    # 全角转为半角
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace('-', '') # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s) # Q2B
        s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # keep punctuation
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    # 
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # remove by ratio of length
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


def main():
    data_dir = './dataset/rawdata'
    dataset_name = 'ted2020'

    prefix = Path(data_dir).absolute() / dataset_name

    src_lang = 'en'
    tgt_lang = 'zh'

    valid_ratio = 0.01 # 3000~4000 would suffice
    train_ratio = 1 - valid_ratio

    data_prefix = f'{prefix}/train_dev.raw'
    test_prefix = f'{prefix}/test.raw'

    # clean dataset
    clean_corpus(data_prefix, src_lang, tgt_lang)
    clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

    # split dataset
    if (prefix/f'train.clean.{src_lang}').exists() \
    and (prefix/f'train.clean.{tgt_lang}').exists() \
    and (prefix/f'valid.clean.{src_lang}').exists() \
    and (prefix/f'valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
    else:
        line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
        labels = list(range(line_num))
        random.shuffle(labels)
        # for lang in [src_lang, tgt_lang]:
        #     train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
        #     valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
        #     count = 0
        #     for line in open(f'{data_prefix}.clean.{lang}', 'r'):
        #         if labels[count] / line_num < train_ratio:
        #             train_f.write(line)
        #         else:
        #             valid_f.write(line)
        #         count += 1
        #     train_f.close()
        #     valid_f.close()

        for lang in [src_lang, tgt_lang]:
            train_sentence = []
            valid_sentence = []
            count = 0
            for line in open(f'{data_prefix}.clean.{lang}', 'r'):
                if labels[count] / line_num < train_ratio:
                    train_sentence.append({"sentence": line.strip()})
                else:
                    valid_sentence.append({"sentence": line.strip()})
                count += 1
            train_df = pd.DataFrame(train_sentence, columns=["sentence"])
            valid_df = pd.DataFrame(valid_sentence, columns=["sentence"])
            train_path = os.path.join(data_dir, dataset_name, f'train.clean.{lang}.csv')
            valid_path = os.path.join(data_dir, dataset_name, f'valid.clean.{lang}.csv')
            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)


if __name__ == "__main__":
    main()