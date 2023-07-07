#!/usr/bin/env python3

"""
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
"""


import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


class MultitaskDataset(Dataset):
    def __init__(self, sst_dataset, para_dataset, sts_dataset):
        self.sst_dataset = sst_dataset
        self.para_dataset = para_dataset
        self.sts_dataset = sts_dataset
        self.len = max(len(sst_dataset), len(para_dataset), len(sts_dataset))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (
            self.sst_dataset[idx % len(self.sst_dataset)],
            self.para_dataset[idx % len(self.para_dataset)],
            self.sts_dataset[idx % len(self.sts_dataset)],
        )

    def collate_fn(self, all_data):
        sst_data = [x[0] for x in all_data]
        para_data = [x[1] for x in all_data]
        sts_data = [x[2] for x in all_data]
        sst_batched_data = self.sst_dataset.collate_fn(sst_data)
        para_batched_data = self.para_dataset.collate_fn(para_data)
        sts_batched_data = self.sts_dataset.collate_fn(sts_data)

        return sst_batched_data, para_batched_data, sts_batched_data


class PretrainDataset(Dataset):
    def __init__(self, sst_data, para_data, sts_data):
        self.sst_data = sst_data
        self.para_data = para_data
        self.sts_data = sts_data
        self.len = len(sst_data) + len(para_data) + len(sts_data)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < len(self.sst_data):
            return self.sst_data[idx][0], self.sst_data[idx][2]
        elif idx < len(self.sst_data) + len(self.para_data):
            row = self.para_data[idx - len(self.sst_data)]
            return row[0], row[3]
        else:
            row = self.sts_data[idx - len(self.sst_data) - len(self.para_data)]
            return row[0], row[3]

    def pad_data(self, all_data):
        sents = [x[0] for x in all_data]
        sent_ids = [x[1] for x in all_data]
        encoding = self.tokenizer(
            sents, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        token_type_ids = torch.LongTensor(encoding["token_type_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        cls = token_ids == self.tokenizer.cls_token_id
        sep = token_ids == self.tokenizer.sep_token_id
        special  = cls | sep
        mask = torch.rand(token_ids.shape) < 0.15
        mask = mask & (attention_mask == 1) & ~special
        mask_mask = (torch.rand(token_ids.shape) < 0.8) & mask
        mask_rand = (torch.rand(token_ids.shape) < 0.5) & mask & ~mask_mask

        lablels = token_ids.detach().clone()
        lablels[~mask] = -100

        token_ids[mask_mask] = self.tokenizer.mask_token_id
        token_ids[mask_rand] = torch.randint(
            0, self.tokenizer.vocab_size, token_ids[mask_rand].shape
        )

        return token_ids, token_type_ids, attention_mask, lablels, sents, sent_ids

    def collate_fn(self, all_data):
        return self.pad_data(all_data)


class SentenceClassificationDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]
        encoding = self.tokenizer(
            sents, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


# class SentencePairDataset2(Dataset):
#     def __init__(self, dataset, args, isRegression=False):
#         self.dataset = dataset
#         self.p = args
#         self.isRegression = isRegression
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

#     def pad_data(self, data):
#         sent1 = [x[0] for x in data]
#         sent2 = [x[1] for x in data]
#         labels = [x[2] for x in data]
#         sent_ids = [x[3] for x in data]

#         encoding1 = self.tokenizer(
#             sent1, return_tensors="pt", padding=True, truncation=True
#         )
#         encoding2 = self.tokenizer(
#             sent2, return_tensors="pt", padding=True, truncation=True
#         )

#         token_ids = torch.LongTensor(encoding1["input_ids"])
#         attention_mask = torch.LongTensor(encoding1["attention_mask"])
#         token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

#         token_ids2 = torch.LongTensor(encoding2["input_ids"])
#         attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
#         token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
#         if self.isRegression:
#             labels = torch.FloatTensor(labels)
#         else:
#             labels = torch.LongTensor(labels)

#         return (
#             token_ids,
#             token_type_ids,
#             attention_mask,
#             token_ids2,
#             token_type_ids2,
#             attention_mask2,
#             labels,
#             sent_ids,
#         )

#     def collate_fn(self, all_data):
#         (
#             token_ids,
#             token_type_ids,
#             attention_mask,
#             token_ids2,
#             token_type_ids2,
#             attention_mask2,
#             labels,
#             sent_ids,
#         ) = self.pad_data(all_data)

#         batched_data = {
#             "token_ids_1": token_ids,
#             "token_type_ids_1": token_type_ids,
#             "attention_mask_1": attention_mask,
#             "token_ids_2": token_ids2,
#             "token_type_ids_2": token_type_ids2,
#             "attention_mask_2": attention_mask2,
#             "labels": labels,
#             "sent_ids": sent_ids,
#         }

#         return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(
            sent1, sent2, return_tensors="pt", padding=True, truncation=True
        )
        encoding2 = self.tokenizer(
            sent2, sent1, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])
        token_ids_r = torch.LongTensor(encoding2["input_ids"])
        token_type_ids_r = torch.LongTensor(encoding2["token_type_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])

        if self.isRegression:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids_r,
            token_type_ids_r,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids_r,
            token_type_ids_r,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "token_ids_r": token_ids_r,
            "token_type_ids_r": token_type_ids_r,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(
            sent1, sent2, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        return (token_ids, token_type_ids, attention_mask, sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask, sent_ids) = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "sent_ids": sent_ids,
        }

        return batched_data


# class SentencePairTestDataset2(Dataset):
#     def __init__(self, dataset, args):
#         self.dataset = dataset
#         self.p = args
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

#     def pad_data(self, data):
#         sent1 = [x[0] for x in data]
#         sent2 = [x[1] for x in data]
#         sent_ids = [x[2] for x in data]

#         encoding1 = self.tokenizer(
#             sent1, return_tensors="pt", padding=True, truncation=True
#         )
#         encoding2 = self.tokenizer(
#             sent2, return_tensors="pt", padding=True, truncation=True
#         )

#         token_ids = torch.LongTensor(encoding1["input_ids"])
#         attention_mask = torch.LongTensor(encoding1["attention_mask"])
#         token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

#         token_ids2 = torch.LongTensor(encoding2["input_ids"])
#         attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
#         token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])

#         return (
#             token_ids,
#             token_type_ids,
#             attention_mask,
#             token_ids2,
#             token_type_ids2,
#             attention_mask2,
#             sent_ids,
#         )

#     def collate_fn(self, all_data):
#         (
#             token_ids,
#             token_type_ids,
#             attention_mask,
#             token_ids2,
#             token_type_ids2,
#             attention_mask2,
#             sent_ids,
#         ) = self.pad_data(all_data)

#         batched_data = {
#             "token_ids_1": token_ids,
#             "token_type_ids_1": token_type_ids,
#             "attention_mask_1": attention_mask,
#             "token_ids_2": token_ids2,
#             "token_type_ids_2": token_type_ids2,
#             "attention_mask_2": attention_mask2,
#             "sent_ids": sent_ids,
#         }

#         return batched_data


def load_multitask_test_data():
    paraphrase_filename = f"data/quora-test.csv"
    sentiment_filename = f"data/ids-sst-test.txt"
    similarity_filename = f"data/sts-test.csv"

    sentiment_data = []

    with open(sentiment_filename, "r") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            sent = record["sentence"].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, "r") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            # if record['split'] != split:
            #    continue
            paraphrase_data.append(
                (
                    preprocess_string(record["sentence1"]),
                    preprocess_string(record["sentence2"]),
                )
            )

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, "r") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            similarity_data.append(
                (
                    preprocess_string(record["sentence1"]),
                    preprocess_string(record["sentence2"]),
                )
            )

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data


def load_multitask_data(
    sentiment_filename, paraphrase_filename, similarity_filename, split="train"
):
    sentiment_data = []
    num_labels = set()
    if split == "test":
        with open(sentiment_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                sentiment_data.append((sent, sent_id))
    else:
        with open(sentiment_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                label = int(record["sentiment"].strip())
                num_labels.add(label)
                sentiment_data.append((sent, label, sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == "test":
        with open(paraphrase_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                paraphrase_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(paraphrase_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    paraphrase_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == "test":
        with open(similarity_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                similarity_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(similarity_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                similarity_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        float(record["similarity"]),
                        sent_id,
                    )
                )

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, len(num_labels), paraphrase_data, similarity_data
