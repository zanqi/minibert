import pprint
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW, PCGrad
from tqdm import tqdm

from datasets import (
    MultitaskDataset,
    PretrainDataset,
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
    load_multitask_test_data,
)

from evaluation import model_eval_multitask, model_eval_sst, eval_test_model_multitask


TQDM_DISABLE = True


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        ### TODO
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.linear_sentiment = torch.nn.Linear(config.hidden_size, config.num_labels)
        # self.linear_paraphrase = torch.nn.Linear(2 * config.hidden_size, 1)
        # self.linear_similarity = torch.nn.Linear(2 * config.hidden_size, 1)
        self.linear_paraphrase = torch.nn.Linear(config.hidden_size, 1)
        self.linear_similarity = torch.nn.Linear(config.hidden_size, 1)

        # def forward2(self, input_ids, attention_mask):
        "Takes a batch of sentences and produces embeddings for them."
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # return self.bert(input_ids, attention_mask)["pooler_output"]

    def forward(self, input_ids, input_type, attention_mask):
        return self.bert(input_ids, input_type, attention_mask)["pooler_output"]

    def predict_masked_tokens(self, input_ids, input_type, attention_mask, ys=None):
        token_logits = self.bert(input_ids, input_type, attention_mask)["token_logits"]
        if ys is not None:
            return token_logits, F.cross_entropy(
                token_logits.view(-1, token_logits.size(-1)),
                ys.view(-1),
                ignore_index=-100,
            )
        return token_logits

    def predict_sentiment(self, input_ids, attention_mask, ys=None):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        ### TODO
        cls_embedding = self(input_ids, torch.zeros_like(input_ids), attention_mask)
        logits = self.linear_sentiment(self.dropout(cls_embedding))
        if ys is not None:
            return logits, F.cross_entropy(logits, ys.view(-1))
        return logits

    # def predict_paraphrase_lin(
    #     self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, ys=None
    # ):
    #     """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
    #     Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
    #     during evaluation, and handled as a logit by the appropriate loss function.
    #     """
    #     ### TODO
    #     # investigate if concatenating the input_ids and attention_mask is the right way
    #     cls_embedding_1 = self(input_ids_1, attention_mask_1)
    #     cls_embedding_2 = self(input_ids_2, attention_mask_2)
    #     cls_embedding = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)
    #     logits = self.linear_paraphrase(self.dropout(cls_embedding))
    #     if ys is not None:
    #         return logits, F.binary_cross_entropy_with_logits(
    #             logits, ys.float().unsqueeze(1)
    #         )
    #     return logits

    # def predict_paraphrase_cos(
    #     self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, ys=None
    # ):
    #     """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
    #     Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
    #     during evaluation, and handled as a logit by the appropriate loss function.
    #     """
    #     ### TODO
    #     # investigate if concatenating the input_ids and attention_mask is the right way
    #     cls_embedding_1 = self(input_ids_1, attention_mask_1)
    #     cls_embedding_2 = self(input_ids_2, attention_mask_2)
    #     logits = F.cosine_similarity(cls_embedding_1, cls_embedding_2)
    #     if ys is not None:
    #         return logits, F.binary_cross_entropy_with_logits(
    #             logits, ys.float().unsqueeze(1)
    #         )
    #     return logits

    # def predict_similarity_lin(
    #     self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, ys=None
    # ):
    #     """Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
    #     Note that your output should be unnormalized (a logit).
    #     """
    #     ### TODO
    #     # investigate if concatenating the input_ids and attention_mask is the right way
    #     cls_embedding_1 = self(input_ids_1, attention_mask_1)
    #     cls_embedding_2 = self(input_ids_2, attention_mask_2)
    #     cls_embedding = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)
    #     logits = F.relu(self.linear_similarity(self.dropout(cls_embedding)))
    #     if ys is not None:
    #         return logits, F.mse_loss(logits, ys.float().unsqueeze(1))
    #     return logits

    # def predict_similarity_cos(
    #     self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, ys=None
    # ):
    #     cls_embedding_1 = self(input_ids_1, attention_mask_1)
    #     cls_embedding_2 = self(input_ids_2, attention_mask_2)
    #     logits = (1 + F.cosine_similarity(cls_embedding_1, cls_embedding_2)) * 5 / 2
    #     if ys is not None:
    #         return logits, F.mse_loss(logits, ys)
    #     return logits

    def predict_paraphrase(
        self, input_ids, input_type, input_ids_r, input_type_r, attention_mask, ys=None
    ):
        cls_embedding = self(input_ids, input_type, attention_mask)
        cls_embedding_r = self(input_ids_r, input_type_r, attention_mask)
        logits = self.linear_paraphrase(self.dropout(cls_embedding + cls_embedding_r))
        if ys is not None:
            return logits, F.binary_cross_entropy_with_logits(
                logits, ys.float().unsqueeze(1)
            )
        return logits

    # def predict_similarity_no_flip(
    #     self, input_ids, input_type, attention_mask, ys=None
    # ):
    #     cls_embedding = self(input_ids, input_type, attention_mask)
    #     logits = F.relu(self.linear_similarity(self.dropout(cls_embedding)))
    #     if ys is not None:
    #         return logits, F.mse_loss(logits, ys.unsqueeze(1))
    #     return logits

    def predict_similarity(
        self,
        input_ids,
        input_type,
        input_ids_r,
        input_type_r,
        attention_mask,
        ys=None,
    ):
        cls_embedding = self(input_ids, input_type, attention_mask)
        cls_embedding_r = self(input_ids_r, input_type_r, attention_mask)
        logits = 5 * F.sigmoid(
            self.linear_similarity(self.dropout(cls_embedding + cls_embedding_r))
        )
        if ys is not None:
            return logits, F.mse_loss(logits, ys.unsqueeze(1))
        return logits


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def get_device(device_name):
    device = torch.device(device_name)
    print(f"using device {device}")
    return device


## Currently only trains on sst dataset
def train_multitask(args):
    device = get_device(args.device)
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split="train"
    )
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split="dev"
    )

    pretrain_dataset = PretrainDataset(sst_train_data, para_train_data, sts_train_data)
    # (
    #     token_ids,
    #     token_type_ids,
    #     attention_mask,
    #     lablels,
    #     sents,
    #     sent_ids,
    # ) = pretrain_dataset.collate_fn([pretrain_dataset[0], pretrain_dataset[1]])

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    multitask_train_dataset = MultitaskDataset(
        sst_train_data, para_train_data, sts_train_data
    )

    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        shuffle=True,
        batch_size=8,
        collate_fn=pretrain_dataset.collate_fn,
    )

    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=multitask_train_dataset.collate_fn,
    )

    sst_train_dataloader = DataLoader(
        sst_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sst_train_data.collate_fn,
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sst_dev_data.collate_fn,
    )

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
    )

    sts_train_dataloader = DataLoader(
        sts_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sts_train_data.collate_fn,
    )
    sts_dev_dataloader = DataLoader(
        sts_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_data.collate_fn,
    )

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "num_labels": num_labels,
        "hidden_size": 768,
        "data_dir": ".",
        "option": args.option,
    }

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    # optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    # pc_grad = PCGrad(optimizer)

    # In task pretraining
    iter = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in tqdm(
            pretrain_dataloader,
            desc=f"pretrain-multi-{epoch}",
            disable=TQDM_DISABLE,
        ):
            if args.pretrain_max_iters and iter >= args.pretrain_max_iters:
                break

            # if iter % args.eval_interval == 0:
            #     (
            #         best_dev_sst_acc,
            #         best_dev_para_acc,
            #         best_dev_sts_corr,
            #     ) = eval_and_save_model(
            #         args,
            #         device,
            #         sst_train_dataloader,
            #         sst_dev_dataloader,
            #         para_train_dataloader,
            #         para_dev_dataloader,
            #         sts_train_dataloader,
            #         sts_dev_dataloader,
            #         config,
            #         model,
            #         lr,
            #         optimizer,
            #         best_dev_sst_acc,
            #         best_dev_para_acc,
            #         best_dev_sts_corr,
            #         iter,
            #     )

            optimizer.zero_grad()

            loss = get_lm_loss(device, model, batch)
            loss.backward()
            optimizer.step()
            # pc_grad.pc_backward([sst_loss, para_loss, sts_loss])
            # pc_grad.step()

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(
                f"epoc {epoch}/{args.epochs} iter {iter}/{len(pretrain_dataloader) * args.epochs}: pretrain loss {loss.item():.4f}, time {dt*1000:.2f}ms"
            )
            iter += 1

    best_dev_sst_acc, best_dev_para_acc, best_dev_sts_corr = 0, 0, -100
    t0 = time.time()

    iter = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(
            multitask_train_dataloader,
            desc=f"train-multi-{epoch}",
            disable=TQDM_DISABLE,
        ):
            if args.max_iters and iter >= args.max_iters:
                break

            if iter % args.eval_interval == 0:
                (
                    best_dev_sst_acc,
                    best_dev_para_acc,
                    best_dev_sts_corr,
                ) = eval_and_save_model(
                    args,
                    device,
                    sst_train_dataloader,
                    sst_dev_dataloader,
                    para_train_dataloader,
                    para_dev_dataloader,
                    sts_train_dataloader,
                    sts_dev_dataloader,
                    config,
                    model,
                    lr,
                    optimizer,
                    best_dev_sst_acc,
                    best_dev_para_acc,
                    best_dev_sts_corr,
                    iter,
                )

            optimizer.zero_grad()
            # pc_grad.zero_grad()
            sst_loss, para_loss, sts_loss = get_multi_batch_loss(device, model, batch)
            loss = sst_loss + para_loss + sts_loss
            loss.backward()
            optimizer.step()
            # pc_grad.pc_backward([sst_loss, para_loss, sts_loss])
            # pc_grad.step()

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print_iter_info(
                args.epochs,
                len(multitask_train_dataloader),
                iter,
                epoch,
                sst_loss,
                para_loss,
                sts_loss,
                dt,
            )
            iter += 1

    # iter = 0
    # for epoch in range(args.sst_epochs):
    #     model.train()
    #     for batch in tqdm(
    #         sst_train_dataloader, desc=f"train-sst-{epoch}", disable=TQDM_DISABLE
    #     ):
    #         if args.sst_iters and iter >= args.sst_iters:
    #             break

    #         if iter % args.eval_interval == 0:
    #             (
    #                 best_dev_sst_acc,
    #                 best_dev_para_acc,
    #                 best_dev_sts_corr,
    #             ) = eval_and_save_model(
    #                 args,
    #                 device,
    #                 sst_train_dataloader,
    #                 sst_dev_dataloader,
    #                 para_train_dataloader,
    #                 para_dev_dataloader,
    #                 sts_train_dataloader,
    #                 sts_dev_dataloader,
    #                 config,
    #                 model,
    #                 lr,
    #                 optimizer,
    #                 best_dev_sst_acc,
    #                 best_dev_para_acc,
    #                 best_dev_sts_corr,
    #                 iter,
    #             )

    #         optimizer.zero_grad()
    #         loss = get_sst_batch_loss(device, model, batch)

    #         loss.backward()
    #         optimizer.step()

    #         t1 = time.time()
    #         dt = t1 - t0
    #         t0 = t1
    #         print_iter_info(
    #             args.sst_epochs, sst_train_dataloader, iter, epoch, loss, dt, "sst"
    #         )
    #         iter += 1

    # iter = 0
    # for epoch in range(args.para_epochs):
    #     model.train()
    #     for batch in tqdm(
    #         para_train_dataloader, desc=f"train-para-{epoch}", disable=TQDM_DISABLE
    #     ):
    #         if args.para_iters and iter >= args.para_iters:
    #             break

    #         if iter % args.eval_interval == 0:
    #             (
    #                 best_dev_sst_acc,
    #                 best_dev_para_acc,
    #                 best_dev_sts_corr,
    #             ) = eval_and_save_model(
    #                 args,
    #                 device,
    #                 sst_train_dataloader,
    #                 sst_dev_dataloader,
    #                 para_train_dataloader,
    #                 para_dev_dataloader,
    #                 sts_train_dataloader,
    #                 sts_dev_dataloader,
    #                 config,
    #                 model,
    #                 lr,
    #                 optimizer,
    #                 best_dev_sst_acc,
    #                 best_dev_para_acc,
    #                 best_dev_sts_corr,
    #                 iter,
    #             )

    #         optimizer.zero_grad()
    #         loss = get_para_batch_loss(device, model, batch)

    #         loss.backward()
    #         optimizer.step()

    #         t1 = time.time()
    #         dt = t1 - t0
    #         t0 = t1
    #         print_iter_info(
    #             args.para_epochs, para_train_dataloader, iter, epoch, loss, dt, "para"
    #         )
    #         iter += 1

    # iter = 0
    # for epoch in range(args.sts_epochs):
    #     model.train()
    #     for batch in tqdm(
    #         sts_train_dataloader, desc=f"train-sts-{epoch}", disable=TQDM_DISABLE
    #     ):
    #         if args.sts_iters and iter >= args.sts_iters:
    #             break
    #         if iter % args.eval_interval == 0:
    #             (
    #                 best_dev_sst_acc,
    #                 best_dev_para_acc,
    #                 best_dev_sts_corr,
    #             ) = eval_and_save_model(
    #                 args,
    #                 device,
    #                 sst_train_dataloader,
    #                 sst_dev_dataloader,
    #                 para_train_dataloader,
    #                 para_dev_dataloader,
    #                 sts_train_dataloader,
    #                 sts_dev_dataloader,
    #                 config,
    #                 model,
    #                 lr,
    #                 optimizer,
    #                 best_dev_sst_acc,
    #                 best_dev_para_acc,
    #                 best_dev_sts_corr,
    #                 iter,
    #             )

    #         optimizer.zero_grad()
    #         loss = get_sts_batch_loss(device, model, batch)

    #         loss.backward()
    #         optimizer.step()

    #         t1 = time.time()
    #         dt = t1 - t0
    #         t0 = t1
    #         print_iter_info(
    #             args.sts_epochs, sts_train_dataloader, iter, epoch, loss, dt, "sts"
    #         )
    #         iter += 1


def print_iter_info(epochs, batches, iter, epoch, sst_loss, para_loss, sts_loss, dt):
    if iter % args.print_interval == 0:
        total = sst_loss.item() + para_loss.item() + sts_loss.item()
        print(
            f"epoc {epoch}/{epochs} iter {iter}/{batches * epochs}: sst loss {sst_loss.item():.4f}, para loss {para_loss.item():.4f}, sts loss {sts_loss.item():.4f},  multi loss {total:.4f}, time {dt*1000:.2f}ms"
        )


def eval_and_save_model(
    args,
    device,
    sst_train_dataloader,
    sst_dev_dataloader,
    para_train_dataloader,
    para_dev_dataloader,
    sts_train_dataloader,
    sts_dev_dataloader,
    config,
    model,
    lr,
    optimizer,
    best_dev_sst_acc,
    best_dev_para_acc,
    best_dev_sts_corr,
    iter,
):
    (
        _,
        _,
        _,
        dev_paraphrase_accuracy,
        dev_sentiment_accuracy,
        dev_sts_corr,
    ) = eval_model(
        args,
        device,
        sst_train_dataloader,
        sst_dev_dataloader,
        para_train_dataloader,
        para_dev_dataloader,
        sts_train_dataloader,
        sts_dev_dataloader,
        model,
        lr,
        iter,
    )
    # if dev_paraphrase_accuracy > best_dev_para_acc:
    if (
        dev_paraphrase_accuracy + dev_sentiment_accuracy + dev_sts_corr
        > best_dev_para_acc + best_dev_sst_acc + best_dev_sts_corr
    ):
        best_dev_para_acc = dev_paraphrase_accuracy
        best_dev_sst_acc = dev_sentiment_accuracy
        best_dev_sts_corr = dev_sts_corr
        save_model(model, optimizer, args, config, args.filepath)
    return best_dev_sst_acc, best_dev_para_acc, best_dev_sts_corr


def eval_model(
    args,
    device,
    sst_train_dataloader,
    sst_dev_dataloader,
    para_train_dataloader,
    para_dev_dataloader,
    sts_train_dataloader,
    sts_dev_dataloader,
    model,
    lr,
    iter,
):
    (
        train_paraphrase_accuracy,
        _,
        _,
        train_sentiment_accuracy,
        _,
        _,
        train_sts_corr,
        _,
        _,
        train_para_loss,
        train_sst_loss,
        train_sts_loss,
    ) = model_eval_multitask(
        sst_train_dataloader,
        para_train_dataloader,
        sts_train_dataloader,
        model,
        device,
        "train",
        args.eval_iters,
    )
    (
        dev_paraphrase_accuracy,
        _,
        _,
        dev_sentiment_accuracy,
        _,
        _,
        dev_sts_corr,
        _,
        _,
        dev_para_loss,
        dev_sst_loss,
        dev_sts_loss,
    ) = model_eval_multitask(
        sst_dev_dataloader,
        para_dev_dataloader,
        sts_dev_dataloader,
        model,
        device,
        "dev",
        args.eval_iters,
    )

    if args.wandb_log:
        wandb.log(
            {
                "train/paraphrase_accuracy": train_paraphrase_accuracy,
                "train/sentiment_accuracy": train_sentiment_accuracy,
                "train/sts_corr": train_sts_corr,
                "train/sst_loss": train_sst_loss,
                "train/paraphrase_loss": train_para_loss,
                "train/sts_loss": train_sts_loss,
                "train/multi_loss": train_sst_loss + train_para_loss + train_sts_loss,
                "dev/paraphrase_accuracy": dev_paraphrase_accuracy,
                "dev/sentiment_accuracy": dev_sentiment_accuracy,
                "dev/sts_corr": dev_sts_corr,
                "dev/sst_loss": dev_sst_loss,
                "dev/paraphrase_loss": dev_para_loss,
                "dev/sts_loss": dev_sts_loss,
                "dev/multi_loss": dev_sst_loss + dev_para_loss + dev_sts_loss,
                "iter": iter,
                "lr": lr,
            }
        )
    return (
        train_paraphrase_accuracy,
        train_sentiment_accuracy,
        train_sts_corr,
        dev_paraphrase_accuracy,
        dev_sentiment_accuracy,
        dev_sts_corr,
    )


def get_para_batch_loss(device, model, batch):
    b_ids, b_type, b_ids_r, b_type_r, b_mask, b_labels = (
        batch["token_ids"],
        batch["token_type_ids"],
        batch["attention_mask"],
        batch["token_ids_r"],
        batch["token_type_ids_r"],
        batch["labels"],
    )

    b_ids = b_ids.to(device)
    b_type = b_type.to(device)
    b_mask = b_mask.to(device)
    b_ids_r = b_ids.to(device)
    b_type_r = b_type.to(device)
    b_labels = b_labels.to(device)

    _, loss = model.predict_paraphrase(
        b_ids, b_type, b_ids_r, b_type_r, b_mask, b_labels
    )
    return loss


def get_sts_batch_loss(device, model, batch):
    b_ids, b_type, b_mask, b_ids_r, b_type_r, b_labels = (
        batch["token_ids"],
        batch["token_type_ids"],
        batch["attention_mask"],
        batch["token_ids_r"],
        batch["token_type_ids_r"],
        batch["labels"],
    )

    b_ids = b_ids.to(device)
    b_type = b_type.to(device)
    b_mask = b_mask.to(device)
    b_ids_r = b_ids.to(device)
    b_type_r = b_type.to(device)
    b_labels = b_labels.to(device)
    _, loss = model.predict_similarity(
        b_ids, b_type, b_ids_r, b_type_r, b_mask, b_labels
    )
    return loss


# def get_sts_batch_loss2(device, model, batch):
#     b_ids_1, b_type_1, b_mask_1, b_ids_2, b_type_2, b_mask_2, b_labels = (
#         batch["token_ids_1"],
#         batch["token_type_ids_1"],
#         batch["attention_mask_1"],
#         batch["token_ids_2"],
#         batch["token_type_ids_2"],
#         batch["attention_mask_2"],
#         batch["labels"],
#     )

#     b_ids_1 = b_ids_1.to(device)
#     b_mask_1 = b_mask_1.to(device)
#     b_ids_2 = b_ids_2.to(device)
#     b_mask_2 = b_mask_2.to(device)
#     b_labels = b_labels.to(device)
#     _, loss = model.predict_similarity_lin(b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels)
#     return loss


def get_multi_batch_loss(device, model, batch):
    return (
        get_sst_batch_loss(device, model, batch[0]),
        get_para_batch_loss(device, model, batch[1]),
        get_sts_batch_loss(device, model, batch[2]),
    )


def get_lm_loss(device, model, batch):
    token_ids, token_type_ids, attention_mask, labels, sents, sent_ids = batch

    token_ids = token_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    _, loss = model.predict_masked_tokens(
        token_ids, token_type_ids, attention_mask, labels
    )
    return loss


def get_sst_batch_loss(device, model, batch):
    b_ids, b_mask, b_labels = (
        batch["token_ids"],
        batch["attention_mask"],
        batch["labels"],
    )
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)
    _, loss = model.predict_sentiment(b_ids, b_mask, b_labels)
    return loss


def test_model(args):
    with torch.no_grad():
        device = get_device(args.device)
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        (
            dev_paraphrase_accuracy,
            _,
            _,
            dev_sentiment_accuracy,
            _,
            _,
            dev_sts_corr,
            _,
            _,
            dev_para_loss,
            dev_sst_loss,
            dev_sts_loss,
        ) = eval_test_model_multitask(args, model, device)
        if args.wandb_log:
            wandb.log(
                {
                    "dev/paraphrase_accuracy": dev_paraphrase_accuracy,
                    "dev/sentiment_accuracy": dev_sentiment_accuracy,
                    "dev/sts_corr": dev_sts_corr,
                    "dev/paraphrase_loss": dev_para_loss,
                    "dev/sst_loss": dev_sst_loss,
                    "dev/sts_loss": dev_sts_loss,
                }
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=8000)
    parser.add_argument("--pretrain_max_iters", type=int, default=8000)
    parser.add_argument("--sst_epochs", type=int, default=5)
    parser.add_argument("--para_epochs", type=int, default=1)
    parser.add_argument("--sts_epochs", type=int, default=5)
    parser.add_argument("--sst_iters", type=int, default=None)
    parser.add_argument("--para_iters", type=int, default=None)
    parser.add_argument("--sts_iters", type=int, default=None)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=("cpu", "cuda", "mps")
    )
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--wandb_log", type=bool, default=True)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="multi-task-bert",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="multi-" + str(time.time()),
        help="wandb run name",
    )

    parser.add_argument(
        "--sst_dev_out", type=str, default="predictions/sst-dev-output.csv"
    )
    parser.add_argument(
        "--sst_test_out", type=str, default="predictions/sst-test-output.csv"
    )

    parser.add_argument(
        "--para_dev_out", type=str, default="predictions/para-dev-output.csv"
    )
    parser.add_argument(
        "--para_test_out", type=str, default="predictions/para-test-output.csv"
    )

    parser.add_argument(
        "--sts_dev_out", type=str, default="predictions/sts-dev-output.csv"
    )
    parser.add_argument(
        "--sts_test_out", type=str, default="predictions/sts-test-output.csv"
    )

    # hyper parameters
    parser.add_argument(
        "--batch_size",
        help="sst: 64, cfimdb: 8 can fit a 12GB GPU",
        type=int,
        default=1,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 2e-5",
        default=1e-3,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.option}-{args.epochs}-{args.lr}-multitask.pt"  # save path
    # Here's a potential point to load different configs and overwrite args
    if args.wandb_log:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"{args.option}-{args.wandb_run_name}",
            config=args,
        )

    seed_everything(args.seed)  # fix the seed for reproducibility
    pprint.pprint(f"args: {args}")
    train_multitask(args)
    test_model(args)
