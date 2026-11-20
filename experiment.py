"""Compare a TF-IDF baseline with a tiny CPU Transformer on synthetic tickets."""
import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
import torch
from torch import nn

# Keywords describe the same intent across splits, while templates AND entities
# are held out. Labels are never injected into the input.
TOPICS = {
    "account": ("reset my password", "sign in", "recover my login",
                "unlock my account", "replace the password", "use my login again"),
    "billing": ("correct the invoice", "explain an extra charge", "download a receipt",
                "update my billing address", "question a charge", "find the last invoice"),
    "delivery": ("track my shipment", "find the delivery date", "locate my parcel",
                 "change the shipping address", "check my shipment status", "trace my parcel"),
    "refund": ("refund the payment", "return an item", "request my money back",
               "check my refund", "get a refund", "see when my money comes back"),
    "technical": ("fix the app crash", "stop an error message", "report a broken screen",
                  "solve an app freeze", "investigate an app error", "diagnose a crash"),
}
TRAIN = (
    "Hi, can you help me {action} for {entity}?",
    "{entity}: I need to {action}.",
    "I want to {action}. This is about {entity}.",
    "Could the support team help me {action} on {entity}?",
    "Please tell me how to {action} for {entity}.",
)
TEST = (
    "Regarding {entity}, my main request is to {action}.",
    "What are the steps to {action}? Reference {entity}.",
    "Following up on {entity}. Is it possible to {action}?",
)
TRAIN_ENTITIES = ("fennel", "maple", "birch", "clover", "sage", "violet")
TEST_ENTITIES = ("alder", "orchid", "elm", "cedar")
LABELS = list(TOPICS)
MAX_LEN = 32
TOKEN = re.compile(r"[a-z0-9]+")


def dataset(train):
    templates = TRAIN if train else TEST
    entities = TRAIN_ENTITIES if train else TEST_ENTITIES
    texts, labels = [], []
    for label, actions in enumerate(TOPICS.values()):
        for template, entity, action in itertools.product(
                templates, entities, actions[:4] if train else actions[4:]):
            texts.append(template.format(entity=entity, action=action))
            labels.append(label)
    return texts, np.asarray(labels, dtype=np.int64)


def vocabulary(train_texts):
    counts = Counter(token for text in train_texts
                     for token in TOKEN.findall(text.lower()))
    return {token: index for index, token in enumerate(sorted(counts), start=2)}


def encode(texts, vocab):
    batch = torch.zeros((len(texts), MAX_LEN), dtype=torch.long)
    for row, text in enumerate(texts):
        values = [vocab.get(token, 1) for token in TOKEN.findall(text.lower())[:MAX_LEN]]
        batch[row, :len(values)] = torch.tensor(values, dtype=torch.long)
    return batch


class TinyTransformer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.tokens = nn.Embedding(size, 32, padding_idx=0)
        self.positions = nn.Embedding(MAX_LEN, 32)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, len(LABELS))

    def forward(self, ids):
        valid = ids != 0
        embedded = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        encoded = self.encoder(embedded, src_key_padding_mask=~valid)
        pooled = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)


def metrics(truth, prediction):
    return {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, labels=range(len(LABELS)),
                             average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=range(len(LABELS))).tolist(),
    }


def evaluate(epochs, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train, train_y = dataset(True)
    test, test_y = dataset(False)
    if set(train).intersection(test):
        raise RuntimeError("Train/test overlap")
    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                             LogisticRegression(solver="newton-cg", max_iter=500, random_state=seed))
    baseline.fit(train, train_y)
    vocab = vocabulary(train)
    train_tensor, test_tensor = encode(train, vocab), encode(test, vocab)
    transformer = TinyTransformer(len(vocab) + 2)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=0.003)
    loss = nn.CrossEntropyLoss()
    train_targets = torch.from_numpy(train_y)
    transformer.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(train)).split(64):
            optimizer.zero_grad()
            loss(transformer(train_tensor[idx]), train_targets[idx]).backward()
            optimizer.step()
    transformer.eval()
    with torch.no_grad():
        transformer_pred = transformer(test_tensor).argmax(1).numpy()
    return {
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "pytorch": torch.__version__},
        "design": "disjoint templates, entity names, and action phrases; shared task vocabulary",
        "seed": seed,
        "epochs": epochs,
        "labels": LABELS,
        "train_count": len(train),
        "test_count": len(test),
        "transformer_parameters": sum(p.numel() for p in transformer.parameters()),
        "tfidf_logistic_regression": metrics(test_y, baseline.predict(test)),
        "tiny_transformer": metrics(test_y, transformer_pred),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    report = json.dumps(evaluate(args.epochs, args.seed), indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
