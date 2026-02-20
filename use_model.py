import torch
import json
from train_pylaia import CRNN

from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
root: str = "C:/Users/dhlabadmin/Desktop/m-test/pylaia-iam/"

with open(root+"pylaia_church_slavonic_model/model_config.json", "r") as f:
    cfg = json.load(f)

# num_classes = len(open(root+"my_model/symbols.txt", encoding="utf-8-sig").read().splitlines()) + 1
symbols = [line.rstrip("\n\r")
           for line in open(root+"pylaia_church_slavonic_model/symbols.txt", encoding="utf-8-sig")
           if line.rstrip("\n\r")]

num_classes = len(symbols)

model = CRNN(
    img_height=cfg["img_height"],
    num_channels=1,
    num_classes=num_classes,
    cnn_filters=cfg["cnn_filters"],
    cnn_poolsize=cfg["cnn_poolsize"],
    rnn_hidden=cfg["rnn_hidden"],
    rnn_layers=cfg["rnn_layers"],
    dropout=cfg["dropout"]
)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# minimal test version of a CRNN for pylaia
# class CRNN(nn.Module):
#     """
#     Minimal PyLaia-style CRNN for inference only.

#     Architecture:
#     CNN -> collapse height -> BiLSTM -> Linear -> log_softmax
#     """

#     def __init__(
#         self,
#         img_height,
#         num_channels,
#         num_classes,
#         cnn_filters,
#         cnn_poolsize,
#         rnn_hidden,
#         rnn_layers,
#         dropout,
#     ):
#         super().__init__()

#         assert len(cnn_filters) == len(cnn_poolsize)

#         # ---- CNN ----
#         layers = []
#         in_channels = num_channels

#         for out_channels, pool in zip(cnn_filters, cnn_poolsize):
#             layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU(inplace=True))

#             if pool > 0:
#                 layers.append(nn.MaxPool2d(pool))

#             in_channels = out_channels

#         self.cnn = nn.Sequential(*layers)

#         # Compute feature size after CNN height collapse
#         self.feature_height = img_height
#         for pool in cnn_poolsize:
#             if pool > 0:
#                 self.feature_height //= pool

#         rnn_input_size = cnn_filters[-1] * self.feature_height

#         # ---- BiLSTM ----
#         self.rnn = nn.LSTM(
#             input_size=rnn_input_size,
#             hidden_size=rnn_hidden,
#             num_layers=rnn_layers,
#             dropout=dropout if rnn_layers > 1 else 0,
#             bidirectional=True,
#         )

#         # ---- Classifier ----
#         self.fc = nn.Linear(rnn_hidden * 2, num_classes)

#     def forward(self, x):
#         """
#         x: [B, C, H, W]
#         returns: [T, B, C] (CTC format)
#         """

#         features = self.cnn(x)
#         b, c, h, w = features.size()

#         # Collapse height dimension
#         features = features.permute(3, 0, 1, 2)  # [W, B, C, H]
#         features = features.contiguous().view(w, b, c * h)

#         rnn_out, _ = self.rnn(features)
#         logits = self.fc(rnn_out)

#         return F.log_softmax(logits, dim=2)

# model = CRNN(
#     img_height=cfg["img_height"],
#     num_channels=1,
#     num_classes=num_classes,
#     cnn_filters=cfg["cnn_filters"],
#     cnn_poolsize=cfg["cnn_poolsize"],
#     rnn_hidden=cfg["rnn_hidden"],
#     rnn_layers=cfg["rnn_layers"],
#     dropout=cfg["dropout"]
# )

# state = torch.load(root+"my_model/model.pt", map_location="cpu")
# checkpoint = torch.load(root+"my_model/model.pt", map_location=device)
checkpoint = torch.load(root+"pylaia_church_slavonic_model/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# def load_idx2char(symbols_path):
#     with open(symbols_path, encoding="utf-8-sig") as f:
#         symbols = [line.rstrip("\n\r") for line in f if line.rstrip("\n\r")]

#     idx2char = {i + 1: ch for i, ch in enumerate(symbols)}
#     idx2char[0] = ""
#     return idx2char


# idx2char = load_idx2char(root+"my_model/symbols.txt")
idx2char = checkpoint["idx2char"]
# print(list(idx2char.items())[:20])


def decode_predictions(log_probs, idx2char):
    preds = log_probs.argmax(dim=2)
    results = []

    for b in range(preds.shape[1]):
        prev = None
        text = []

        for t in range(preds.shape[0]):
            idx = preds[t, b].item()
            if idx != prev and idx != 0:
                text.append(idx2char[idx])
            prev = idx

        results.append("".join(text))

    return results


def load_image(path, img_height):
    img = Image.open(path).convert("L")
    w, h = img.size
    new_w = int(w * img_height / h)
    img = img.resize((new_w, img_height), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img = transform(img)
    return img.unsqueeze(0)


image = load_image("01.png", cfg["img_height"]).to(device)
# print(image.min().item(), image.max().item(), image.mean().item())
# print(image.shape)
with torch.no_grad():
    log_probs = model(image)
    text = decode_predictions(log_probs, idx2char)

print(text)
