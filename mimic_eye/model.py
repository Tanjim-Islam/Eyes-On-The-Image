from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from . import LABELS, PAPER


def load_weights(path):
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def transformer(name, config, initialize, **kwargs):
    from transformers import AutoConfig, AutoModel

    if initialize:
        return AutoModel.from_pretrained(name, **kwargs)
    fields = {key: val for key, val in config.items() if key != "model_type"}
    architecture = AutoConfig.for_model(config["model_type"], **fields)
    return AutoModel.from_config(architecture, **kwargs)


def image_encoder(spec, initialize, weights):
    kind = spec["image_backbone"]
    if kind == "biovil-t":
        from health_multimodal.image.model.model import ImageModel
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder

        encoder = get_biovil_t_image_encoder() if initialize else ImageModel(
            img_encoder_type="resnet50_multi_image", joint_feature_size=128,
        )
        return encoder, encoder.feature_size
    if kind == "cxr-clip":
        from torchvision.models import resnet50

        encoder = resnet50(weights=None)
        encoder.fc = nn.Identity()
        if initialize:
            state = load_weights(weights)
            state = state.get("model", state)
            prefix = "image_encoder.resnet."
            encoder.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)})
        return encoder, 2048
    kwargs = {"attn_implementation": "eager"} if spec["enhanced"] else {}
    encoder = transformer(spec["vision_model"], spec.get("vision_config"), initialize, **kwargs)
    spec["vision_config"] = encoder.config.to_dict()
    if weights:
        state = load_weights(weights)
        state = state.get("model", state)
        keys = encoder.state_dict().keys()
        transferred = {}
        for key, value in state.items():
            key = key.removeprefix("vit.").removeprefix("image_encoder.")
            if key in keys:
                transferred[key] = value
        encoder.load_state_dict(transferred)
    return encoder, encoder.config.hidden_size


def box_encoder(kind, initialize):
    from torchvision import models

    if kind == "cnn":
        layers = []
        for in_channels, out_channels in zip((1, 32, 64, 128), (32, 64, 128, 256)):
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.GELU(), nn.MaxPool2d(2),
            ))
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(1)), 256
    if kind == "convnext-t":
        network = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if initialize else None)
        return nn.Sequential(network.features, network.avgpool, *list(network.classifier)[:2]), 768
    if kind == "resnet-50":
        network = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if initialize else None)
        network.fc = nn.Identity()
        return network, 2048
    if kind == "efficientnetv2-s":
        network = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if initialize else None)
        network.classifier = nn.Identity()
        return network, 1280
    raise ValueError(f"Unknown box backbone {kind}")


def projection(in_features, width):
    return nn.Sequential(
        nn.Linear(in_features, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(width, width), nn.LayerNorm(width), nn.Dropout(0.15),
    )


class Classifier(nn.Module):
    def __init__(self, spec, initialize=False, encoder_weights=None):
        super().__init__()
        self.spec = dict(spec)
        self.image_encoder, image_width = image_encoder(self.spec, initialize, encoder_weights)
        width = self.spec["width"]
        self.image_projection = projection(image_width, width)
        count = 1
        if spec["box"]:
            self.box_encoder, box_width = box_encoder(spec["box_backbone"], initialize)
            self.box_projection = projection(box_width, width)
            count += 1
        if spec.get("freeze_image_box"):
            if initialize:
                prefix = "box_encoder."
                state = load_weights(encoder_weights).get("model", {})
                box_weights = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
                if not box_weights:
                    raise ValueError("The text encoder comparison requires a trained image-box checkpoint")
                self.box_encoder.load_state_dict(box_weights)
            self.image_encoder.requires_grad_(False)
            self.box_encoder.requires_grad_(False)
        if spec["text"]:
            self.text_encoder = transformer(spec["text_model"], spec.get("text_config"), initialize)
            self.spec["text_config"] = self.text_encoder.config.to_dict()
            self.text_encoder.requires_grad_(False)
            self.text_projection = projection(self.text_encoder.config.hidden_size, width)
            count += 1
        if spec["gaze"]:
            hidden = spec["gaze_hidden"]
            self.fixation_embedding = nn.Linear(4, 64)
            self.fixation_encoder = nn.GRU(64, hidden, 2, batch_first=True, bidirectional=True, dropout=0.15)
            self.fixation_projection = projection(4 * hidden, width)
            count += 1
        if spec["enhanced"]:
            self.gaze_token = nn.Parameter(torch.zeros(1, 1, image_width))
            self.gaze_token_projection = nn.Linear(width, image_width)
            self.gaze_attention = nn.MultiheadAttention(image_width, 8, dropout=0.15, batch_first=True)
            self.gaze_mix_gate = nn.Linear(width, 1)
            self.gaze_token_gate = nn.Linear(width, 1)
            self.gaze_decoder = nn.Linear(width, 14 * 14)
        if count > 1:
            self.fusion_attention = nn.MultiheadAttention(width, 8, dropout=0.15, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(count * width, 2 * width), nn.LayerNorm(2 * width), nn.GELU(),
                nn.Dropout(0.15), nn.Linear(2 * width, width), nn.LayerNorm(width),
            )
        self.head = nn.Linear(width, len(LABELS))
        if spec["enhanced"]:
            self.condition_heads = nn.ModuleList(nn.Linear(width, 1) for _ in LABELS)

    def encode_image(self, image):
        kind = self.spec["image_backbone"]
        if kind == "biovil-t":
            return self.image_encoder(image).img_embedding, None
        if kind == "cxr-clip":
            return self.image_encoder(image), None
        output = self.image_encoder(pixel_values=image, output_attentions=self.spec["enhanced"])
        tokens = output.last_hidden_state
        embedding = self.image_projection(tokens[:, 0])
        if not self.spec["enhanced"]:
            return embedding, None
        with torch.no_grad():
            identity = torch.eye(tokens.shape[1], device=image.device)
            rollout = identity.unsqueeze(0).expand(len(tokens), -1, -1)
            for attention in output.attentions:
                attention = attention.float().mean(dim=1) + identity
                rollout = (attention / attention.sum(dim=-1, keepdim=True)) @ rollout
            weights = rollout[:, 0, 1:]
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
            confidence = 1 - entropy / weights.new_tensor(weights.shape[1]).log()
        pooled = self.image_projection((tokens[:, 1:] * weights.unsqueeze(-1)).sum(dim=1))
        gate = (self.gaze_mix_gate(pooled).sigmoid() * confidence).clamp(0.0, 0.8)
        return embedding.lerp(pooled, gate.to(embedding.dtype)), tokens[:, 1:]

    def forward(self, image, box=None, gaze=None, gaze_mask=None, text=None, text_present=None):
        embedding, patches = self.encode_image(image)
        if self.spec["image_backbone"] != "vit-chexpert":
            embedding = self.image_projection(embedding)
        zero = torch.zeros_like(embedding)
        features = [embedding]
        present = [torch.ones(len(image), dtype=torch.bool, device=image.device)]

        if self.spec["box"]:
            valid = box.flatten(1).any(dim=1) if box is not None else ~present[0]
            if box is None:
                features.append(zero)
            else:
                pixels = box if self.spec["box_backbone"] == "cnn" else box.expand(-1, 3, -1, -1)
                features.append(self.box_projection(self.box_encoder(pixels)) * valid[:, None])
            present.append(valid)
        if self.spec["text"]:
            if text is None:
                features.append(zero)
                present.append(~present[0])
            else:
                self.text_encoder.eval()
                with torch.no_grad():
                    encoded = self.text_encoder(**text).last_hidden_state[:, 0]
                features.append(self.text_projection(encoded) * text_present[:, None])
                present.append(text_present)
        if self.spec["gaze"]:
            valid = gaze_mask.any(dim=1) if gaze_mask is not None else ~present[0]
            fixation = zero
            if gaze is not None:
                packed = pack_padded_sequence(
                    self.fixation_embedding(gaze), gaze_mask.sum(dim=1).clamp_min(1).cpu(),
                    batch_first=True, enforce_sorted=False,
                )
                _, hidden = self.fixation_encoder(packed)
                fixation = self.fixation_projection(hidden.transpose(0, 1).flatten(1)) * valid[:, None]
            features.append(fixation)
            present.append(valid)
            if self.spec["enhanced"]:
                token = self.gaze_token + self.gaze_token_projection(fixation).unsqueeze(1)
                attended, _ = self.gaze_attention(token, patches, patches, need_weights=False)
                gate = self.gaze_token_gate(fixation).sigmoid().clamp(max=0.4)
                features[0] = embedding + gate * self.image_projection(attended[:, 0]) * valid[:, None]

        if len(features) == 1:
            fused = features[0]
        else:
            tokens = torch.stack(features, dim=1)
            mask = torch.stack(present, dim=1)
            attended, _ = self.fusion_attention(tokens, tokens, tokens, key_padding_mask=~mask, need_weights=False)
            fused = self.fusion(((attended + tokens) * mask.unsqueeze(-1)).flatten(1))
        logits = self.head(fused)
        saliency = None
        if self.spec["enhanced"]:
            logits = 0.5 * (logits + torch.cat([head(fused) for head in self.condition_heads], dim=1))
            saliency = self.gaze_decoder(fused).reshape(-1, 14, 14)
        return logits, saliency, features, present


def classification_loss(logits, labels, pos_weight):
    log_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    probability = logits.sigmoid()
    wrong = (1 - probability) * labels + probability * (1 - labels)
    return (log_loss * wrong.square()).mean()


def contrastive_loss(image, modality, valid):
    if valid.sum() < 2:
        return image.new_zeros(())
    left = F.normalize(image[valid].float())
    right = F.normalize(modality[valid].float())
    scores = left @ right.T / PAPER["contrastive_temperature"]
    targets = torch.arange(len(scores), device=scores.device)
    return (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2
