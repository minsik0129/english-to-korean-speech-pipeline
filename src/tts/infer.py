import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from g2pkk import G2p
from jamo import h2j

# vendored 코드(model/ 폴더)를 파이썬이 찾을 수 있도록 경로에 추가
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
sys.path.insert(0, MODEL_DIR)

import hparams as hp
import utils
from fastspeech2 import FastSpeech2
from text import text_to_sequence

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
STATS_DIR = os.path.join(MODEL_DIR, "stats")


def load_config(config_path: str = "configs/tts_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fastspeech2(checkpoint_step: int, device: str):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_step}.pth.tar")
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    model.eval()
    return model.to(device)


def load_vocoder():
    vocoder_path = os.path.join(CHECKPOINT_DIR, "vocgan_kss_pretrained_model_epoch_4500.pt")
    return utils.get_vocgan(ckpt_path=vocoder_path)


def text_to_tensor(text: str, device: str):
    g2p = G2p()
    phone = g2p(text)
    phone = h2j(phone)
    phone = list(filter(lambda p: p != " ", phone))
    phone = "{" + "}{".join(phone) + "}"
    phone = re.sub(r"\{[^\w\s]?\}", "{sil}", phone)
    phone = phone.replace("}{", " ")
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)


def synthesize(model, vocoder, text_tensor, device: str, output_path: str):
    mean_mel, std_mel = torch.tensor(
        np.load(os.path.join(STATS_DIR, "mel_stat.npy")), dtype=torch.float
    ).to(device)
    f0_stat = torch.tensor(
        np.load(os.path.join(STATS_DIR, "f0_stat.npy")), dtype=torch.float
    ).to(device)
    energy_stat = torch.tensor(
        np.load(os.path.join(STATS_DIR, "energy_stat.npy")), dtype=torch.float
    ).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    src_len = torch.from_numpy(np.array([text_tensor.shape[1]])).to(device)

    dur_pitch_energy_aug = [1.0, 1.0, 1.0]
    _, mel_postnet, *_ = model(
        text_tensor, src_len,
        dur_pitch_energy_aug=dur_pitch_energy_aug,
        f0_stat=f0_stat, energy_stat=energy_stat,
    )

    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel_postnet_torch = utils.de_norm(
        mel_postnet_torch.transpose(1, 2), mean_mel, std_mel
    ).transpose(1, 2)

    utils.vocgan_infer(mel_postnet_torch, vocoder, path=output_path)


if __name__ == "__main__":
    config = load_config()
    device = config["device"]

    model = load_fastspeech2(config["checkpoint_step"], device)
    vocoder = load_vocoder()

    text = "안녕하세요, 테스트입니다."
    text_tensor = text_to_tensor(text, device)

    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tts_test.wav")

    synthesize(model, vocoder, text_tensor, device, output_path)
    print(f"저장 완료: {output_path}")