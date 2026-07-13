import yaml
import whisper


def load_config(config_path: str = "configs/stt_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_size: str = "base"):
    return whisper.load_model(model_size)


def transcribe(model, audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return result["text"]


if __name__ == "__main__":
    config = load_config()
    model = load_model(config["model_size"])
    text = transcribe(model, "data/input/test_en.mp3")
    print(text)