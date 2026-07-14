import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_config(config_path: str = "configs/translation_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_name: str, src_lang: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def translate(tokenizer, model, tgt_lang: str, text: str, max_length: int) -> str:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=max_length,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    config = load_config()
    tokenizer, model = load_model(config["model_name"], config["src_lang"])
    text = "Harry came letting Bloom to victory over Norway and the World Cup."
    korean = translate(tokenizer, model, config["tgt_lang"], text, config["max_length"])
    print(korean)