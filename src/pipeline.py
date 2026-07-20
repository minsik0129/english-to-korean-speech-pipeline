import os

from stt.whisper_stt import load_config as load_stt_config, load_model as load_whisper_model, transcribe
from translation.translator import load_config as load_translation_config, load_model as load_translation_model, translate
from tts.infer import load_config as load_tts_config, load_fastspeech2, load_vocoder, text_to_tensor, synthesize


def run_pipeline(audio_path: str, output_path: str):
    # 1. STT: 영어 음성 -> 영어 텍스트
    stt_config = load_stt_config()
    whisper_model = load_whisper_model(stt_config["model_size"])
    english_text = transcribe(whisper_model, audio_path)
    print("STT 결과:", english_text)

    # 2. 번역: 영어 텍스트 -> 한국어 텍스트
    translation_config = load_translation_config()
    tokenizer, translation_model = load_translation_model(
        translation_config["model_name"], translation_config["src_lang"]
    )
    korean_text = translate(
        tokenizer, translation_model, translation_config["tgt_lang"],
        english_text, translation_config["max_length"]
    )
    print("번역 결과:", korean_text)

    # 3. TTS: 한국어 텍스트 -> 한국어 음성
    tts_config = load_tts_config()
    device = tts_config["device"]
    fastspeech2_model = load_fastspeech2(tts_config["checkpoint_step"], device)
    vocoder = load_vocoder()
    text_tensor = text_to_tensor(korean_text, device)
    synthesize(fastspeech2_model, vocoder, text_tensor, device, output_path)
    print("최종 저장:", output_path)


if __name__ == "__main__":
    audio_path = "data/input/test_en.mp3" #어떤 경로의 무슨 파일을 넣을지 확인해야. 입력 파일(영어음성)
    output_path = "data/output/pipeline_result.wav" #출력되는 아웃풋 파일. 이름 임의로 변경가능
    os.makedirs("data/output", exist_ok=True)
    run_pipeline(audio_path, output_path)
