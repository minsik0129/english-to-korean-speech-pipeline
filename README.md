# English-to-Korean Speech Pipeline

영어 음성 → 영어 텍스트(STT) → 한국어 텍스트(번역) → 한국어 음성(TTS)
end-to-end 파이프라인 구축 프로젝트.

## 상태
프로젝트 구조 초기화 완료. 모듈 구현 진행 중.

## 구성
- STT: Whisper (local)
- 번역: Helsinki-NLP/opus-mt-en-ko
- TTS: [HGU-DLLAB/Korean-FastSpeech2-Pytorch](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch) 기반, pretrained 체크포인트로 추론