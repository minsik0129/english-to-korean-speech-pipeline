# English-to-Korean Speech Pipeline

영어 음성 → 영어 텍스트(STT) → 한국어 텍스트(번역) → 한국어 음성(TTS)
end-to-end 파이프라인 구축 프로젝트.

## 상태
STT + 번역 + TTS 세 모듈 완성, `pipeline.py`로 전체 연결 완료.

## 구성
- **STT**: Whisper `base` 모델 (local, CPU)
- **번역**: `facebook/nllb-200-distilled-600M` (local, CPU)
- **TTS**: [HGU-DLLAB/Korean-FastSpeech2-Pytorch](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch) 기반, pretrained 체크포인트로 추론 (vendored, 라이선스는 `src/tts/THIRD_PARTY_LICENSE` 참고)

## 실행
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

TTS 모듈을 쓰려면 `SETUP.md`의 추가 시스템 셋업(mecab 사전 경로)을 먼저 진행해야 함.

체크포인트는 git에 포함 안 됨 — `SETUP.md` 참고해서 별도 다운로드 필요.

```powershell
python src/pipeline.py
```

`data/input/`에 있는 영어 음성 파일을 읽어서 `data/output/`에 한국어 음성 파일을 생성함.