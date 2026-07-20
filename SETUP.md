# 로컬 환경 추가 셋업 (git으로 관리 안 되는 부분)

이 프로젝트를 다른 컴퓨터에서 그대로 클론했을 때, 아래 작업을 반복해야 TTS 모듈이 정상 동작한다.
Windows + 경로에 공백이 있는 폴더에서 한국어 형태소분석기(`eunjeon`/mecab)를 쓸 때 생기는 문제 때문이다.

## 1. pretrained 체크포인트 다운로드
- FastSpeech2: https://drive.google.com/file/d/1qkFuNLqPIm-A5mZZDPGK1mnp0_Lh00PN/view
- VocGAN: https://drive.google.com/file/d/1GxaLlTrEhq0aXFvd_X1f4b-ev7-FH8RB/view

받은 뒤 `src/tts/checkpoints/`에 배치.

## 2. mecab 사전 데이터를 공백 없는 경로로 복사
Windows에서 MeCab의 C++ 인자 파서가 경로 중간의 공백을 처리 못 하는 문제가 있음.
프로젝트 폴더 경로에 공백이 있으면(`C:\End to End translator` 등) 반드시 필요.

```powershell
mkdir C:\mecab_data
Copy-Item "venv\Lib\site-packages\eunjeon\data\*" -Destination "C:\mecab_data\" -Recurse
```

## 3. eunjeon 패키지 패치
`venv\Lib\site-packages\eunjeon\_mecab.py`에서 `Mecab.__init__`의 기본 `dicpath`를
1번에서 복사한 공백 없는 경로를 가리키도록 수정.

원본:
```python
def __init__(self, dicpath=os.path.abspath(os.path.join(installpath, 'data/mecabrc'))):
```
수정:
```python
def __init__(self, dicpath=r"C:\mecab_data\mecabrc"):
```