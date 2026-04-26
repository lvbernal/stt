# Whisper Audio Transcription CLI

Small Python CLI app that converts audio files in a folder to text files using OpenAI Whisper.

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: Whisper needs `ffmpeg` installed on your system.

## 2) Run (Folder -> Folder)

```bash
python transcribe.py path/to/input_audios path/to/output_texts
```

Scan subfolders too:

```bash
python transcribe.py path/to/input_audios path/to/output_texts --recursive
```

Use a different model/language:

```bash
python transcribe.py path/to/input_audios path/to/output_texts --model small --language es
```

More stable decoding (slower, usually better quality):

```bash
python transcribe.py path/to/input_audios path/to/output_texts --model large --beam-size 8 --temperature 0
```

Overwrite existing `.txt` outputs:

```bash
python transcribe.py path/to/input_audios path/to/output_texts --overwrite
```

Single-line output (legacy behavior):

```bash
python transcribe.py path/to/input_audios path/to/output_texts --single-line
```

Include timestamps per line:

```bash
python transcribe.py path/to/input_audios path/to/output_texts --timestamps
```

## CLI Options

```text
positional:
  input_dir             input folder with audio files
  output_dir            output folder for .txt files

optional:
  --model               whisper model (default: medium)
  --language            language code (en, es, fr, ...)
  --device              torch device (cpu, cuda)
  --temperature         decoding temperature
  --beam-size           beam width for decoding stability
  --best-of             candidates for sampling
  --condition-on-previous-text / --no-condition-on-previous-text
  --fp16 / --no-fp16    force fp16 on/off
  --recursive           include audio files in subfolders
  --overwrite           overwrite existing output files
  --single-line         output one line for whole file
  --timestamps          include timestamps on each line
```
