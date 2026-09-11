# Third-Party Licenses and Attributions

`mdd-service-de` is licensed under the [MIT License](LICENSE).

This project incorporates, depends upon, or interfaces with third-party software, libraries, and machine learning models. Below is an overview of their respective licenses and attribution notices.

---

## 1. Python Dependencies

| Package | License | Copyright / Upstream Project |
| :--- | :--- | :--- |
| **fastapi** | MIT | Copyright (c) 2018 Sebastián Ramírez |
| **uvicorn** | BSD-3-Clause | Copyright (c) 2017-present, Encode OSS Ltd. |
| **torch**, **torchvision**, **torchaudio** | BSD-3-Clause | Copyright (c) 2016-present, Meta Platforms, Inc. and affiliates. |
| **transformers** | Apache-2.0 | Copyright 2018-The Hugging Face team. |
| **accelerate** | Apache-2.0 | Copyright 2021 The HuggingFace Team. |
| **openpronounce** | MIT | Copyright (c) 2024 Jean-François Lépine |
| **sounddevice** | MIT | Copyright (c) 2015-2024 Matthias Geier (underlying PortAudio under PortAudio license) |
| **soundfile** | BSD-3-Clause | Copyright (c) 2013-2024 Bastian Bechtold |
| **requests** | Apache-2.0 | Copyright 2019 Kenneth Reitz |
| **python-multipart** | Apache-2.0 | Copyright 2014 Andrew Dunai, Marcelo Trylesinski |
| **numpy** | BSD-3-Clause | Copyright (c) 2005-2024, NumPy Developers |
| **pytest** | MIT | Copyright (c) 2004-2024 Holger Krekel and others |

---

## 2. Dynamic Library Disclosures

### `libsndfile` (LGPL-2.1-or-later)
`python-soundfile` dynamically links to unmodified pre-compiled binaries of `libsndfile` licensed under the **GNU Lesser General Public License (LGPL) v2.1 or later**.
- Source code for `libsndfile`: [https://github.com/libsndfile/libsndfile](https://github.com/libsndfile/libsndfile)
- In accordance with LGPL v2.1 Section 6, users may replace or re-link `libsndfile` with compatible versions in their runtime environment.

---

## 3. Pre-trained AI Models

### `Qwen/Qwen3-ASR-0.6B` / `Qwen/Qwen3-ASR-0.6B-hf`
- **License:** [Apache License 2.0](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE)
- **Copyright:** Copyright 2026 Alibaba Cloud (Qwen Team).

### `facebook/wav2vec2-lv-60-espeak-cv-ft`
- **License:** [Apache License 2.0](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
- **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.

### Mozilla Common Voice German Checkpoints
- **Dataset / Model License:** Derived from Common Voice under [Creative Commons CC0 1.0 Universal (Public Domain Dedication)](https://commonvoice.mozilla.org/).

---

## 4. System Prerequisites (External Tools)

The following tools are external operating system binaries and are not bundled or distributed within this repository:

- **`ffmpeg`** ([LGPL v2.1+ / GPL v2.0+](https://www.ffmpeg.org/legal.html)): Invoked out-of-process via standard OS execution boundaries for audio decoding and resampling.
- **`espeak-ng`** ([GPL v3.0+](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING)): Invoked out-of-process for German phonetic grapheme-to-phoneme (G2P) conversion.
