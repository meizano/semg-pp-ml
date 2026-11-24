# Related work survey: what to take for `semg-pp-ml`

Survey of the 28 public sEMG / EMG repositories listed in the request (metadata and READMEs read from GitHub, 2026-11-24). Each row says what the repo is, whether it is worth using, and what this project should take from it.

Licence matters before copying code: this project's README says MIT, but there is no `LICENSE` file in the repo yet. GPL/AGPL code (EMGdecomPy, openhdemg is BSD-3, HDC-EMG, semgsim, CIMS dual-stream) can be read for ideas, but its code should not be copied into an MIT project. Code in this repo's augmentation module was written from scratch.

## 1. Data augmentation and artificial data (most relevant to this project)

| Repo | Lang / licence | What it is | What to take |
|---|---|---|---|
| [Josef-Djarf/sEMG-Sim](https://github.com/Josef-Djarf/sEMG-Sim) | Python / MIT | Master's-thesis sEMG simulator: motor-unit pool, MUAPs, electrode geometry, volume conductor | Model structure for synthetic data, as described in its README: motor-unit properties, volume conductor, electrode setup. **Its design informed `semg_pp_ml.augmentation.synthetic`** (written from scratch; I read the README, not the full source). |
| [ime-luebeck/semgsim](https://github.com/ime-luebeck/semgsim) | R / GPL-3 | Simulation of sEMG and force from a motor-unit pool (Petersen & Rostalski 2019) | The more complete physiological reference. Use it to validate the synthetic generator's EMG-force relation; do not copy code (GPL). |
| [Yazhou-Z/EMG_CNN](https://github.com/Yazhou-Z/EMG_CNN) | Python / none | CNN and reservoir model on a Kaggle sEMG set; expands data by **interpolation and slicing** | Confirms that interpolation/slicing helps a small dataset. Implemented as `time_warp` and window-level augmentation. |
| [scheck-k/ste-gan](https://github.com/scheck-k/ste-gan) | Python / none | GAN that predicts articulatory EMG from speech (800 Hz output) | Generative approach for artificial EMG. Heavier than the physics-based route; a later option, not for this step. |
| [Zeng-Jia/CMKD-MINDS](https://github.com/Zeng-Jia/CMKD-MINDS) | Python / none | Cross-modality knowledge distillation, A-mode ultrasound to sEMG | Transfers knowledge from another modality instead of making synthetic windows. Interesting if ultrasound data is ever available. |
| [Zeng-Jia/awesome-sEMG-hand-gesture-recognition](https://github.com/Zeng-Jia/awesome-sEMG-hand-gesture-recognition) | Python / none | PyTorch re-implementations (WeiNet, XceptionTime, BiTCN) on NinaPro DB1/DB2 | Reference architectures and NinaPro preprocessing notebooks (`extractFile_db*`, `process_db*`). Good model-zoo candidates. |
| [bladesav/sEMG-CNNTransformer](https://github.com/bladesav/sEMG-CNNTransformer) | Python / none | Lightweight CNN-Transformer on trans-radial sEMG (NinaPro DB2 zipped) | Matches the existing `cnn_transformer` / MAFN models. Compare accuracy and parameter count. |

## 2. Datasets

| Repo | Lang / licence | What it is | What to take |
|---|---|---|---|
| [x-labs-xyz/awesome-emg-data](https://github.com/x-labs-xyz/awesome-emg-data) | none | Curated list of public EMG datasets and papers, focus on raw EMG | Best starting point for finding more data to pair with NinaPro DB8. |
| [aljazfrancic/myo-readings-dataset](https://github.com/aljazfrancic/myo-readings-dataset) | Jupyter / MIT | Myo armband (8 ch, 200 Hz, int8), 8 wrist/hand gestures, many participants, curated sessions | Second real dataset. Needs a loader (8 channels, signed-byte values in `[-128, 127]`). |
| [wanglichenxj/EV-Action](https://github.com/wanglichenxj/EV-Action-Electromyography-Vision-Multi-Modal-Action-Dataset) | none | Multimodal EMG + RGB/depth/skeleton action dataset (FG 2020) | Useful if vision + EMG fusion is ever studied. Not needed now. |
| [felixmaldonadoos/Facial-EMG-and-Kinematics](https://github.com/felixmaldonadoos/Facial-EMG-and-Kinematics) | Jupyter / Unlicense | Facial EMG with landmark kinematics, Arduino-triggered | Preprocessing class and sample data for facial EMG. Low priority. |

## 3. Signal processing and feature extraction

| Repo | Lang / licence | What it is | What to take |
|---|---|---|---|
| [JingweiToo/EMG-Feature-Extraction-Toolbox](https://github.com/JingweiToo/EMG-Feature-Extraction-Toolbox) | MATLAB / BSD-3 | 40 time-domain and related EMG features (MAV, WL, ZC, SSC, …) | **Recommended next step:** a time-domain feature module (MAV, RMS, WL, ZC, SSC, WAMP, …) for classical models (SVM, RF, KNN already in the project). Re-implement in Python. |
| [axopy/pytrigno](https://github.com/axopy/pytrigno) | Python / MIT | Streams EMG from the Delsys Trigno system (2000 Hz, ±11 mV) | Relevant only if live Delsys acquisition is added. Not for offline NinaPro work. |
| [resurfemg-org/ReSurfEMG](https://github.com/resurfemg-org/ReSurfEMG) | Jupyter / Apache-2.0 | Respiratory EMG analysis library (filters, ECG removal, dashboard) | Reference for filter and analysis settings in a sEMG library with a notebook interface. Respiratory EMG is a different domain. |
| [GiacomoValliPhD/openhdemg](https://github.com/GiacomoValliPhD/openhdemg) | Python / BSD-3 | High-density EMG decomposition and analysis | Reference for HD-EMG workflows. Could be a dependency if HD-EMG data is added. |
| [The-Motor-Unit/EMGdecomPy](https://github.com/The-Motor-Unit/EMGdecomPy) | Python / GPL-3 | Blind-source decomposition into motor units (Negro et al. 2016) | Idea reference for motor-unit decomposition (also validates the synthetic MU model). GPL: do not copy code. |
| [kkapre/EMG-Decomposition](https://github.com/kkapre/EMG-Decomposition) | MATLAB / none | MUAP shape extraction with filter, spike detection, k-means | Small reference implementation. |
| [gbhand/EMG-CNN](https://github.com/gbhand/EMG-CNN) | Jupyter / none | CNN for muscle fatigue, Stockwell transform windows | Fatigue labels and STFT/ST input representation. Optional feature. |

## 4. Classification methods and benchmarks

| Repo | Lang / licence | What it is | What to take |
|---|---|---|---|
| [malele4th/sEMG_DeepLearning](https://github.com/malele4th/sEMG_DeepLearning) | Jupyter / none (Chinese) | Tutorial on deep learning for sEMG gesture recognition (476 stars) | Checklist of training practice and common baselines. Cross-check the training details against the project's CNN/LSTM pipeline. |
| [Laknath1996/sEMG-Hand-Gesture-Recognition](https://github.com/Laknath1996/sEMG-Hand-Gesture-Recognition) | Python / MIT | Real-time gesture recognition from Temporal Muscle Activation maps (ICASSP 2020/2021) | A real-time inference baseline. TMA maps are a good input representation to try. |
| [abbas-rahimi/HDC-EMG](https://github.com/abbas-rahimi/HDC-EMG) | MATLAB / GPL-3 | Hyperdimensional computing for EMG gestures, compared with SVM | Low-power alternative classifier. Could be ported as a model in `models/embedded.py`. Note: GPL-3. |
| [lawrobel/EMG-Data-Analysis](https://github.com/lawrobel/EMG-Data-Analysis) | Jupyter / none | Hand-movement classification with several classic models | Small reference for classic-ML baselines. Low priority. |
| [xun911/CIMS_sEMG_dualstreamCNN](https://github.com/xun911/CIMS_sEMG_dualstreamCNN) | Python / AGPL-3 | Dual-stream CNN for NinaPro DB2 (intra-subject) | Architecture idea: two input streams (raw + feature). AGPL: read only. |
## 5. Hardware and application demos (no direct code reuse)

| Repo | Lang / licence | What it is | Relevance |
|---|---|---|---|
| [upsidedownlabs/Muscle-BioAmp-Candy](https://github.com/upsidedownlabs/Muscle-BioAmp-Candy) | C++ / NOASSERTION | Small EMG sensor board (gain ×2420, 72–720 Hz band-pass) | Hardware reference. Band-pass of 72–720 Hz is a useful reference for filter presets. Licence is unclear. |
| [ohheyitskartik/Electromyography-sensor](https://github.com/ohheyitskartik/Electromyography-sensor) | none / MIT | Amplify, filter and rectify EMG to 0–3 V for an ADC | Hardware reference. |
| [hm-badhon/Prosthetic-Limb-Control-using-Electromyography-EMG-Sensor](https://github.com/hm-badhon/Prosthetic-Limb-Control-using-Electromyography-EMG-Sensor) | C++ / none | Robotic-arm control with an EMG sensor | Application demo only. |
| [yuesha-yc/being-doomfist](https://github.com/yuesha-yc/being-doomfist) | C++ / GPL-3 | Game control with EMG (CalHacks) | Application demo only. |
| [gqbaotran/Project-Automotive-Control-using-Surface-Electromyography](https://github.com/gqbaotran/Project-Automotive-Control-using-Surface-Electromyography) | MATLAB / none | Vehicle braking assist from sEMG (UTokyo internship) | Application demo (vehicle control from sEMG). |

## Recommendations for this repository

Done in this change:
1. **Augmentation module** (`src/semg_pp_ml/augmentation/transforms.py`): noise at a target SNR, amplitude scaling, time warping, electrode shift, cross-talk, channel dropout, baseline wander, mains hum, same-class mixup. Tests in `tests/test_augmentation.py`.
2. **Synthetic sEMG generator** (`synthetic.py`): motor-unit pool with size-principle recruitment, MUAP templates, class-specific spatial patterns. Output rises in RMS with force level (tested). CLI: `python -m semg_pp_ml.augmentation --synthetic ...`.

Worth doing next, in rough priority:
3. **Replace the silent random fallback in `data/loader.py`.** When the NinaPro folder is missing, the loader returns pure random noise with a sine (`np.random.randn`). Swap in `generate_synthetic_semg` so demos produce plausible data. Behaviour change, so do it on purpose.
4. **Time-domain feature module** from the JingweiToo toolbox (MAV, RMS, WL, ZC, SSC, WAMP) for the classical models.
5. **Myo dataset loader** (aljazfrancic) as a second real dataset for cross-dataset tests.
6. **Validate the synthetic model** against semgsim on EMG-force relation, and against real NinaPro RMS by class, before using it for training.
7. **Add a LICENSE file** to match the MIT claim in the README, and ignore `__pycache__` (compiled `.pyc` files are currently committed).
