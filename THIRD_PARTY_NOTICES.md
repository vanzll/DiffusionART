# Third-Party Notices

Original project code is licensed under the Apache License, Version 2.0; see
`LICENSE` and `NOTICE`. Third-party portions retain the terms and notices below.
The project license does not grant rights over model weights or datasets.
Review each asset's terms separately before downloading or redistributing it.

## Flow-GRPO

Flow transition equations and migrated research implementations trace to
https://github.com/yifan123/flow_grpo. The source research tree retains this
license, reproduced here for the derived portions:

MIT License

Copyright (c) 2025 Jie Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Algorithm References

- Decord: https://github.com/dmlc/decord. `third_party/decord` contains a
  repackaged official PyPI 0.6.0 Linux wheel under the retained Apache-2.0
  license. Only its internal wheel tag and RECORD are corrected; runtime
  code and shared libraries are unchanged. Source hashes, the rebuild script,
  and license notices are included alongside the wheel.
- GenEval training scorer: https://github.com/yifan123/reward-server at
  `8245af22b75fe640950a880ad31fe702a5d9a014`, `reward_server/gen_eval.py`.
  Its MIT license (Copyright 2023 Jie Liu) is preserved in
  `licenses/reward-server-MIT.txt`. `rewards/geneval_engine.py` is a
  dependency-injected port preserving the scoring rules. Native model loading
  follows the separately audited deployment changes documented in
  `docs/geneval-migration.md`. Detector/CLIP weights and benchmark data are not
  redistributed; their individual terms and original GenEval lineage still
  require review before publishing a reproduction bundle.

- DiffusionNFT: https://github.com/NVlabs/DiffusionNFT. Its Apache-2.0
  license is preserved in `licenses/DiffusionNFT-APACHE-2.0.txt`; the upstream
  license Git blob inspected for this snapshot is
  `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`.
  The migrated deterministic sampler follows the research `nft_solver.py`,
  which attributes its solver to MixGRPO. That provenance is retained in
  `training/nft_sampling.py`; upstream solver/license lineage remains a release
  audit item, not a blanket redistribution authorization.
- DanceGRPO: https://github.com/XueZeyue/DanceGRPO, inspected at
  `15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344`. VideoReward's inference ABI,
  uniform frame selection and resize geometry are referenced by
  `rewards/video_reward.py`, through the research ReFL adapter. The source
  `vision_process.py` attributes its origins to
  https://github.com/kq-chen/qwen-vl-utils. The inspected DanceGRPO Apache-2.0
  license is preserved in `licenses/DanceGRPO-APACHE-2.0.txt` (Git blob
  `a7b8b5e6a6dc555ad43165aa5b8f65c120032468`). Qwen2-VL patch packing follows
  the native processor layout; tests compare against the separately installed
  Transformers implementation. The native model/inferencer code and weights
  are not bundled; their dependency and model-license acceptance remains open.
  The SD3 Dance adapter references the flow-SDE equations and final-x0 scoring
  in `fastvideo/train_grpo_flux_lora.py` at that same revision. Its prompt-major
  grouping and timestep selection adaptations are documented separately.
- UniRL ReFL: https://github.com/Tencent-Hunyuan/UniRL/tree/main/experimental/refl
- PickScore: https://github.com/yuvalkirstain/PickScore and
  https://huggingface.co/yuvalkirstain/PickScore_v1. The native reward adapter
  loads separately installed Transformers components; no model weights,
  processor assets or Pick-a-Pic dataset are redistributed.
- CLIP: https://github.com/openai/CLIP and
  https://huggingface.co/openai/clip-vit-large-patch14. The CLIPScore adapter
  follows the research wrapper's tensor preprocessing and logit/30 convention,
  loading separately installed Transformers and torchvision implementations.
  No CLIP source tree, model weights or processor assets are bundled.
- HPSv2.1: https://github.com/tgxs002/HPSv2 and
  https://huggingface.co/xswu/HPSv2. The reward adapter uses the separately
  installed Apache-2.0 `hpsv2==1.2.0` package's ViT-H-14 model, tokenizer and
  preprocessing. `third_party/hpsv2` contains an asset-complete wheel with
  license notices and tokenizer vocabulary. An unused GUI import is removed;
  model, preprocessing and scoring code remain unchanged. These installation
  repairs and source hashes are documented there. No model
  weights are bundled. The v2.1
  checkpoint is downloaded separately at a fixed Hub revision and verified
  by SHA256; scoring preserves the historical paired feature-dot-product rule.
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR, adapter reference
  version `v2.9.1`. Recognition is delegated to a separately installed engine;
  no PaddleOCR source, dictionaries or inference weights are redistributed.
  The text reward follows the research wrapper and uses separately installed
  RapidFuzz (https://github.com/rapidfuzz/RapidFuzz) for Levenshtein distance.
- Improved Aesthetic Predictor: https://github.com/christophschuhmann/improved-aesthetic-predictor
  at `fe88a163f4661b4ddabba0751ff645e2e620746e`. The native Aesthetic adapter
  follows its scoring-head architecture through the research wrapper. The
  upstream LICENSE is Apache-2.0, Git blob
  `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`, byte-identical to the Apache-2.0
  text already preserved in `licenses/DiffusionNFT-APACHE-2.0.txt`.
  Model/head weights are downloaded separately, not redistributed here.

Kernel and model-contract documents identify adapted equations and source
revisions. These links are not a blanket license to vendor further source.
Model libraries remain separately installed dependencies; native DanceGRPO and
UniRL modules have not been bundled. Their notices must be preserved if later
vendored.

## Visual Assets

The Star History chart uses Patrick Hand SC, licensed under the SIL Open Font
License 1.1. Its license is preserved in `assets/readme/fonts/OFL.txt`.

## MiniMax-H3 reward adapters

The five files audiobox.py, clap.py, desync.py, hpsv3_reward.py and
videoalign.py under src/visual_rl/rewards/_native_av are adapted from
ByteDance verl-omni at ca61e80c42c883e1efa161f6b609cf897fba1787,
under the Apache License 2.0 (licenses/Apache-2.0.txt). Original copyright
headers are retained. Exact URLs, source SHA256 and local import changes are
in that package's provenance.json. Native scoring transforms are preserved;
service/device/media utilities are replaced by explicit standalone boundaries.

Model weights and external scorer source/assets are not redistributed.
Their independent repository revisions and licenses must be supplied and
recorded in the asset manifest; the software does not accept licenses for users.

The opt-in sixth scorer's decoded preprocessing bridge in
`src/visual_rl/rewards/_native_av/imagebind.py` is adapted from Meta
ImageBind's `imagebind/data.py` at
`53680b02d7e37b19b124fa37bae4b6c98c38f5be`:
https://github.com/facebookresearch/ImageBind.
Portions Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
This derived preprocessing is licensed under CC-BY-NC-SA-4.0; the full upstream
license is preserved in `licenses/ImageBind-CC-BY-NC-SA-4.0.txt`. Local changes
replace file decoders with canonical decoded tensors, explicitly downmix stereo,
namespace and byte-verify separately supplied model source, and require a local
checkpoint. The upstream clip timing/frame/resize conventions also refer to
ImageBind's pinned PyTorchVideo revision
`6cdc929315aab1b5674b6dcf73b16ec99147735f`; PyTorchVideo is separately sourced
under Apache-2.0 and no PyTorchVideo source is bundled. Exact source metadata
is recorded in `_native_av/provenance.json` and `imagebind.SOURCE_SHA256`.
ImageBind's noncommercial/share-alike terms apply independently of this
repository's Apache-2.0 license. ImageBind implementation files and weights
remain separately supplied, with their own explicit manifest provenance.
