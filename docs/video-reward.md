# VideoReward

`rewards.video_reward.VideoReward` wraps one frozen native VideoReward model
and its processor. The [Hunyuan integration](hunyuan-runner.md) supplies the
native inferencer and `build_prompt` function. Pretrained GPU acceptance is
still pending.

The two scoring interfaces intentionally remain distinct:

- `score_paths(paths, prompts)` scores actual MP4s using native preprocessing.
  It first verifies each stream's FPS with `ffprobe`, then calls the native
  scorer with `use_norm=False`, configured RM FPS and `max_pixels=200704`.
  Fixed checkpoint statistics are applied exactly once by the shared reward
  interface. Scoring remains batch-one, as in the paper's video protocol.
- `forward(ContinuousRGBInput(...))` preserves gradients from the reward model
  through frame selection, resize, RGB normalization and Qwen2-VL patch packing.
  This is the ReFL objective input, not an MP4 or uint8 straight-through estimator.

Both return the shared `ScoreResult` with raw VQ/MQ/TA, fixed-normalized values,
explicit weights and their weighted sum. Reporting must label continuous and
MP4 rewards separately; identical models/statistics do not make these inputs
equivalent. Use MP4 scores for comparisons with the other algorithms.

The default protocol is 8 video FPS, 4 RM FPS, uniform temporal sampling and
`num_frames=None`. This samples 26 frames from a 53-frame video, including both
endpoints. A fixed two-frame scorer or an 18 FPS MP4 is rejected rather than
silently scored under a different protocol. The metadata check does not replace
full decode validation in the encoder/publication layer.

The adapter freezes reward weights but keeps input gradients. It does not
change model placement, VAE checkpointing or CPU offload policy. Backend code
owns those choices and must load a compatible pinned native model/processor.
CPU tests verify normalization, differentiability, frozen weights, FPS rejection
and patch order against Transformers. Actual VideoReward-weight/native decoded
pixel parity and pretrained GPU execution remain acceptance work.
