# MiniMax-H3 audio-video interface

Native latent, media and algorithm contracts. See [setup and recipes](minimax-h3.md).

## Shared ABI v1

- `visual_rl.training.audio_video_types.AVState(video, audio, video_mask=None,
  audio_mask=None)`: native floating latent batches; boolean feature-prefix/full
  masks; `select(rows)`, `to(device,dtype=None)`, `detached_cpu()` preserve identity.
- `AVMedia(video,audio,fps,sample_rate)`: RGB `[B,3,T,H,W]` in `[0,1]`, waveform
  `[B,channels,samples]`, explicit rates, no silent trimming/resampling.
- H3 adapter `predict_velocity(state,video_sigma,audio_sigma,conditioning)`
  returns AVState in public `dz/dsigma` convention; both outputs differentiable.
  `decode(state)` returns AVMedia. `trainable_module` is the transformer.
  Adapter conditioning exposes `select(rows)` and `to(device)`.
- Generic training prediction callback:
  `predict(state,video_sigma,audio_sigma,rows) -> AVState`, where rows index the
  original prompt-major batch, independently of physical microbatching.
- Keep both actual schedules, states, old velocities and stochastic replay
  observations; never infer equal sigma from a shared loop index. Cache/old
  tensors detached, current predictions differentiable.

## Frozen science

ART U2: every prompt contributes K/2 rows per window; complete-K routed covariance
frozen across both; independent modality RMS/mask/sigma/feature mean and equal
loss weights. Exclude final rollout transition.

NFT U1: adopt plan 6.3 recommended AV extension; no existing shared-scalar H3
contract is overridden. Routed rewards, prompt group centering/global population
std +1e-4 separately per modality; source-matched shared fresh-noising timestep
distribution (`floor(1000*sigma)/1000`); beta1/advclip5/residual eps1e-5; separate normalized-x0 mirror losses
equally weighted. Old frozen throughout; one refresh after successful update,
decay min(j*.00125,.6), j0 copies current. No pretrained reference object,
forward or checkpoint payload for any of the three algorithms.

Fast U2: pinned H3 CPS, continuous sparse length3 window, half-open candidate range [0,8), seed42; observed-next-state replay; combine
.5*video feature-mean logp + .5*audio feature-mean logp BEFORE one PPO ratio;
shared scalar advantage. Full-group statistics precede disjoint row windows.
DeSync appears in both routed rewards but shared scalar only once.
