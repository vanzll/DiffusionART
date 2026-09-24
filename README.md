# DiffusionART

**Efficient, Stable and Scalable Reinforcement of Diffusion Model**

[Project page](https://vanzll.github.io/DiffusionART/)

Zhenglin Wan, Jiayu Chen, Zhengfeiyang Zhang, Pengfei Zhou, Wangbo Zhao,
Fangyuan Kong, Pengfei Wan, Kung Gai, Yang You

HPC-AI Lab, National University of Singapore · Kling AI · HKUST

## Code coming soon

We are organizing and cleaning up the research code for release. Training code,
reproducible configurations, and setup instructions are coming soon.

DiffusionART uses aggregation-first reward tilting and adaptive restoration for
efficient diffusion post-training. It supports scalar and pairwise feedback,
with experiments on text-to-image, text-to-video, and image-to-video generation.

## Project page

The project page includes the core insights, quantitative results, and selected
before/after images and playable videos. Its static source is in [`docs/`](docs/).
It is served by GitHub Pages from `main:/docs`; no backend or account is needed.

## Release history

The future code release will be imported as a clean source snapshot. Private
experiment-repository history, logs, credentials, and training artifacts will
not be carried into this repository. Third-party notices and applicable
licenses will be preserved in the release.

## Development

Open `docs/index.html` locally to preview the page. All public media is local to
the site. No analytics, external font service, or third-party JavaScript is used.

`scripts/prepare_assets.py` builds the curated media bundle from separately held
research assets. It does not copy raw experiment configs or private records.

The implementation and its license will be announced with the code release.
