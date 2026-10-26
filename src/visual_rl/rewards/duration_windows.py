"""Opt-in full-duration coverage for fixed-window AV reward models.

Keep native scoring functions and transforms unchanged. Each fixed-length
window produces one native score; only the arithmetic mean of those scores is
used as the clip reward, so a longer clip does not gain extra reward weight.
"""

import math


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def clap_window_seconds(model):
    """Read the actual local CLAP processor's audio limit; never guess it."""
    extractor = getattr(getattr(model, "processor", None), "feature_extractor", None)
    count = getattr(extractor, "nb_max_samples", None)
    rate = getattr(extractor, "sampling_rate", None)
    seconds = getattr(extractor, "max_length_s", None)
    if _positive(count) and _positive(rate):
        duration = count / rate
        if _positive(seconds) and not math.isclose(duration, seconds, rel_tol=1e-6):
            raise ValueError("CLAP processor has inconsistent native window limits")
        return duration
    if _positive(seconds):
        return float(seconds)
    raise ValueError("full-duration CLAP requires a processor with an explicit audio window limit")


def media_windows(video, extra, seconds):
    """Cover both native-rate streams with paired, bounded, overlapping crops.

    Windows have evenly spaced starts from zero through the final possible
    start. Their number is ceil(duration/window), with the final window ending
    at the media tail. Frame/sample boundaries round from the SAME timestamp;
    AV codec endpoint rounding can change their durations by at most one frame.
    A short clip is returned unchanged, including object identities.
    """
    if not _positive(seconds):
        raise ValueError("native reward window duration must be positive")
    fps, rate = extra["fps"], extra["audio_sample_rate"]
    audio = extra["audio"]
    duration = max(len(video) / fps, audio.shape[-1] / rate)
    if duration <= seconds:
        return ((video, extra),)
    frames, samples = math.floor(seconds * fps), math.floor(seconds * rate)
    if min(frames, samples) < 1:
        raise ValueError("native reward window is shorter than one frame/sample")
    count = math.ceil(duration / seconds)
    windows = []
    for index in range(count):
        start = (duration - seconds) * index / (count - 1)
        first_frame, first_sample = round(start * fps), round(start * rate)
        window_video = video[first_frame:first_frame + frames]
        window_extra = {**extra, "audio": audio[..., first_sample:first_sample + samples]}
        if not len(window_video) or not window_extra["audio"].shape[-1]:
            raise ValueError("native reward window has no valid media")
        windows.append((window_video, window_extra))
    return tuple(windows)


async def score_windows(video, extra, seconds, compute):
    """Average already-transformed window scores and matching raw diagnostics."""
    windows = media_windows(video, extra, seconds)
    if len(windows) == 1:
        return await compute(*windows[0])
    results = [await compute(frames, info) for frames, info in windows]
    averaged = {key for key in results[0] if key == "score" or key.endswith("_raw")}
    if "score" not in averaged or any(
            {key for key in result if key == "score" or key.endswith("_raw")} != averaged
            for result in results):
        raise ValueError("reward windows must return matching score/raw diagnostic keys")
    result = dict(results[0])
    for key in averaged:
        values = [float(item[key]) for item in results]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite fixed-window reward")
        result[key] = math.fsum(values) / len(values)
    return result
