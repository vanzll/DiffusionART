"""Small static gallery, with opt-in explicit Jupyter Contents mapping."""

import json
import posixpath
from dataclasses import dataclass
from html import escape
from urllib.parse import quote, urlsplit

from .manifest import _integer, relative_path


@dataclass(frozen=True)
class JupyterContents:
    """Map manifest media-root-relative paths to one explicit Contents root.

    base_url includes the deployment prefix, e.g. https://host/user/name/.
    contents_root is the logical path of media_root, e.g. experiments/run42.
    Neither value is inferred from physical mounts, referrers, or page URLs.
    """

    base_url: str
    contents_root: str

    def __post_init__(self):
        if not isinstance(self.base_url, str) or any(
            ord(c) <= 32 or ord(c) == 127 for c in self.base_url
        ):
            raise ValueError("invalid Jupyter base_url")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or "\\" in self.base_url
            or "%" in url.path
            or "?" in self.base_url
            or "#" in self.base_url
        ):
            raise ValueError("base_url requires HTTP(S), without credentials/query/fragment")
        # Accessing port validates malformed and out-of-range ports.
        url.port
        prefix = url.path.removesuffix("/").removeprefix("/")
        if url.path == "//":
            raise ValueError("noncanonical base_url path")
        relative_path(prefix, allow_empty=True)
        relative_path(self.contents_root, allow_empty=True)
        object.__setattr__(self, "base_url", self.base_url.rstrip("/") + "/")

    def logical_path(self, path):
        relative_path(path)
        return "/".join(p for p in (self.contents_root, path) if p)

    def contents_url(self, path):
        return (
            self.base_url
            + "api/contents/"
            + quote(self.logical_path(path), safe="/")
            + "?content=1&type=file&format=base64"
        )


def _script_json(value):
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_gallery(
    manifest,
    *,
    audit,
    gallery_path="index.html",
    jupyter=None,
    max_media_bytes=64 * 1024 * 1024,
    max_concurrent=2,
    max_loaded=4,
    timeout_ms=15000,
):
    """Return HTML only, without filesystem/network writes or service startup.

    gallery_path is relative to the same media root as every SampleRecord.path.
    The caller publishes beside the media and owns bounded/failover storage I/O.
    Audit is a generation-time snapshot; browsers report later access failures.
    """
    relative_path(gallery_path)
    for name, value, cap in (
        ("max_media_bytes", max_media_bytes, 256 * 1024 * 1024),
        ("max_concurrent", max_concurrent, 4),
        ("max_loaded", max_loaded, 4),
        ("timeout_ms", timeout_ms, 120000),
    ):
        _integer(value, name, 1)
        if value > cap:
            raise ValueError(f"{name} exceeds hard cap {cap}")
    if jupyter is not None and not isinstance(jupyter, JupyterContents):
        raise ValueError("jupyter must be an explicit JupyterContents mapping")
    if (
        audit.manifest_sha256 != manifest.sha256
        or len(audit.samples) != len(manifest.expected_sample_ids)
        or {a.sample_id for a in audit.samples} != set(manifest.expected_sample_ids)
    ):
        raise ValueError("audit does not match manifest identity")
    records = {s.sample_id: s for s in manifest.samples}
    audits = {a.sample_id: a for a in audit.samples}
    rows, media = [], []
    for sample_id in manifest.expected_sample_ids:
        sample, check = records.get(sample_id), audits[sample_id]
        status = check.media_status + ("; missing_scores" if not check.scores_present else "")
        if check.detail:
            status += ": " + check.detail
        if sample is None:
            rows.append(
                "<tr><th>"
                + escape(sample_id)
                + '</th><td colspan="7">'
                + escape(status)
                + "</td></tr>"
            )
            continue
        score = sample.scores
        raw = "missing" if score is None else json.dumps(dict(score.raw), sort_keys=True)
        fixed = (
            "missing"
            if score is None or score.fixed_normalized is None
            else json.dumps(dict(score.fixed_normalized), sort_keys=True)
        )
        weighted = (
            "missing"
            if score is None
            else f"{score.weighted:g} ({score.space}); weights={dict(score.weights)}"
        )
        player = '<span class="status">' + escape(status) + "</span>"
        if check.media_status == "verified":
            suffix = posixpath.splitext(sample.path)[1].lower()
            mime = {".mp4": "video/mp4", ".webm": "video/webm"}.get(suffix)
            if mime is None:
                player += "<p>Unsupported preview format</p>"
            else:
                index = len(media)
                local = "./" + quote(
                    posixpath.relpath(sample.path, posixpath.dirname(gallery_path) or "."), safe="/"
                )
                media.append(
                    {
                        "sample_id": sample_id,
                        "local": local,
                        "sha256": sample.content_sha256,
                        "mime": mime,
                        "url": None if jupyter is None else jupyter.contents_url(sample.path),
                        "logical_path": None
                        if jupyter is None
                        else jupyter.logical_path(sample.path),
                    }
                )
                player += (
                    f'<div class="player" data-media="{index}"><video controls preload="none" playsinline aria-label="'
                    + escape(sample_id, quote=True)
                    + '"></video><button type="button" class="load" title="Load video" aria-label="Load video">&#9654;</button> '
                    '<button type="button" class="release" title="Release video" aria-label="Release video">&#9632;</button>'
                    '<p class="message" role="status">Not loaded</p></div>'
                )
        cells = [
            sample_id,
            sample.prompt,
            f"seed={sample.seed}; step={sample.optimizer_step}; frames={sample.framecount}; fps={sample.fps:g}",
            sample.path,
            raw,
            fixed,
            weighted,
        ]
        rows.append(
            "<tr>"
            + "".join("<td>" + escape(c) + "</td>" for c in cells)
            + "<td>"
            + player
            + "</td></tr>"
        )
    payload = {
        "media": media,
        "max_bytes": max_media_bytes,
        "max_concurrent": max_concurrent,
        "max_loaded": max_loaded,
        "timeout_ms": timeout_ms,
    }
    return (
        _HEAD
        + f"<h1>Evaluation: step {manifest.optimizer_step}</h1>"
        + f'<p id="completion">{"Complete" if audit.complete else "Incomplete"}: {audit.completed_count}/{len(manifest.expected_sample_ids)} samples</p>'
        + '<p class="digest">Manifest SHA-256: '
        + manifest.sha256
        + "</p>"
        + '<div class="table"><table><thead><tr>'
        + "".join(
            "<th>" + s + "</th>"
            for s in (
                "Sample ID",
                "Prompt",
                "Identity / video",
                "Path",
                "Raw scores",
                "Fixed-normalized scores",
                "Weighted score",
                "Media status",
            )
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + '<script type="application/json" id="gallery-data">'
        + _script_json(payload)
        + "</script><script>"
        + _SCRIPT
        + "</script></body></html>"
    )


_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Evaluation gallery</title>
<style>
*{box-sizing:border-box}body{font:14px/1.4 system-ui,sans-serif;margin:16px;color:#222;background:white;letter-spacing:0}
h1{font-size:22px}p,.digest{overflow-wrap:anywhere}.table{overflow:auto}table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #ccc;text-align:left;vertical-align:top;padding:8px;min-width:130px;max-width:280px;overflow-wrap:anywhere}
th{background:#f2f2f2}video{display:block;width:240px;height:160px;object-fit:contain;background:#161616}
button{width:34px;height:30px;margin-top:6px;border:1px solid #aaa;border-radius:4px;background:white;cursor:pointer}
.message,.status{font-size:12px}.digest{font-size:12px;color:#555}button:focus-visible{outline:2px solid #007e80}
</style></head><body>"""


_SCRIPT = r"""(() => {
'use strict';
function trustedURL(value, origin) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || origin === 'null' || url.origin !== origin
      || url.username || url.password || url.hash) throw Error('Contents API must use the existing page origin');
  return url.href;
}
async function boundedBody(response, limit) {
  if (Number(response.headers.get('content-length')) > limit) throw Error('Response exceeds byte limit');
  if (!response.body) throw Error('Streaming response body unavailable');
  const reader = response.body.getReader(), chunks = [];
  let size = 0;
  try {
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.length;
      if (size > limit) throw Error('Response exceeds byte limit');
      chunks.push(value);
    }
  } catch (error) {await reader.cancel().catch(() => {}); throw error;}
  finally {reader.releaseLock();}
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {bytes.set(chunk, offset); offset += chunk.length;}
  return bytes;
}
async function contentsBytes(item, config, signal, origin) {
  const url = trustedURL(item.url, origin);
  const response = await fetch(url, {credentials:'same-origin', redirect:'error', cache:'no-store', signal});
  if (!response.ok || response.redirected) throw Error('Contents API HTTP ' + response.status);
  if (response.url && new URL(response.url).origin !== origin) throw Error('Cross-origin response');
  if (!(response.headers.get('content-type') || '').includes('application/json')) throw Error('Contents API did not return JSON');
  const bytes = await boundedBody(response, Math.ceil(config.max_bytes * 4 / 3) + 65536);
  const model = JSON.parse(new TextDecoder().decode(bytes));
  if (model.type !== 'file' || model.format !== 'base64' || model.path !== item.logical_path
      || typeof model.content !== 'string') throw Error('Contents response path/type/format mismatch');
  const encoded = model.content.replace(/\s/g, '');
  if (encoded.length > 4 * Math.ceil(config.max_bytes / 3)) throw Error('Media exceeds byte limit');
  const media = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  if (!media.length || media.length > config.max_bytes) throw Error('Empty or oversized media');
  if (!globalThis.crypto?.subtle) throw Error('SHA-256 unavailable; use the existing secure Jupyter origin');
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', media)), b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== item.sha256) throw Error('Media content hash mismatch');
  return media;
}
function mount(doc, config, loc) {
  let downloading = 0;
  const active = new Set(), states = [];
  function release(state, message='Released') {
    state.epoch++;
    state.controller?.abort(); state.controller = null;
    state.cancelReady?.(); state.cancelReady = null;
    clearTimeout(state.timer); state.timer = null;
    state.video.onloadeddata = null; state.video.onerror = null;
    state.video.pause(); state.video.removeAttribute('src'); state.video.load();
    if (state.objectURL) URL.revokeObjectURL(state.objectURL);
    state.objectURL = null; active.delete(state); state.message.textContent = message;
  }
  async function load(state) {
    if (active.has(state)) return;
    if (active.size >= config.max_loaded || downloading >= config.max_concurrent) {
      state.message.textContent = 'Load limit reached; release a video and retry'; return;
    }
    active.add(state); downloading++;
    const epoch = ++state.epoch, controller = new AbortController();
    state.controller = controller; state.message.textContent = 'Loading';
    state.timer = setTimeout(() => release(state, 'Media load timed out'), config.timeout_ms);
    try {
      let src;
      if (state.item.url) {
        const bytes = await contentsBytes(state.item, config, controller.signal, loc.origin);
        if (state.epoch !== epoch) return;
        state.objectURL = URL.createObjectURL(new Blob([bytes], {type:state.item.mime}));
        src = state.objectURL;
      } else {
        if (!['file:', 'http:', 'https:'].includes(loc.protocol)) throw Error('Blob preview requires explicit Jupyter base_url + contents_root');
        src = state.item.local;
      }
      if (state.epoch !== epoch) return;
      await new Promise((resolve, reject) => {
        state.cancelReady = () => reject(Error('Media load cancelled'));
        state.video.onloadeddata = resolve;
        state.video.onerror = () => reject(Error('Media missing or decode/access failed'));
        state.video.src = src; state.video.load();
      });
      if (state.epoch !== epoch) return;
      state.cancelReady = null;
      state.video.onerror = () => {if (state.epoch === epoch) release(state, 'Media decode/access failed');};
      clearTimeout(state.timer); state.timer = null;
      state.message.textContent = state.item.url ? 'Loaded; content hash verified' : 'Loaded; generation-time hash audit only';
    } catch (error) {if (state.epoch === epoch) release(state, error.message);}
    finally {downloading--;}
  }
  for (const player of doc.querySelectorAll('[data-media]')) {
    const state = {item:config.media[Number(player.dataset.media)], video:player.querySelector('video'),
      message:player.querySelector('.message'), epoch:0, objectURL:null, timer:null, controller:null, cancelReady:null};
    states.push(state);
    player.querySelector('.load').onclick = () => load(state);
    player.querySelector('.release').onclick = () => release(state);
  }
  const releaseAll = () => states.forEach(s => release(s));
  doc.addEventListener('visibilitychange', () => {if (doc.hidden) releaseAll();});
  globalThis.addEventListener?.('pagehide', releaseAll);
  return {states, active, load, release, releaseAll, downloading:() => downloading};
}
if (typeof document === 'undefined') {
  module.exports = {trustedURL, boundedBody, contentsBytes, mount}; return;
}
mount(document, JSON.parse(document.getElementById('gallery-data').textContent), location);
})();"""
