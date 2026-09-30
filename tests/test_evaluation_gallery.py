import json
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock

from visual_rl.evaluation import (
    EvaluationManifest,
    JupyterContents,
    SampleRecord,
    audit_manifest,
    render_gallery,
)
from visual_rl.evaluation.gallery import _SCRIPT
from visual_rl.rewards import FixedNormalization, weighted_score


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = "media #?%25.mp4"
        (self.root / self.path).write_bytes(b"fixture")
        scores = weighted_score(
            {"quality": 4},
            {"quality": 2},
            space="fixed_normalized",
            normalization={"quality": FixedNormalization(1, 3, "toy")},
        )
        self.sample = SampleRecord.from_file(
            media_root=self.root,
            sample_id="id-1",
            prompt="A test prompt",
            seed=1,
            optimizer_step=5,
            framecount=8,
            fps=4,
            path=self.path,
            scores=scores,
        )
        self.manifest = EvaluationManifest(5, ("id-1",), (self.sample,))
        self.audit = audit_manifest(self.manifest, media_root=self.root)

    def render(self, **kwargs):
        return render_gallery(self.manifest, audit=self.audit, **kwargs)

    def payload(self, html):
        return json.loads(
            re.search(
                r'<script type="application/json" id="gallery-data">(.*?)</script>', html, re.S
            )[1]
        )

    def test_relative_media_pure_render_no_binary_cdn_or_eager_src(self):
        with (
            mock.patch("builtins.open", side_effect=AssertionError("I/O")),
            mock.patch("os.stat", side_effect=AssertionError("I/O")),
        ):
            html = self.render(gallery_path="gallery/index.html")
        data = self.payload(html)
        self.assertEqual(data["media"][0]["local"], "./../media%20%23%3F%2525.mp4")
        self.assertIsNone(data["media"][0]["url"])
        self.assertNotIn("<script src=", html)
        self.assertNotIn("data:video", html)
        self.assertNotRegex(html, r"<video[^>]+\bsrc=")
        self.assertIn("Complete: 1/1", html)
        self.assertIn("Fixed-normalized scores", html)
        self.assertIn("2 (fixed_normalized)", html)

    def test_html_and_script_json_escaping(self):
        attack = '</script><img src=x onerror="globalThis.pwned=1">&\u2028\u2029'
        self.sample = replace(self.sample, prompt=attack, sample_id=attack)
        self.manifest = EvaluationManifest(5, (attack,), (self.sample,))
        self.audit = audit_manifest(self.manifest, media_root=self.root)
        html = self.render()
        self.assertNotIn(attack, html)
        self.assertEqual(html.count("</script>"), 2)
        self.assertEqual(self.payload(html)["media"][0]["sample_id"], attack)
        self.assertIn("\\u003c", html)
        self.assertIn("\\u2028", html)

        class Tags(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)
                for name, value in attrs:
                    if name.startswith("on"):
                        raise AssertionError("injected event handler")

        parser = Tags()
        parser.feed(html)
        self.assertNotIn("img", parser.tags)

    def test_missing_entries_visible_and_mismatched_audit_rejected(self):
        self.manifest = replace(
            self.manifest,
            expected_sample_ids=("id-1", "absent"),
            samples=(replace(self.sample, scores=None),),
        )
        (self.root / self.path).unlink()
        self.audit = audit_manifest(self.manifest, media_root=self.root)
        html = self.render()
        for text in (
            "Incomplete: 0/2",
            "missing_file",
            "missing_record",
            "missing_scores",
            "absent",
        ):
            self.assertIn(text, html)
        self.assertEqual(self.payload(html)["media"], [])
        with self.assertRaises(ValueError):
            render_gallery(self.manifest, audit=replace(self.audit, manifest_sha256="0" * 64))

    def test_explicit_mapping_special_paths_and_invalid_configuration(self):
        adapter = JupyterContents(base_url="https://host/user/person/", contents_root="shared/run")
        html = self.render(jupyter=adapter)
        item = self.payload(html)["media"][0]
        self.assertEqual(item["logical_path"], "shared/run/" + self.path)
        self.assertEqual(
            item["url"],
            "https://host/user/person/api/contents/shared/run/media%20%23%3F%2525.mp4?content=1&type=file&format=base64",
        )
        self.assertNotIn("document.referrer", html)
        with self.assertRaises(TypeError):
            JupyterContents(base_url="https://host/")
        for base in [
            "file:///tmp",
            "https://a:b@host/",
            "https://host/?token=x",
            "https://host/#",
            "//host/",
            "https://host/a/../b",
            "https://host/%2e%2e/",
            "https://host//",
            "https://host/a\\b",
            "https://host:99999/",
            "https://ho\nst/",
        ]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                JupyterContents(base, "shared")
        for root in ["/abs", "../escape", "a//b", "https://host/x"]:
            with self.assertRaises(ValueError):
                JupyterContents("https://host/", root)
        self.assertEqual(JupyterContents("https://host", "").logical_path("v.mp4"), "v.mp4")
        for kwargs in [
            dict(gallery_path="../x"),
            dict(max_concurrent=5),
            dict(max_loaded=0),
            dict(max_media_bytes=2**40),
            dict(timeout_ms=True),
            dict(jupyter={}),
        ]:
            with self.assertRaises(ValueError):
                self.render(**kwargs)


@unittest.skipUnless(shutil.which("node"), "Node required for gallery lifecycle regressions")
class JavaScriptTests(unittest.TestCase):
    def js(self, code):
        script = (
            _SCRIPT
            + '\nconst h = module.exports; const assert = require("node:assert/strict");\n'
            + _HARNESS
        )
        result = subprocess.run(
            [shutil.which("node"), "-"],
            input=script
            + "\n(async()=>{\n"
            + code
            + "\n})().catch(e=>{console.error(e);process.exitCode=1;});",
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_origin_bytes_hash_and_response_validation(self):
        self.js(r"""
assert.equal(h.trustedURL(item.url, origin), item.url);
for (const url of ['https://evil.test/x', 'file:///tmp/x', 'https://user:pass@host/api', 'https://host/api#x'])
  assert.throws(() => h.trustedURL(url, origin));
assert.throws(() => h.trustedURL(item.url, 'null'));
globalThis.fetch = async (url, options) => {
  assert.equal(options.credentials, 'same-origin'); assert.equal(options.redirect, 'error');
  return response(model);
};
assert.equal(Buffer.from(await h.contentsBytes(item, cfg, new AbortController().signal, origin)).toString(), 'video');
for (const change of [{path:'wrong.mp4'}, {format:'text'}, {type:'directory'}, {content:'eA=='}]) {
  globalThis.fetch = async () => response({...model, ...change});
  await assert.rejects(h.contentsBytes(item, cfg, new AbortController().signal, origin));
}
globalThis.fetch = async () => new Response('login', {headers:{'content-type':'text/html'}});
await assert.rejects(h.contentsBytes(item, cfg, new AbortController().signal, origin), /JSON/);
globalThis.fetch = async () => response(model);
await assert.rejects(h.contentsBytes(item, {...cfg,max_bytes:1}, new AbortController().signal, origin), /limit|oversized/);
""")

    def test_response_body_is_byte_bounded_even_without_content_length(self):
        self.js(r"""
const r = new Response(new ReadableStream({start(c) {c.enqueue(new Uint8Array(11)); c.close();}}));
await assert.rejects(h.boundedBody(r, 10), /byte limit/);
await assert.rejects(h.boundedBody(new Response('x', {headers:{'content-length':'20'}}), 10), /byte limit/);
assert.equal((await h.boundedBody(new Response('ok'), 10)).length, 2);
""")

    def test_base64_s_and_wrapped_whitespace_are_preserved(self):
        self.js(r"""
const bytes = Buffer.from([0xb2, 0xff, 0x00, 0xb2]);
const sha = require('node:crypto').createHash('sha256').update(bytes).digest('hex');
const encoded = bytes.toString('base64');
assert.ok(encoded.includes('s'));
for (const content of [encoded, ' \n'+encoded.slice(0,2)+'\r\n'+encoded.slice(2)+'\t']) {
  globalThis.fetch = async () => response({...model, content});
  const decoded = await h.contentsBytes({...item, sha256:sha}, cfg, new AbortController().signal, origin);
  assert.deepEqual(Buffer.from(decoded), bytes);
}
""")

    def test_on_demand_concurrency_release_and_preview_blob(self):
        self.js(r"""
let calls = 0, revoked = [], created = 0;
globalThis.fetch = async () => {calls++; return response(model);};
URL.createObjectURL = () => 'blob:fixture-' + (++created);
URL.revokeObjectURL = url => revoked.push(url);
const doc = fakeDocument(3), app = h.mount(doc, {...cfg,media:[item,item,item]}, {origin,protocol:'blob:'});
assert.equal(calls, 0);
await app.load(app.states[0]);
assert.equal(calls, 1); assert.match(app.states[0].message.textContent, /hash verified/);
assert.equal(app.active.size, 1);
app.release(app.states[0]); assert.deepEqual(revoked, ['blob:fixture-1']);
assert.equal(app.states[0].video.src, undefined);
await app.load(app.states[1]); await app.load(app.states[2]);
await app.load(app.states[0]); assert.equal(calls, 3); // loaded cap = 2
assert.match(app.states[0].message.textContent, /limit/);
doc.hidden = true; doc.listeners.visibilitychange(); assert.equal(app.active.size, 0);
assert.equal(revoked.length, 3);
const local = h.mount(fakeDocument(1), {...cfg,media:[{...item,url:null,local:'./x.mp4'}]}, {origin,protocol:'blob:'});
await local.load(local.states[0]); assert.match(local.states[0].message.textContent, /explicit Jupyter/);
assert.equal(local.active.size, 0);
""")

    def test_cancelled_late_fetch_never_resurrects_player_or_leaks_url(self):
        self.js(r"""
let finish, created = 0;
globalThis.fetch = () => new Promise(resolve => {finish = () => resolve(response(model));});
URL.createObjectURL = () => {created++; return 'blob:late';};
const app = h.mount(fakeDocument(2), {...cfg,media:[item,item],max_concurrent:1}, {origin,protocol:'blob:'});
const pending = app.load(app.states[0]);
assert.equal(app.downloading(), 1);
await app.load(app.states[1]); assert.match(app.states[1].message.textContent, /limit/);
app.release(app.states[0]);
await app.load(app.states[1]); assert.match(app.states[1].message.textContent, /limit/);
finish(); await pending;
assert.equal(created, 0); assert.equal(app.downloading(), 0); assert.equal(app.active.size, 0);
""")

    def test_local_readiness_is_concurrency_bounded_timeout_and_errors_visible(self):
        self.js(r"""
const local = {...item,url:null,local:'./media/x.mp4'};
const app = h.mount(fakeDocument(2, false), {...cfg,media:[local,local],max_concurrent:1,timeout_ms:20}, {origin,protocol:'file:'});
const pending = app.load(app.states[0]);
await app.load(app.states[1]); assert.match(app.states[1].message.textContent, /limit/);
await pending; assert.match(app.states[0].message.textContent, /timed out/);
assert.equal(app.active.size, 0); assert.equal(app.downloading(), 0);
const ready = h.mount(fakeDocument(1), {...cfg,media:[local]}, {origin,protocol:'file:'});
await ready.load(ready.states[0]); ready.states[0].video.onerror();
assert.match(ready.states[0].message.textContent, /failed/); assert.equal(ready.active.size, 0);
""")


_HARNESS = r"""
const origin = 'https://host';
const model = {type:'file',format:'base64',path:'shared/x.mp4',content:Buffer.from('video').toString('base64')};
const item = {url:origin+'/api/contents/shared/x.mp4?content=1&type=file&format=base64',logical_path:'shared/x.mp4',
  mime:'video/mp4',sha256:require('node:crypto').createHash('sha256').update('video').digest('hex')};
const cfg = {max_bytes:100,max_concurrent:2,max_loaded:2,timeout_ms:1000};
function response(data) {return new Response(JSON.stringify(data), {headers:{'content-type':'application/json'}});}
function fakeDocument(count, ready=true) {
  const players = Array.from({length:count}, (_, index) => {
    const elements = {
      video:{pause(){},removeAttribute(k){delete this[k];},load(){if (ready && this.src) queueMicrotask(()=>this.onloadeddata?.());}},
      '.message':{textContent:''},'.load':{},'.release':{}
    };
    return {dataset:{media:String(index)},querySelector:s=>elements[s]};
  });
  return {listeners:{},hidden:false,querySelectorAll:()=>players,addEventListener(name,fn){this.listeners[name]=fn;}};
}
"""


if __name__ == "__main__":
    unittest.main()
