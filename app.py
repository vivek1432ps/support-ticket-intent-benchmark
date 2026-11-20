"""A local-only GUI for comparing the benchmark’s trained ticket classifiers."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch

from experiment import LABELS, MAX_LEN, TOKEN, encode, fit_models

LIMIT = 480
EXAMPLES = {
    "Refund": "Regarding alder, my main request is to get a refund.",
    "Account": "What are the steps to replace the password? Reference elm.",
    "Billing": "Following up on cedar. Is it possible to question a charge?",
    "Delivery": "Regarding orchid, my main request is to trace my parcel.",
    "Technical": "What are the steps to diagnose a crash? Reference alder.",
}
DEFAULT = EXAMPLES["Refund"]
HUMAN = {"account": "Account access", "billing": "Billing", "delivery": "Delivery",
         "refund": "Refund", "technical": "Technical issue"}
REPORT = Path(__file__).resolve().parent / "examples" / "report.json"

CSS = """
:root{color-scheme:dark;--c:#ebeeff;--muted:#aab4d1;--accent:#89e6ff;--iris:#a8aaff;--edge:#354369}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(ellipse at 85% 0%,#1f2853,transparent 52%),#11172b;color:var(--c);font:15px/1.5 system-ui,sans-serif}
main{max-width:1130px;margin:auto;padding:0 26px 35px} header{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:18px 0;border-bottom:1px solid var(--edge)}
.brand,.eyebrow,.badge{font-weight:800;font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--accent)}.badge{color:#d1d9ff;border:1px solid var(--edge);padding:5px 10px;border-radius:30px}
h1,h2,p{margin:0}h1{font-size:clamp(36px,5.5vw,65px);line-height:1;letter-spacing:-.06em}h1 em{font-style:normal;color:var(--accent)}
h2{font-size:21px;line-height:1.2;letter-spacing:-.025em}.muted,small{color:var(--muted)}.hero{padding:25px 0 19px}.hero p{margin-top:7px}.eyebrow{margin-bottom:6px}
.card{background:#1b2442;border:1px solid var(--edge);border-radius:16px;padding:18px;min-width:0}
.input-card label{font-weight:750;display:block;margin-bottom:5px} textarea{background:#101832;border:1px solid #4a5a83;border-radius:8px;resize:vertical;color:#f0f4ff;font:inherit;width:100%;padding:10px;min-height:61px}
.actions{display:flex;align-items:center;justify-content:space-between;gap:9px;flex-wrap:wrap;margin-top:6px}.examples{display:flex;gap:6px;flex-wrap:wrap}
button{border-radius:8px;border:1px solid #4d5c85;color:#e9edff;background:#253153;font:700 13px/1.15 system-ui;padding:7px 9px;cursor:pointer}
.primary{background:#89e6ff;color:#102640;border:0;padding:9px 15px}button:hover{filter:brightness(1.16)}:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.note{margin:12px 0 0;padding:6px 11px;background:#303052;border:1px solid #5b5689;border-radius:8px;color:#ebdeff;font-size:13px}.error{color:#ffd1cf}
.deck{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.model:first-child{border-top:3px solid var(--accent)}.model:last-child{border-top:3px solid var(--iris)}
.top{font-size:32px;line-height:1.12;font-weight:820;letter-spacing:-.045em;margin-top:6px;color:#b6ecff}.model:last-child .top{color:#c4c6ff}
ol{list-style:none;padding:0;margin:8px 0 0}li{display:grid;grid-template-columns:105px 1fr 42px;gap:8px;align-items:center;font-size:12px;margin:4px 0}
.track,.fill{height:6px;border-radius:8px}.track{background:#344062}.fill{background:var(--accent)}.model:last-child .fill{background:var(--iris)}
footer{margin:11px 2px 0;color:var(--muted);font-size:12px}footer strong{color:#d9e0ff}
@media(max-width:650px){main{padding-inline:14px}.deck{grid-template-columns:1fr}h1{font-size:43px}.card{padding:14px}.badge{font-size:9px}}
"""


def classify(bundle, message):
    baseline, transformer, vocab = bundle
    tokens = TOKEN.findall(message.lower())
    if not message.strip() or not tokens:
        raise ValueError("Enter a short English ticket with at least one letter or number.")
    if len(message) > LIMIT:
        raise ValueError(f"Keep the ticket to {LIMIT} characters or fewer.")
    with torch.inference_mode():
        transformer_scores = transformer(encode([message], vocab)).softmax(-1)[0].numpy()
    baseline_scores = baseline.predict_proba([message])[0]
    return baseline_scores, transformer_scores, len(tokens), sum(t not in vocab for t in tokens[:MAX_LEN])


def render(bundle, report, message, error=None, scores=None):
    if scores is None and error is None:
        scores = classify(bundle, message)
    if error:
        note = f'<p class="note error" role="alert">{escape(error)}</p>'
        cards = ""
    else:
        base, tiny, length, unknown = scores
        b,t = int(base.argmax()),int(tiny.argmax())
        text = f"Models agree: {HUMAN[LABELS[b]]}." if b == t else f"Models disagree: {HUMAN[LABELS[b]]} vs. {HUMAN[LABELS[t]]}."
        extra = f" The Transformer only reads the first {MAX_LEN} tokens." if length > MAX_LEN else ""
        note = f'<p class="note"><strong>{text}</strong> Transformer unknown tokens: {unknown}/{min(length,MAX_LEN)}.{extra}</p>'
        cards = '<section class="deck" aria-label="Model predictions">' + panel("TF-IDF + logistic regression",base,report["tfidf_logistic_regression"]) + panel("Tiny Transformer",tiny,report["tiny_transformer"]) + '</section>'
    buttons = "".join(f'<button name="example" value="{key}">{key}</button>' for key in EXAMPLES)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ticket Lab | Compare intent models</title><style>{CSS}</style><body><main>
<header><span class="brand">Ticket Lab / NLP benchmark</span><span class="badge">Local · Synthetic English data</span></header>
<section class="hero"><div class="eyebrow">Side-by-side inference</div><h1>One ticket. <em>Two models.</em></h1>
<p class="muted">Watch a classical baseline and a tiny Transformer read the same message.</p></section>
<section class="card input-card"><form method="post" action="/">
<label for="message">Support ticket text</label><textarea id="message" name="message" maxlength="{LIMIT}" rows="2" spellcheck="false">{escape(message)}</textarea>
<div class="actions"><button class="primary" type="submit">Compare predictions →</button><div class="examples" aria-label="Load built-in examples">{buttons}</div></div>
</form></section>{note}{cards}
<footer><strong>Benchmark context.</strong> 600 synthetic training tickets · 120 held-out synthetic tickets · 5 classes. Model scores are not calibrated confidence. Not evaluated on real customer messages. Demo runs locally and does not save text. AI-assisted portfolio prototype.</footer>
</main></body></html>'''


def panel(name, scores, metrics):
    order = scores.argsort()[::-1]
    bars = "".join(f'<li><span>{HUMAN[LABELS[i]]}</span><div class="track"><div class="fill" style="width:{scores[i]*100:.3f}%"></div></div><b>{scores[i]:.0%}</b></li>' for i in order)
    return f'''<article class="card model"><h2>{name}</h2><small>Held-out accuracy <b>{metrics["accuracy"]:.1%}</b> · Macro-F1 <b>{metrics["macro_f1"]:.3f}</b></small>
<p class="top">{HUMAN[LABELS[order[0]]]}</p><ol aria-label="Ranked class scores">{bars}</ol></article>'''


class Handler(BaseHTTPRequestHandler):
    # Deliberately don't log request lines or ticket content.
    def log_message(self, *_):
        return

    def send_page(self, status, page):
        body = page.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def valid_path(self):
        return urlsplit(self.path).path == "/"

    def do_GET(self):
        if not self.valid_path():
            self.send_error(404)
            return
        self.send_page(200,render(self.server.bundle,self.server.report,DEFAULT))

    def do_POST(self):
        if not self.valid_path():
            self.send_error(404)
            return
        message = ""
        try:
            n = int(self.headers.get("Content-Length",0))
            if not 0 < n <= 7000 or self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.close_connection = True
                raise ValueError("Submit a short ticket using the form.")
            form = parse_qs(self.rfile.read(n).decode("utf-8"), keep_blank_values=True, max_num_fields=4)
            message = form.get("message",[""])[0]
            example = form.get("example",[None])[0]
            if example in EXAMPLES:
                message = EXAMPLES[example]
            scores = classify(self.server.bundle, message)
        except (UnicodeError, ValueError) as exc:
            safe_errors = {"Enter a short English ticket with at least one letter or number.",
                           f"Keep the ticket to {LIMIT} characters or fewer."}
            error = str(exc) if str(exc) in safe_errors else "Submit a short ticket using the form."
            self.send_page(400,render(self.server.bundle,self.server.report,message[:LIMIT],error=error))
            return
        self.send_page(200,render(self.server.bundle,self.server.report,message,scores=scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    print("Training both models locally from synthetic examples…", flush=True)
    bundle = fit_models(report["epochs"], report["seed"])
    with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
        server.bundle,server.report = bundle,report
        print(f"Open http://127.0.0.1:{args.port}",flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
