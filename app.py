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
:root{color-scheme:dark;--ink:#eafaff;--muted:#aabaca;--cyan:#38f7fe;--pink:#ff52a5;--line:#303b50;--panel:#0e1422}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(ellipse at 85% 5%,#250f2c 0,transparent 45%),radial-gradient(ellipse at 0 35%,#082533 0,transparent 45%),#080a13;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:0 28px 35px}
header{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:17px 0;border-bottom:1px solid #305057}
.brand,.eyebrow,.badge{font:800 11px/1.4 ui-monospace,Consolas,monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan)}
.brand:before{content:"";display:inline-block;width:9px;height:9px;background:var(--cyan);transform:rotate(45deg);margin-right:10px;box-shadow:0 0 15px #38f7fe90}
.badge{color:#f3a3ca;border:1px solid #73415f;padding:5px 10px;background:#2a1329}
h1,h2,p{margin:0}h1{font-size:clamp(36px,5.5vw,69px);line-height:1;letter-spacing:-.06em}h1 em{font-style:normal;color:var(--cyan);text-shadow:0 0 21px #38f7fe46}
h2{font-size:20px;line-height:1.2;letter-spacing:-.025em}
.muted,small{color:var(--muted)}.hero{padding:25px 0 19px}.hero p{margin-top:8px}.eyebrow{margin-bottom:7px}
.card{position:relative;background:var(--panel);border:1px solid var(--line);padding:18px;min-width:0}
.card:after{position:absolute;right:-1px;top:-1px;width:12px;height:12px;content:"";border-top:2px solid #667089;border-right:2px solid #667089;pointer-events:none}
.input-card{background:linear-gradient(110deg,#101e2c,#131424);border-left:2px solid var(--cyan)}
.input-card label{font-weight:750;display:block;margin-bottom:5px}
textarea{background:#090f1b;border:1px solid #43556e;resize:vertical;color:#f0f8ff;font:inherit;width:100%;padding:10px;min-height:61px}
.actions{display:flex;align-items:center;justify-content:space-between;gap:9px;flex-wrap:wrap;margin-top:6px}.examples{display:flex;gap:6px;flex-wrap:wrap}
button{border:1px solid #4c4564;color:#e8e0f2;background:#20172c;font:700 13px/1.15 system-ui;padding:7px 10px;cursor:pointer}
.primary{background:var(--cyan);color:#08252d;border:1px solid var(--cyan);padding:9px 15px;box-shadow:0 0 18px #38f7fe26}
button:hover{border-color:var(--pink);color:#ffacd5}.primary:hover{color:#08252d;background:#9afaff;border-color:#9afaff}
:focus-visible{outline:2px solid var(--cyan);outline-offset:3px}
.note{margin:12px 0 0;padding:7px 11px;background:#251328;border-left:3px solid var(--pink);color:#ffd6ea;font-size:13px}.error{color:#ffd3cc}
.deck{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.model:first-child{border-top:3px solid var(--cyan);background:linear-gradient(140deg,#0f2430 0,var(--panel) 48%)}
.model:last-child{border-top:3px solid var(--pink);background:linear-gradient(140deg,#261627 0,var(--panel) 48%)}
.model:first-child:after{border-color:var(--cyan)}.model:last-child:after{border-color:var(--pink)}
.top{font-size:32px;line-height:1.12;font-weight:820;letter-spacing:-.045em;margin-top:7px;color:var(--cyan)}
.model:last-child .top{color:#ff7fb9}
ol{list-style:none;padding:0;margin:8px 0 0}li{display:grid;grid-template-columns:105px 1fr 42px;gap:8px;align-items:center;font-size:12px;margin:5px 0}
.track,.fill{height:6px}.track{background:#293449}.fill{background:var(--cyan);box-shadow:0 0 7px #38f7fe50}
.model:last-child .fill{background:var(--pink);box-shadow:0 0 7px #ff52a550}
footer{margin:13px 2px 0;color:var(--muted);font-size:12px}footer strong{color:#d9e0ff}
@media(max-width:650px){main{padding-inline:14px}.deck{grid-template-columns:1fr}h1{font-size:43px}.card{padding:14px}.badge{font-size:9px;letter-spacing:.08em}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
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
