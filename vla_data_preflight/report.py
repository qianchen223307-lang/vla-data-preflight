import html
import json
from pathlib import Path


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ("check_id", "severity", "path", "message", "hint")
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(d[c]))}</td>" for c in columns) + "</tr>" for d in report["diagnostics"])
    content = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>VLA preflight</title><style>body{font:16px system-ui;margin:2rem;max-width:1200px;color:#233c54}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd5df;padding:.7rem;vertical-align:top;text-align:left}pre{white-space:pre-wrap}</style><h1>VLA data preflight</h1>"
    content += "<p>" + html.escape(report["data_kind"]) + " · LIBERO/QwenOFT · no model loading</p><pre>" + html.escape(json.dumps(report["summary"], indent=2)) + "</pre><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead><tbody>" + (rows or "<tr><td colspan='5'>No diagnostics.</td></tr>") + "</tbody></table><p>" + " ".join(html.escape(s) for s in report["limitations"]) + "</p></html>"
    (output / "report.html").write_text(content, encoding="utf-8")
