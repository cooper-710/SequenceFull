from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

def render_html_to_pdf(template_path: str, ctx: Dict[str, Any], out_pdf: Path) -> None:
    tpl = Path(template_path).resolve()
    env = Environment(loader=FileSystemLoader(str(tpl.parent)),
                      autoescape=select_autoescape(["html"]))
    template = env.get_template(tpl.name)
    html = template.render(**_safe_ctx(ctx))

    out_pdf = Path(out_pdf).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_html = out_pdf.with_suffix(".html")

    project_root = out_pdf.parents[2]
    base_tag = f'<base href="{project_root.as_uri()}/">'
    if "<head>" in html:
        html = html.replace("<head>", f"<head>{base_tag}", 1)
    else:
        html = base_tag + html

    out_html.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(out_html.as_uri(), wait_until="load")
        page.pdf(path=str(out_pdf), print_background=True, prefer_css_page_size=True)
        browser.close()


def _safe_ctx(ctx):
    return ctx if isinstance(ctx, dict) else {}
