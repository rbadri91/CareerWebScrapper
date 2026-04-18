"""
Send the job search report via Resend (https://resend.com).

Setup:
  1. Sign up free at resend.com — 100 emails/day, 3 000/month on free tier.
  2. Go to API Keys → Create API Key.
  3. Set RESEND_API_KEY=re_... in .env
  4. Set REPORT_EMAIL=<recipient address> in .env
     (defaults to LINKEDIN_EMAIL if not set)

The report is sent as an HTML email with the Markdown file attached.
"""

import asyncio
import logging
import os
import re
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

RESEND_SEND_URL = "https://api.resend.com/emails"
RESEND_FROM = "Job Search Bot <onboarding@resend.dev>"   # use resend default sender on free plan


# ──────────────────────────────────────────────
# Minimal Markdown → HTML
# ──────────────────────────────────────────────

def _md_to_html(md: str) -> str:
    """Convert the report Markdown to simple inline-styled HTML."""
    lines = md.splitlines()
    html: list[str] = []
    in_table = False

    for line in lines:
        # HR
        if re.match(r"^-{3,}$", line.strip()):
            if in_table:
                html.append("</table>")
                in_table = False
            html.append("<hr style='border:1px solid #ddd;'>")
            continue

        # Table rows
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells if c):
                continue  # separator row
            if not in_table:
                html.append(
                    '<table border="1" cellpadding="5" cellspacing="0" '
                    'style="border-collapse:collapse;font-size:12px;margin:6px 0;">'
                )
                in_table = True
            tag = "th" if "<table" in (html[-1] if html else "") else "td"
            row = "".join(f"<{tag} style='padding:4px 8px;'>{c}</{tag}>" for c in cells)
            html.append(f"<tr>{row}</tr>")
            continue
        else:
            if in_table:
                html.append("</table>")
                in_table = False

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            lvl = len(m.group(1))
            colours = {1: "#1a73e8", 2: "#333", 3: "#555", 4: "#666"}
            html.append(
                f'<h{lvl} style="color:{colours.get(lvl,"#333")};'
                f'margin:12px 0 4px;">{m.group(2)}</h{lvl}>'
            )
            continue

        # Blockquote (outreach messages)
        if line.startswith("> "):
            content = _inline(line[2:])
            html.append(
                f'<blockquote style="border-left:3px solid #1a73e8;'
                f'padding:6px 12px;background:#f8f9ff;color:#333;margin:4px 0;">'
                f'{content}</blockquote>'
            )
            continue

        # Bold "**...**" metadata lines (Role:, LinkedIn:, etc.)
        converted = _inline(line)
        if converted.strip():
            html.append(f'<p style="margin:2px 0;">{converted}</p>')
        else:
            html.append("<br>")

    if in_table:
        html.append("</table>")

    body = "\n".join(html)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;font-size:13px;color:#333;
             max-width:960px;margin:auto;padding:24px;">
{body}
</body></html>"""


def _inline(text: str) -> str:
    """Apply inline Markdown transformations (bold, code, links)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


# ──────────────────────────────────────────────
# Resend API call
# ──────────────────────────────────────────────

async def _send_via_resend(
    api_key: str,
    to: str,
    subject: str,
    html: str,
    md_path: Path,
) -> bool:
    """POST to Resend /emails. Returns True on success."""
    import base64
    attachment_b64 = base64.b64encode(md_path.read_bytes()).decode()

    payload = {
        "from": RESEND_FROM,
        "to": [to],
        "subject": subject,
        "html": html,
        "attachments": [
            {
                "filename": md_path.name,
                "content": attachment_b64,
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            RESEND_SEND_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            body = await resp.json(content_type=None)
            if resp.status in (200, 201):
                logger.info("Report emailed to %s (id=%s)", to, body.get("id", "?"))
                return True
            else:
                logger.error(
                    "Resend error %d: %s", resp.status, body.get("message", body)
                )
                return False


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def send_report(report_path: str) -> bool:
    """
    Email the report at report_path via Resend.
    Returns True on success, False on any error (non-fatal — won't crash the pipeline).
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        logger.info("RESEND_API_KEY not set — skipping email")
        return False

    recipient = os.getenv("REPORT_EMAIL") or os.getenv("LINKEDIN_EMAIL", "")
    if not recipient:
        logger.warning("No recipient address — set REPORT_EMAIL in .env")
        return False

    path = Path(report_path)
    if not path.exists():
        logger.error("Report file not found: %s", report_path)
        return False

    md_content = path.read_text(encoding="utf-8")
    html_content = _md_to_html(md_content)
    subject = f"Job Search Report — {path.stem.replace('job_search_', '').replace('_', ' ')}"

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    try:
        if running:
            # Called from inside asyncio.run() — spin up a new thread with its own loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    _send_via_resend(api_key, recipient, subject, html_content, path),
                )
                return future.result(timeout=30)
        else:
            return asyncio.run(
                _send_via_resend(api_key, recipient, subject, html_content, path)
            )
    except Exception as exc:
        logger.error("Failed to send email: %s", exc)
        return False
