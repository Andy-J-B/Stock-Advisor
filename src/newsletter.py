# ---------------------------------------------------------------------------
# Email newsletter — daily stock & portfolio update
# ---------------------------------------------------------------------------

"""Professional daily email newsletter for the single Advisor user.

Builds an HTML digest from the nightly brief (scores, market overview,
enrichment hints) plus live portfolio stats and fundamentals, then sends it
over SMTP.  Designed to run right after ``brief --persist`` in CI, or as a
standalone ``python main.py newsletter``.

Sending uses stdlib ``smtplib`` only.  If no SMTP is configured the rendered
HTML is written to ``data/newsletter_<date>.html`` instead, so the layout can
be previewed before credentials are added.
"""

from __future__ import annotations

import html as html_escape
import os
import smtplib
import ssl
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate

from src import brief, data_client, portfolio

DEFAULT_RECIPIENT = "080.abae@gmail.com"

# Short-form fundamentals pulled from the yfinance info dict.
_FUND_KEYS: list[tuple[str, str]] = [
    ("Market Cap", "formatted_market_cap"),
    ("P/E (trailing)", "trailingPE"),
    ("P/E (forward)", "forwardPE"),
    ("Price/Book", "priceToBook"),
    ("Div Yield", "dividendYield"),
    ("52wk High", "fiftyTwoWeekHigh"),
    ("52wk Low", "fiftyTwoWeekLow"),
    ("Target Price", "targetMeanPrice"),
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def recipients() -> list[str]:
    raw = _env("EMAIL_TO", DEFAULT_RECIPIENT)
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def smtp_config() -> dict:
    return {
        "host": _env("SMTP_HOST"),
        "port": int(_env("SMTP_PORT", "587") or "587"),
        "user": _env("SMTP_USER"),
        "password": _env("SMTP_PASSWORD"),
        "from_addr": _env("EMAIL_FROM") or _env("SMTP_USER"),
        "from_name": _env("EMAIL_FROM_NAME", "Stock Advisor"),
    }


def is_smtp_configured() -> bool:
    cfg = smtp_config()
    return bool(cfg["host"] and cfg["user"] and cfg["password"])


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _fundamentals(tickers: list[str]) -> dict[str, dict]:
    """Best-effort fundamentals keyed by ticker, '' values when unavailable."""
    out: dict[str, dict] = {}
    for t in dict.fromkeys(tickers):
        info = data_client.get_ticker_info(t)
        cap = info.get("marketCap")
        div = info.get("dividendYield")
        out[t] = {
            "Market Cap": (
                f"{cap / 1e9:.2f}B" if isinstance(cap, (int, float)) and cap else "N/A"
            ),
            "P/E (trailing)": _fmt_num(info.get("trailingPE")),
            "P/E (forward)": _fmt_num(info.get("forwardPE")),
            "Price/Book": _fmt_num(info.get("priceToBook")),
            # yfinance reports garbage dividend yields for CDR/ETF tickers
            # (e.g. 71%) — clamp to a plausible bound.
            "Div Yield": (
                f"{div * 100:.2f}%"
                if isinstance(div, (int, float)) and 0 < div <= 0.25
                else "N/A"
            ),
            "52wk High": _fmt_price(info.get("fiftyTwoWeekHigh")),
            "52wk Low": _fmt_price(info.get("fiftyTwoWeekLow")),
            "Target Price": _fmt_price(info.get("targetMeanPrice")),
        }
    return out


def _fmt_num(v) -> str:
    if isinstance(v, (int, float)) and v not in (None, 0, 0.0):
        return f"{float(v):.2f}"
    return "N/A"


def _fmt_price(v) -> str:
    if isinstance(v, (int, float)) and v:
        return f"${float(v):,.2f}"
    return "N/A"


def portfolio_snapshot() -> dict:
    """Portfolio summary (CAD), falling back to the remote Supabase snapshot
    when the local database has no holdings (e.g. inside CI)."""
    snap = portfolio.snapshot()
    if not snap["rows"] and not snap["cash"] and not snap["invested"]:
        remote = brief.fetch_remote_portfolio_snapshot()
        if remote:
            snap = remote
    return snap


def _classify(run: brief.BriefRun) -> dict:
    """Bucket each score by conviction for the email sections."""
    buys, watch, alerts, invested = [], [], [], []
    holdings = {s.ticker for s in run.scores if s.portfolio_weight and s.portfolio_weight > 0}
    for s in run.scores:
        entry = {
            "ticker": s.ticker,
            "composite": s.composite,
            "components": (s.sentiment, s.technical, s.ml_pred, s.analyst),
            "headline": s.top_headline,
            "analyst_breakdown": s.analyst_breakdown,
            "signal": s.signal_agreement,
            "anomaly": s.anomaly_flag,
            "anomaly_detail": s.anomaly_detail,
            "recommendation": s.recommendation,
            "price": s.price,
            "day_pct": s.day_change_pct,
            "weight": s.portfolio_weight or 0.0,
            "delta": s.score_delta,
            "rank_change": s.rank_change,
        }
        if entry["composite"] >= 25:
            buys.append(entry)
        elif entry["composite"] <= -25:
            alerts.append(entry)
        else:
            watch.append(entry)
        if entry["ticker"] in holdings:
            invested.append(entry)
    return {"buys": buys, "watch": watch, "alerts": alerts, "invested": invested}


# ---------------------------------------------------------------------------
# Rendering (email-safe inline CSS)
# ---------------------------------------------------------------------------

_STYLE = """
    body { margin: 0; padding: 0; width: 100%;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Helvetica, Arial, sans-serif; -webkit-text-size-adjust: none; }
    table, tr, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt;
      border-collapse: collapse; }
    .bd { background: #F3F4F6; }
    .card { background: #FFFFFF; }
    .di { color: #0A0B0D !important; }
    .ds { color: #5B616E !important; }
    .dm { color: #9AA0A6 !important; }
    .pos { color: #0A7A3D !important; }
    .neg { color: #C3312B !important; }
    .thead td { background: #F7F8FA; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .6px; color: #5B616E;
      border-top: 1px solid #E6E8EB; border-bottom: 1px solid #E6E8EB; }
    a { color: #0052FF; }
    .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 700; white-space: nowrap; }
    .pill.buy { background: #E7F5EC; color: #0A7A3D; }
    .pill.watch { background: #FFF4DA; color: #9C6B00; }
    .pill.alert { background: #FCEBEA; color: #C3312B; }
    .pill.muted { background: #F1F2F4; color: #5B616E; }
    .news li { margin-bottom: 8px; font-size: 13px; }
    .num { text-align: right; }
    @media screen and (max-width: 640px) {
      .card { width: 100% !important; }
      .bd { padding: 12px 8px !important; }
      .sec-hide { display: none !important; }
    }
    @media (prefers-color-scheme: dark) {
      .bd { background: #121318 !important; }
      .card { background: #202124 !important; border-color: #3C4043 !important; }
      .di { color: #FFFFFF !important; }
      .ds { color: #B3B9C4 !important; }
      .dm { color: #7C828E !important; }
      .sec { border-color: #3C4043 !important; }
      .kpidv { border-left-color: #3C4043 !important; }
      .thead td { background: #2A2D33 !important; color: #B3B9C4 !important;
        border-color: #3C4043 !important; }
      .tbl td { border-bottom-color: #3C4043 !important; }
    }
"""


def _chg(v: float, pct: bool = False, digits: int = 2, prefix: str = "") -> str:
    cls = "pos" if v >= 0 else "neg"
    body = f"{'+' if v >= 0 else '−'}{abs(v):.{digits}f}"
    if pct:
        body += "%"
    return f'<span class="{cls}">{prefix}{body}</span>'


def _money(v: float, suffix: str = "") -> str:
    return f"${v:,.2f}{suffix}"


def _plain(v) -> str:
    if v is None or (isinstance(v, float) and v == 0.0):
        return "—"
    return str(v)


def _ticker_cell(t: str) -> str:
    return f"<b>{html_escape.escape(t)}</b>"


def _rec_pill(rec) -> str:
    """Color-coded recommendation pill: buy / watch / alert / neutral."""
    r = (rec or "").lower()
    if any(k in r for k in ("add", "buy", "accumulate")):
        kind = "buy"
    elif any(k in r for k in ("avoid", "reduce", "trim", "sell")):
        kind = "alert"
    elif any(k in r for k in ("hold", "watch", "over")):
        kind = "watch"
    else:
        kind = "muted"
    return f'<span class="pill {kind}">{html_escape.escape(str(rec or "—"))}</span>'


def _thead(cells, right: set | None = None) -> str:
    """Gray uppercase table-header row."""
    right = right or set()
    out = ['<tr>']
    for i, c in enumerate(cells):
        style = "padding:9px 12px;"
        if i in right:
            style += "text-align:right;"
        out.append(f'<td class="thead" style="{style}">{c}</td>')
    out.append('</tr>')
    return "".join(out)


def _trow(cells_html, last: bool = False, right: set | None = None) -> str:
    right = right or set()
    out = ["<tr>"]
    for i, c in enumerate(cells_html):
        style = "padding:10px 12px;font-size:13px;"
        if i in right:
            style += "text-align:right;"
        if not last:
            style += "border-bottom:1px solid #F1F2F4;"
        out.append(f'<td class="tbl" style="{style}">{c}</td>')
    out.append("</tr>")
    return "".join(out)


def _exec_summary(run: brief.BriefRun, mkt: dict, snap: dict) -> str:
    """AI executive summary when Gemini is available, else a rules-based one."""
    top = max(run.scores, key=lambda s: s.composite) if run.scores else None
    worst = min(run.scores, key=lambda s: s.composite) if run.scores else None
    idx = [
        f"{name} {d['chg_pct']:+.2f}%"
        for name, d in mkt.get("indices", {}).items()
    ]
    prompt = (
        "You are a concise financial editor. Write a 3-4 sentence executive "
        "summary for a daily investor newsletter. Use these facts only and do "
        "not add new claims:\n"
        + f"- Indices: {', '.join(idx) or 'n/a'}\n"
        + f"- Portfolio net worth (CAD): ${snap['net_worth']:,.2f}, "
        f"today {snap['day_chg']:+,.2f} ({snap['day_pct']:+.2f}%)\n"
        + (f"- Top conviction: {top.ticker} at {top.composite:+.1f}; "
           f"lowest: {worst.ticker} at {worst.composite:+.1f}\n"
           if top else "- No tickers scored yet\n")
        + f"- {sum(bool(s.anomaly_flag) for s in run.scores)} anomaly flags today\n"
        + "Style: professional, non-advice, 3 sentences."
    )
    text = _gemini_or_none(prompt)

    if not text:
        parts = [f"Markets are {idx[0] if idx else 'mixed'} today."]
        if top and worst and top.ticker != worst.ticker:
            parts.append(
                f"{top.ticker} leads the scorecard at {top.composite:+.1f} "
                f"while {worst.ticker} trails at {worst.composite:+.1f}."
            )
        parts.append(
            f"Your portfolio is worth ${snap['net_worth']:,.2f} CAD "
            f"({snap['day_chg']:+,.2f} today), and "
            f"{sum(bool(s.anomaly_flag) for s in run.scores)} names were "
            f"flagged for unusual activity."
        )
        text = " ".join(parts)
    return text


def _gemini_or_none(prompt: str) -> str | None:
    try:
        from src.advisor import _gemini_generate
        return _gemini_generate(prompt)
    except Exception:
        return None


def render_email(run: brief.BriefRun, snap: dict, funds: dict,
                 company_news: dict | None = None) -> str:
    """Build the full HTML newsletter body (table layout, email-client safe)."""
    mkt = run.market or {}
    buckets = _classify(run)
    holdings_tickers = [r["ticker"] for r in snap["rows"]]
    buys_tickers = [b["ticker"] for b in buckets["buys"]]
    news_map = company_news or {}

    d = run.run_date.isoformat()
    h = ['<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
         'width="100%" class="bd" bgcolor="#F3F4F6" '
         'style="background:#F3F4F6;padding:24px 16px;"><tr><td align="center">',
         '<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
         'width="620" class="card" bgcolor="#FFFFFF" '
         'style="width:620px;max-width:100%;background:#FFFFFF;'
         'border:1px solid #E6E8EB;border-radius:24px;overflow:hidden;">']

    # ---- Header ----
    h.append(
        '<tr><td style="background:#0A0B0D;padding:28px 34px 22px;">'
        '<div style="color:#FFFFFF;font-size:12px;font-weight:700;'
        'letter-spacing:2.5px;text-transform:uppercase;">Stock Advisor</div>'
        '<div style="color:#FFFFFF;font-size:26px;font-weight:700;'
        'margin-top:4px;letter-spacing:-0.5px;">Daily Brief</div>'
        '<div style="color:#A8B0BC;font-size:13px;margin-top:6px;">'
        f'{d} · Your personalized market &amp; portfolio report</div></td></tr>')
    h.append(
        '<tr><td bgcolor="#00B388" style="height:6px;line-height:6px;'
        'font-size:6px;">&nbsp;</td></tr>')

    # ---- Executive summary ----
    h.append(
        '<tr><td class="sec" style="padding:26px 34px 24px;">'
        '<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
        'width="100%"><tr>'
        '<td style="border-left:3px solid #0052FF;padding-left:16px;">'
        '<div class="sec-t" style="font-size:11px;font-weight:700;'
        'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
        'Executive Summary</div>'
        f'<p class="di" style="margin:10px 0 0;font-size:15px;line-height:23px;">'
        f'{html_escape.escape(_exec_summary(run, mkt, snap))}</p>'
        '</td></tr></table></td></tr>')

    # ---- KPI strip ----
    kpi = []
    for i, (label, value) in enumerate([
        ("Net Worth (CAD)", _money(snap["net_worth"])),
        ("Today", f'{_chg(snap["day_chg"])} '
                  f'<span class="dm" style="color:#9AA0A6;">· '
                  f'{_chg(snap["day_pct"], pct=True)}</span>'),
        ("All-Time", f'{_chg(snap["all_time_cad"])} '
                     f'<span class="dm" style="color:#9AA0A6;">· '
                     f'{_chg(snap["all_time_pct"], pct=True)}</span>'),
        ("Cash (CAD)", _money(snap["cash"])),
    ]):
        style = "padding:2px 0 2px 18px;border-left:1px solid #EEF0F3;"
        if i == 0:
            style = "padding:2px 0;"
        kpi.append(
            f'<td class="kpidv" style="{style}">'
            f'<div class="ds" style="font-size:11px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.3px;">{label}</div>'
            f'<div class="di" style="font-size:18px;font-weight:700;'
            f'margin-top:4px;">{value}</div></td>')
    h.append(
        '<tr><td class="sec" style="padding:20px 34px;border-bottom:1px solid #EEF0F3;">'
        f'<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
        f'width="100%"><tr>{"".join(kpi)}</tr></table></td></tr>')

    # ---- Market snapshot ----
    h.append(
        '<tr><td class="sec" style="padding:24px 34px 26px;">'
        '<div class="sec-t" style="font-size:11px;font-weight:700;'
        'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
        'Market Snapshot</div>'
        '<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
        'width="100%" class="tbl" style="margin-top:12px;">'
        f'{_thead(["Index", "Close", "Change"], right={1, 2})}')
    for name, dct in mkt.get("indices", {}).items():
        h.append(
            _trow([html_escape.escape(name), _money(dct["close"]),
                   _chg(dct["chg_pct"], pct=True)], right={1, 2}))
    h.append('</table>')
    news = mkt.get("news", [])
    if news:
        h.append(
            '<div class="ds" style="font-size:11px;font-weight:700;'
            'text-transform:uppercase;letter-spacing:.8px;margin:18px 0 4px;">'
            'Today\u2019s Headlines</div>')
        for a in news[:6]:
            title = a.get("title", "") or ""
            link = a.get("link", "")
            t = html_escape.escape(title)
            body = (f'<a href="{html_escape.escape(link)}" '
                    f'style="color:#0052FF;text-decoration:none;">{t}</a>'
                    if link else t)
            pub = f' <span class="dm" style="color:#9AA0A6;">— {html_escape.escape(a.get("publisher", ""))}</span>'
            h.append(
                f'<div class="ds" style="padding:5px 0 5px 14px;font-size:13px;'
                f'line-height:19px;border-left:2px solid #E6E8EB;">{body}{pub}</div>')
    h.append('</td></tr>')

    # ---- Portfolio snapshot ----
    h.append(
        '<tr><td class="sec" style="padding:24px 34px 26px;">'
        '<div class="sec-t" style="font-size:11px;font-weight:700;'
        'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
        'Portfolio</div>')
    if snap["rows"]:
        h.append(
            '<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
            'width="100%" class="tbl" style="margin-top:12px;">'
            f'{_thead(["Ticker", "Shares", "Price", "Day", "Value (CAD)",
                       "Return", "Weight"], right={1, 2, 3, 4, 5, 6})}')
        for r in snap["rows"]:
            w = r["value_cad"] / snap["invested"] * 100 if snap["invested"] else 0.0
            h.append(
                _trow([
                    _ticker_cell(r["ticker"]),
                    f'{r["shares"]:g}',
                    _money(r["price"]) if r["price"] else "—",
                    _chg(r["day_pct"], pct=True),
                    f'<b class="di">{_money(r["value_cad"])}</b>',
                    _chg(r["return_pct"], pct=True),
                    f'<span class="dm" style="color:#9AA0A6;">{w:.1f}%</span>',
                ], right={1, 2, 3, 4, 5, 6}))
        h.append('</table>')
    else:
        h.append('<p class="ds" style="margin:6px 0 0;color:#5B616E;">'
                 'No equity positions.</p>')
    h.append('</td></tr>')

    # ---- Top stocks to consider ----
    if buckets["buys"]:
        h.append(
            '<tr><td class="sec" style="padding:24px 34px 26px;">'
            '<div class="sec-t" style="font-size:11px;font-weight:700;'
            'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
            'Top Stocks To Consider</div>')
        h.append(_fund_table(buckets["buys"], funds, "buy"))
        h.append('</td></tr>')

    # ---- Watch out ----
    if buckets["alerts"]:
        h.append(
            '<tr><td class="sec" style="padding:24px 34px 26px;">'
            '<div class="sec-t" style="font-size:11px;font-weight:700;'
            'letter-spacing:1.5px;text-transform:uppercase;'
            'color:#C3312B;">Watch Out</div>'
            '<table role="presentation" cellspacing="0" cellpadding="0" border="0" '
            'width="100%" class="tbl" style="margin-top:12px;">'
            f'{_thead(["Ticker", "Composite", "Day", "Position", "Reason"], right={1, 2})}')
        for e in buckets["alerts"]:
            pos = "Held" if e["weight"] else "Watch"
            weight = f" · {e['weight']:.0f}% of portfolio" if e["weight"] else ""
            reason = e["recommendation"] or (e["anomaly_detail"] or "").replace("\n", " ")
            h.append(
                _trow([
                    _ticker_cell(e["ticker"]),
                    _chg(e["composite"]),
                    _chg(e["day_pct"] or 0, pct=True),
                    f'{pos}{html_escape.escape(weight)}',
                    f'<span class="ds" style="font-size:12px;color:#5B616E;">'
                    f'{html_escape.escape(reason)}</span>',
                ], right={1, 2}))
        h.append('</table></td></tr>')

    # ---- Holdings valuation snapshot ----
    if funds:
        held = {t: funds[t] for t in holdings_tickers if t in funds}
        if held:
            h.append(
                '<tr><td class="sec" style="padding:24px 34px 26px;">'
                '<div class="sec-t" style="font-size:11px;font-weight:700;'
                'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
                'Holdings — Valuation Snapshot</div>')
            h.append(_fund_table_flat(held))
            h.append('</td></tr>')

    # ---- Buy-list valuation snapshot ----
    if funds:
        wanted = {t: funds[t] for t in buys_tickers if t in funds and t not in holdings_tickers}
        if wanted:
            h.append(
                '<tr><td class="sec" style="padding:24px 34px 26px;">'
                '<div class="sec-t" style="font-size:11px;font-weight:700;'
                'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
                'Top Picks — Valuation Snapshot</div>')
            h.append(_fund_table_flat(wanted))
            h.append('</td></tr>')

    # ---- Per-company news ----
    if news_map:
        h.append(
            '<tr><td class="sec" style="padding:24px 34px 26px;">'
            '<div class="sec-t" style="font-size:11px;font-weight:700;'
            'letter-spacing:1.5px;text-transform:uppercase;color:#5B616E;">'
            'Company News</div>')
        for tk, items in list(news_map.items())[:6]:
            if not items:
                continue
            h.append(f'<div class="di" style="font-size:13px;font-weight:700;'
                     f'margin:14px 0 4px;">{html_escape.escape(tk)}</div>')
            for a in items[:3]:
                title = a.get("title", "") or ""
                link = a.get("link", "")
                t = html_escape.escape(title)
                body = (f'<a href="{html_escape.escape(link)}" '
                        f'style="color:#0052FF;text-decoration:none;">{t}</a>'
                        if link else t)
                h.append(f'<div class="ds" style="padding:2px 0 2px 14px;'
                         f'font-size:13px;line-height:19px;'
                         f'border-left:2px solid #E6E8EB;">{body}</div>')
        h.append('</td></tr>')

    # ---- Footer ----
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M (%Z)").replace(" ()", "")
    prefs = html_escape.escape(_env("EMAIL_TO", DEFAULT_RECIPIENT))
    h.append(
        f'<tr><td bgcolor="#F7F8FA" '
        f'style="background:#F7F8FA;border-top:1px solid #E6E8EB;'
        f'padding:22px 34px 26px;border-bottom-left-radius:23px;'
        f'border-bottom-right-radius:23px;">'
        f'<div class="ds" style="font-size:12px;line-height:19px;">'
        f'Generated {ts}. Scores reflect a weighted composite of sentiment, '
        f'technicals, machine learning and analyst consensus, minus an anomaly '
        f'penalty where flagged.</div>'
        f'<div class="ds" style="font-size:12px;line-height:19px;margin-top:10px;">'
        f'This email is for informational purposes only and is not financial '
        f'advice. Past performance does not guarantee future results.</div>'
        f'<div class="ds" style="font-size:12px;margin-top:14px;">'
        f'Sent by <b class="di">Stock Advisor</b> · '
        f'<a href="mailto:{prefs}" style="color:#0052FF;text-decoration:none;">'
        f'Preferences</a></div></td></tr>')

    h.append('</table></td></tr></table>')

    body = "\n".join(h)
    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<style>{_STYLE}</style></head>"
            '<body class="bd" '
            'style="margin:0;padding:0;width:100%;background:#F3F4F6;">'
            f"{body}</body></html>")


def _fund_table(entries: list, funds: dict, chip: str) -> str:
    rows = ['<table role="presentation" cellspacing="0" cellpadding="0" '
            'border="0" width="100%" class="tbl" style="margin-top:12px;">',
            _thead(["Ticker", "Score", "Recommendation", "Why"], right={1})]
    for e in entries[:5]:
        why = _why_line(e)
        rows.append(
            _trow([
                f'{_ticker_cell(e["ticker"])} '
                f'<span class="dm" style="color:#9AA0A6;font-size:11px;">'
                f'{html_escape.escape(e.get("signal") or "")}</span>',
                _chg(e["composite"]),
                _rec_pill(e.get("recommendation")),
                f'<span class="ds" style="font-size:12px;line-height:18px;'
                f'color:#5B616E;">{html_escape.escape(why)}</span>',
            ], right={1}))
    rows.append('</table>')
    return "".join(rows)


def _why_line(e: dict) -> str:
    parts = []
    comp = e["components"]
    labels = ["Sentiment", "Technical", "ML", "Analyst"]
    for lbl, v in zip(labels, comp):
        if v:
            parts.append(f"{lbl} {v:+.1f}")
    if e.get("anomaly"):
        parts.append("anomaly flagged")
    if e.get("headline"):
        hd = str(e["headline"])[:100]
        parts.append(f"top headline: {hd}")
    return " · ".join(parts) if parts else "—"


def _fund_table_flat(funds: dict) -> str:
    names = list(funds)
    rows = ['<table role="presentation" cellspacing="0" cellpadding="0" '
            'border="0" width="100%" class="tbl" style="margin-top:12px;">',
            '<tr><td class="thead" style="padding:9px 12px;">Metric</td>']
    for n in names:
        rows.append(f'<td class="thead" style="padding:9px 12px;text-align:right;">'
                    f'{html_escape.escape(n)}</td>')
    rows.append('</tr>')
    for label, key in _FUND_KEYS:
        rows.append(
            f'<tr><td class="di" style="padding:8px 12px;font-weight:600;'
            f'font-size:12px;border-bottom:1px solid #F1F2F4;">{label}</td>')
        for n in names:
            v = funds[n].get(key, "N/A")
            cell = "—" if v == "N/A" else html_escape.escape(str(v))
            rows.append(f'<td class="num" style="padding:8px 12px;font-size:12px;'
                        f'color:#5B616E;border-bottom:1px solid #F1F2F4;">{cell}</td>')
        rows.append('</tr>')
    rows.append('</table>')
    return "".join(rows)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def send_email(html_body: str, subject: str | None = None) -> bool:
    """Send the newsletter via SMTP. Returns True on success."""
    if not is_smtp_configured():
        return False
    cfg = smtp_config()
    d = date.today().isoformat()
    subj = subject or f"Stock Advisor Daily Brief — {d}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subj
    msg["From"] = formataddr((cfg["from_name"], cfg["from_addr"]))
    msg["To"] = ", ".join(recipients())
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText("Please view this newsletter in an HTML-capable client.", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    if cfg["port"] == 465:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx) as smtp:
            smtp.login(cfg["user"], cfg["password"])
            smtp.sendmail(cfg["from_addr"], recipients(), msg.as_string())
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as smtp:
            smtp.ehlo()
            if smtp.has_extn("starttls"):
                smtp.starttls()
                smtp.ehlo()
            smtp.login(cfg["user"], cfg["password"])
            smtp.sendmail(cfg["from_addr"], recipients(), msg.as_string())
    return True


def save_preview(html_body: str) -> str:
    """Write the rendered HTML to data/ for previewing without SMTP."""
    path = os.path.join("data", f"newsletter_{date.today().isoformat()}.html")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_body)
    return path


def run_newsletter(send: bool = True, preview: bool = False) -> bool:
    """Orchestrate building + sending the daily newsletter."""
    run = brief.run_brief(fetch_market=True)
    snap = portfolio_snapshot()
    tickers = [r["ticker"] for r in snap["rows"]]
    tickers += [s.ticker for s in run.scores]
    tickers = list(dict.fromkeys(tickers))
    funds = _fundamentals(tickers)
    held = [s.ticker for s in run.scores if s.portfolio_weight and s.portfolio_weight > 0]
    company_news = data_client.get_ticker_news_batch(held, limit=3)

    body = render_email(run, snap, funds, company_news)

    if not send or not is_smtp_configured():
        path = save_preview(body)
        print(f"[yellow]No SMTP configured — preview saved to {path}[/yellow]")
        return False

    try:
        ok = send_email(body)
        print(f"[green]Newsletter sent to {', '.join(recipients())}.[/green]" if ok
              else "[red]Sending failed.[/red]")
        return ok
    except Exception as exc:
        print(f"[red]Newsletter send failed: {exc}[/red]")
        save_preview(body)
        return False