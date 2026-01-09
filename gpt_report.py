#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Static HTML report generator (light theme + per-card collapse + global collapse/expand).

Key UX:
- Light theme (white background).
- Abstract is OPEN by default; bigger area.
- Compute area is more compact; evidence/JSON won't overflow horizontally.
- Global buttons: Expand all / Collapse all (collapse keeps only title + venue + compute summary).
- Each paper has its own Expand/Collapse button.
"""

import os
import json
import csv
import html
import argparse
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# IO helpers
# ---------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def read_csv(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]

def ensure_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)

def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = ensure_str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}

def safe_json_dumps(x: Any, indent: int = 2) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, indent=indent)
    except Exception:
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return "null"


# ---------------------------
# Merge helpers
# ---------------------------

def build_multi_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = ensure_str(r.get("paper_id")).strip()
        aid = ensure_str(r.get("arxiv_id")).strip()
        title = ensure_str(r.get("title")).strip()
        if pid:
            idx[pid] = r
        if aid and aid not in idx:
            idx[aid] = r
        if title and title not in idx:
            idx[title] = r
    return idx

def merge_dict(dst: Dict[str, Any], src: Dict[str, Any], prefer_src: bool = True) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in (src or {}).items():
        if prefer_src:
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            out[k] = v
        else:
            if k not in out or out[k] in (None, "", [], {}, "null"):
                out[k] = v
    return out

def pick_key(obj: Dict[str, Any]) -> str:
    for k in ("paper_id", "arxiv_id", "title"):
        v = ensure_str(obj.get(k)).strip()
        if v:
            return v
    return ""


# ---------------------------
# HTML helpers
# ---------------------------

def esc(s: Any) -> str:
    return html.escape(ensure_str(s), quote=True)

def file_link(path: str, base_dir: Optional[str] = None) -> str:
    if not path:
        return ""
    p = os.path.normpath(path)
    if base_dir:
        bd = os.path.abspath(base_dir)
        ap = os.path.abspath(p)
        if ap.startswith(bd + os.sep):
            rel = os.path.relpath(ap, bd).replace(os.sep, "/")
            return rel
    return p.replace(os.sep, "/")

def render_badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{esc(text)}</span>'

def render_kv(label: str, value_html: str) -> str:
    if not value_html:
        return ""
    return f'<div class="kv"><div class="k">{esc(label)}</div><div class="v">{value_html}</div></div>'

def render_links(p: Dict[str, Any], report_base_dir: Optional[str]) -> str:
    paper_url = ensure_str(p.get("paper_url")).strip()
    page_url = ensure_str(p.get("page_url")).strip()
    canonical_url = ensure_str(p.get("canonical_url")).strip()

    pdf_path = ensure_str(p.get("pdf_path")).strip()
    md_path = ensure_str(p.get("md_path")).strip()

    links = []
    if paper_url:
        links.append(f'<a href="{esc(paper_url)}" target="_blank" rel="noopener">Paper URL</a>')
    if page_url:
        links.append(f'<a href="{esc(page_url)}" target="_blank" rel="noopener">Project/Page</a>')
    if canonical_url and canonical_url not in {paper_url, page_url}:
        links.append(f'<a href="{esc(canonical_url)}" target="_blank" rel="noopener">Canonical</a>')

    if pdf_path:
        href = file_link(pdf_path, report_base_dir)
        links.append(f'<a href="{esc(href)}" target="_blank" rel="noopener">Local PDF</a>')
    if md_path:
        href = file_link(md_path, report_base_dir)
        links.append(f'<a href="{esc(href)}" target="_blank" rel="noopener">Local MD</a>')

    if not links:
        return '<span class="muted">No links</span>'
    return " · ".join(links)

def render_gpu_hardware_table(p: Dict[str, Any]) -> str:
    hw = p.get("gpu_hardware")
    if isinstance(hw, str):
        try:
            hw = json.loads(hw)
        except Exception:
            hw = None
    if not isinstance(hw, list) or not hw:
        return '<span class="muted">No explicit GPU hardware parsed.</span>'

    rows = []
    for it in hw:
        if not isinstance(it, dict):
            continue
        model = ensure_str(it.get("model") or "UNKNOWN")
        cnt = it.get("count")
        mem = it.get("mem_gb")
        conf = ensure_str(it.get("confidence") or "")
        cnt_s = esc(cnt) if cnt is not None else "—"
        mem_s = f"{float(mem):.1f} GB" if isinstance(mem, (int, float)) else "—"
        conf_s = esc(conf) if conf else "—"
        rows.append(f"<tr><td>{esc(model)}</td><td>{cnt_s}</td><td>{mem_s}</td><td>{conf_s}</td></tr>")

    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Count</th><th>Memory</th><th>Conf.</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """

def render_match_contexts(p: Dict[str, Any], max_items: int = 14) -> str:
    snippets = p.get("fine_snippets") or p.get("coarse_snippets") or []
    if isinstance(snippets, str):
        try:
            snippets = json.loads(snippets)
        except Exception:
            snippets = []
    if not isinstance(snippets, list) or not snippets:
        return '<span class="muted">No compute evidence snippets.</span>'

    items = []
    for sn in snippets:
        if not isinstance(sn, dict):
            continue
        page = sn.get("page")
        page_tag = f"p{page}" if page is not None and ensure_str(page) != "" else "p?"
        mcs = sn.get("match_contexts") or []
        if isinstance(mcs, str):
            try:
                mcs = json.loads(mcs)
            except Exception:
                mcs = []
        if not mcs:
            t = ensure_str(sn.get("text")).strip()
            if t:
                items.append(f"<li><span class='tag'>{esc(page_tag)}</span><div class='ctx'>{esc(t)}</div></li>")
            continue

        for ctx in mcs:
            if not isinstance(ctx, dict):
                continue
            ctype = ensure_str(ctx.get("type"))
            match = ensure_str(ctx.get("match"))
            context = ensure_str(ctx.get("context")).strip()
            if len(context) > 520:
                context = context[:520] + "..."
            items.append(
                "<li>"
                f"<span class='tag'>{esc(page_tag)}</span>"
                f"<span class='tag2'>{esc(ctype)}</span>"
                f"<span class='match'>{esc(match)}</span>"
                f"<div class='ctx'>{esc(context)}</div>"
                "</li>"
            )
            if len(items) >= max_items:
                break
        if len(items) >= max_items:
            break

    return f"<ul class='ctxlist'>{''.join(items)}</ul>" if items else '<span class="muted">No extractable contexts.</span>'

def render_llm_compute(p: Dict[str, Any]) -> str:
    summary_zh = ensure_str(p.get("compute_summary_zh")).strip()
    structured = p.get("compute_structured")

    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except Exception:
            structured = None

    parts = []
    parts.append(f"<div class='llm-sum'>{esc(summary_zh) if summary_zh else '<span class=\"muted\">No LLM compute summary.</span>'}</div>")

    if isinstance(structured, dict) and structured:
        pretty = safe_json_dumps(structured, indent=2)
        parts.append(
            "<details class='details'>"
            "<summary>LLM structured JSON</summary>"
            f"<pre class='pre'>{esc(pretty)}</pre>"
            "</details>"
        )
    return "".join(parts)

def render_abstract(p: Dict[str, Any]) -> str:
    zh = ensure_str(p.get("abstract_zh")).strip()
    en = ensure_str(p.get("abstract")).strip()
    if not zh and not en:
        return '<div class="muted">No abstract found.</div>'

    blocks = []
    if zh:
        blocks.append(f"<div class='abs'><div class='abs-h'>中文摘要</div><div class='abs-b'>{esc(zh)}</div></div>")
    if en:
        blocks.append(
            "<details class='details'>"
            "<summary>英文摘要（展开）</summary>"
            f"<div class='abs-b'>{esc(en)}</div>"
            "</details>"
        )

    # ✅ 默认展开摘要
    return (
        "<details class='details openable' open>"
        "<summary>摘要（点击收起/展开）</summary>"
        + "".join(blocks) +
        "</details>"
    )

def compact_compute_text(p: Dict[str, Any]) -> str:
    """
    Generate a short compute line for collapsed mode:
    prefer LLM structured -> gpu_compute parsed -> fallback.
    """
    # 1) LLM structured
    structured = p.get("compute_structured")
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except Exception:
            structured = None
    if isinstance(structured, dict):
        gpu_models = structured.get("gpu_models") or []
        gpu_count = structured.get("gpu_count")
        mem = structured.get("gpu_memory_gb")
        gpu_hours = structured.get("gpu_hours")
        ttime = structured.get("training_time")

        parts = []
        if gpu_models:
            parts.append(", ".join([ensure_str(x) for x in gpu_models if ensure_str(x).strip()]))
        if gpu_count is not None:
            parts.append(f"x{gpu_count}")
        if mem is not None:
            parts.append(f"{mem}GB")
        if gpu_hours is not None:
            parts.append(f"{gpu_hours} GPU-hours")
        if ttime:
            parts.append(f"{ttime}")
        if parts:
            return "Compute: " + " ".join(parts)

    # 2) gpu_compute parsed
    arch = p.get("gpu_arch")
    counts = p.get("gpu_counts")
    mems = p.get("gpu_mem_gb")

    if isinstance(arch, str):
        arch_s = arch.strip()
    elif isinstance(arch, list):
        arch_s = ", ".join([ensure_str(x) for x in arch if ensure_str(x).strip()])
    else:
        arch_s = ""

    cnt_s = ""
    if isinstance(counts, list) and counts:
        ints = [x for x in counts if isinstance(x, int)]
        if ints:
            cnt_s = f"x{max(ints)}"
    mem_s = ""
    if isinstance(mems, list) and mems:
        nums = [x for x in mems if isinstance(x, (int, float))]
        if nums:
            mem_s = f"{max(nums):.0f}GB"

    if arch_s:
        return "Compute: " + " ".join([x for x in [arch_s, cnt_s, mem_s] if x]).strip()

    # 3) fallback
    if to_bool(p.get("gpu_detected")):
        return "Compute: GPU mentioned (details inside)"
    return "Compute: unknown"

def compute_search_blob(p: Dict[str, Any]) -> str:
    keys = [
        "title", "alias", "venue", "category", "year",
        "arxiv_id", "doi",
        "paper_url", "page_url", "canonical_url",
        "compute_summary_zh",
    ]
    parts = []
    for k in keys:
        v = ensure_str(p.get(k)).strip()
        if v:
            parts.append(v)
    # add compact compute
    parts.append(compact_compute_text(p))
    return " | ".join(parts).lower()

def render_paper_card(p: Dict[str, Any], report_base_dir: Optional[str]) -> str:
    title = ensure_str(p.get("title")).strip() or "Untitled"
    venue = ensure_str(p.get("venue")).strip()
    year = ensure_str(p.get("year")).strip()
    category = ensure_str(p.get("category")).strip()
    alias = ensure_str(p.get("alias")).strip()
    arxiv = ensure_str(p.get("arxiv_id")).strip()
    doi = ensure_str(p.get("doi")).strip()

    gpu_detected = to_bool(p.get("gpu_detected"))
    pdf_path = ensure_str(p.get("pdf_path")).strip()
    md_path = ensure_str(p.get("md_path")).strip()
    has_zh = bool(ensure_str(p.get("abstract_zh")).strip())

    badges = []
    if gpu_detected:
        badges.append(render_badge("GPU", "ok"))
    if pdf_path:
        badges.append(render_badge("PDF", "info"))
    if md_path:
        badges.append(render_badge("MD", "info"))
    if has_zh:
        badges.append(render_badge("ZH", "ok"))

    links_html = render_links(p, report_base_dir)

    meta_line = " · ".join([x for x in [
        (venue + (f" {year}" if year else "")) if venue or year else "",
        category if category else "",
        (f"Alias: {alias}") if alias else "",
        (f"arXiv: {arxiv}") if arxiv else "",
        (f"DOI: {doi}") if doi else "",
    ] if x])

    # content blocks
    abstract_block = (
        "<div class='sec sec-abs'>"
        "<div class='sec-h'>摘要</div>"
        "<div class='sec-b'>"
        f"{render_abstract(p)}"
        "</div></div>"
    )

    evidence = (
        "<details class='details'>"
        "<summary>算力证据（关键词与上下文）</summary>"
        f"{render_match_contexts(p)}"
        "</details>"
    )

    compute_block = (
        "<div class='sec sec-comp'>"
        "<div class='sec-h'>算力</div>"
        "<div class='sec-b'>"
        f"{render_kv('GPU 硬件解析', render_gpu_hardware_table(p))}"
        f"{render_kv('LLM 润色结果', render_llm_compute(p))}"
        f"{render_kv('证据', evidence)}"
        "</div></div>"
    )

    search_blob = compute_search_blob(p)
    compact_line = compact_compute_text(p)

    # ✅ paper-body 可整体收起；收起后只显示标题/会议信息/算力短句
    return f"""
    <article class="paper" data-search="{esc(search_blob)}" data-collapsed="0" data-has-zh="{str(has_zh).lower()}" data-has-pdf="{str(bool(pdf_path)).lower()}" data-has-gpu="{str(gpu_detected).lower()}">
      <div class="paper-head">
        <div class="head-left">
          <div class="title">{esc(title)}</div>
          <div class="meta">{esc(meta_line) if meta_line else '<span class="muted">No meta.</span>'}</div>
          <div class="mini">{esc(compact_line)}</div>
          <div class="links">{links_html}</div>
        </div>

        <div class="head-right">
          <div class="badges">{''.join(badges)}</div>
          <button class="mini-btn toggle-card" type="button">收起</button>
        </div>
      </div>

      <div class="paper-body">
        <div class="grid">
          {abstract_block}
          {compute_block}
        </div>
      </div>
    </article>
    """


def build_stats(papers: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(papers)
    with_pdf = 0
    gpu_detected = 0
    with_zh = 0
    for p in papers:
        if ensure_str(p.get("pdf_path")).strip():
            with_pdf += 1
        if to_bool(p.get("gpu_detected")):
            gpu_detected += 1
        if ensure_str(p.get("abstract_zh")).strip():
            with_zh += 1
    return {
        "total": total,
        "with_pdf": with_pdf,
        "gpu_detected": gpu_detected,
        "with_zh": with_zh,
    }


def generate_html(papers: List[Dict[str, Any]], report_title: str, report_base_dir: Optional[str]) -> str:
    stats = build_stats(papers)
    cards = "\n".join(render_paper_card(p, report_base_dir) for p in papers)

    # ✅ Light theme + wider layout + better wrapping
    css = r"""
    :root{
      --bg:#ffffff;
      --text:#0f172a;
      --muted:#64748b;
      --card:#ffffff;
      --card2:#f8fafc;
      --border:#e2e8f0;
      --ok:#16a34a;
      --warn:#f59e0b;
      --info:#2563eb;
      --shadow: 0 10px 30px rgba(15,23,42,.08);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: var(--bg);
      color: var(--text);
    }
    .wrap{
      max-width: 1480px;   /* ✅ 更宽 */
      margin:0 auto;
      padding: 22px 18px 56px;
    }

    .topbar{display:flex;gap:14px;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;margin-bottom:14px;}
    h1{margin:0;font-size:22px;letter-spacing:.2px}
    .sub{color:var(--muted);font-size:13px;margin-top:6px}

    .panel{
      display:flex;gap:10px;align-items:center;flex-wrap:wrap;
      background: var(--card2);
      border:1px solid var(--border);
      padding:12px;border-radius:14px;
    }
    .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .input{
      min-width:360px;flex:1;
      background:#fff;
      border:1px solid var(--border);
      border-radius:12px;
      padding:10px 12px;
      color:var(--text);
      outline:none;
    }
    .btn{
      background:#fff;
      border:1px solid var(--border);
      color:var(--text);
      padding:10px 12px;
      border-radius:12px;
      cursor:pointer;
    }
    .btn:hover{border-color:#cbd5e1}
    .chk{
      display:flex;gap:8px;align-items:center;
      color:var(--muted);font-size:12px;
      padding:6px 10px;border:1px solid var(--border);
      border-radius:999px;background:#fff;
    }

    .list{display:flex;flex-direction:column;gap:14px;margin-top:14px}

    .paper{
      background: var(--card);
      border:1px solid var(--border);
      border-radius:16px;
      padding:14px 14px 10px;
      box-shadow: var(--shadow);
    }

    .paper-head{display:flex;gap:12px;align-items:flex-start;justify-content:space-between}
    .head-left{min-width:0;flex:1}
    .head-right{display:flex;flex-direction:column;gap:10px;align-items:flex-end}
    .title{font-size:16px;font-weight:700;line-height:1.35}
    .meta{margin-top:6px;color:var(--muted);font-size:12px}
    .mini{margin-top:6px;color:#0f172a;font-size:12px}
    .links{margin-top:6px;font-size:12px;color:var(--muted)}
    .links a{color:var(--info);text-decoration:none}
    .links a:hover{text-decoration:underline}

    .badges{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
    .badge{font-size:11px;padding:4px 8px;border-radius:999px;border:1px solid var(--border); color:var(--muted);background:#fff}
    .badge.ok{border-color:rgba(22,163,74,.35); color: #166534; background: rgba(22,163,74,.08)}
    .badge.warn{border-color:rgba(245,158,11,.45); color:#92400e; background: rgba(245,158,11,.10)}
    .badge.info{border-color:rgba(37,99,235,.35); color:#1e3a8a; background: rgba(37,99,235,.08)}

    .mini-btn{
      border:1px solid var(--border);
      background:#fff;
      color:var(--text);
      border-radius:12px;
      padding:8px 10px;
      cursor:pointer;
      font-size:12px;
    }
    .mini-btn:hover{border-color:#cbd5e1}

    /* ✅ 折叠模式：隐藏 paper-body，仅保留头部（标题/会议/算力短句/链接） */
    .paper[data-collapsed="1"] .paper-body{display:none}
    .paper[data-collapsed="1"]{padding-bottom:14px}

    /* ✅ 摘要更大，算力更小：grid 比例调整 */
    .grid{
      display:grid;
      grid-template-columns: 1.7fr 1fr;   /* ✅ 左侧摘要更宽 */
      gap:12px;
      margin-top:12px
    }
    @media (max-width: 980px){ .grid{grid-template-columns:1fr} .input{min-width:220px}}

    .sec{
      background: var(--card2);
      border:1px solid var(--border);
      border-radius:14px;
      padding:10px 10px 8px;
      min-width:0;
    }
    .sec-h{font-size:12px;color:#334155;margin-bottom:8px;letter-spacing:.2px}
    .sec-b{display:flex;flex-direction:column;gap:10px}

    /* 摘要区域更“能装”：默认高度更大，超出滚动 */
    .sec-abs .abs-b{
      white-space: pre-wrap;
      line-height:1.65;
      font-size:12.5px;
      max-height: 420px;    /* ✅ 让摘要可视面积更大 */
      overflow:auto;
      padding-right:6px;
    }

    /* 算力区域更紧凑 */
    .sec-comp{font-size:12px}
    .kv{display:grid;grid-template-columns:110px 1fr;gap:10px;align-items:start}
    .k{color:var(--muted);font-size:12px}
    .v{font-size:12px;line-height:1.5;min-width:0}

    .muted{color:var(--muted)}

    details.details{
      background:#fff;
      border:1px solid var(--border);
      border-radius:12px;
      padding:8px 10px;
      min-width:0;
    }
    details.details > summary{cursor:pointer; color:#334155; font-size:12px}

    .abs-h{color:#334155;font-size:12px;margin-bottom:6px}

    .llm-sum{font-size:12px; line-height:1.6; white-space:pre-wrap; overflow-wrap:anywhere}

    /* ✅ JSON 不超宽：pre-wrap + 容器内滚动 */
    .pre{
      margin:8px 0 0;
      background:#0b1220;
      color:#e5e7eb;
      border-radius:12px;
      padding:10px;
      overflow:auto;
      max-height:260px;
      white-space: pre-wrap;     /* ✅ 自动换行 */
      overflow-wrap:anywhere;    /* ✅ 强制断行 */
      word-break: break-word;
    }

    .table-wrap{overflow:auto;border:1px solid var(--border); border-radius:12px;background:#fff}
    table{border-collapse:collapse; width:100%; font-size:12px; table-layout: fixed}
    th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left; word-break: break-word}
    th{color:#334155;background: #f1f5f9; position:sticky; top:0}

    /* ✅ 证据上下文不超宽 */
    .ctxlist{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:10px}
    .ctxlist li{list-style:disc}
    .tag{display:inline-block; padding:2px 6px; border-radius:999px; border:1px solid var(--border); color:#1e3a8a; background:rgba(37,99,235,.08); margin-right:6px; font-size:11px}
    .tag2{display:inline-block; padding:2px 6px; border-radius:999px; border:1px solid var(--border); color:#166534; background:rgba(22,163,74,.08); margin-right:6px; font-size:11px}
    .match{display:inline-block; padding:2px 6px; border-radius:10px; border:1px solid rgba(245,158,11,.45); color:#92400e; background:rgba(245,158,11,.10); margin-right:6px; font-size:11px}
    .ctx{
      margin-top:4px;
      color:var(--text);
      overflow-wrap:anywhere;
      word-break: break-word;
      line-height:1.55;
    }

    .foot{margin-top:18px;color:var(--muted);font-size:12px}
    """

    js = r"""
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    function setCollapsed(card, collapsed){
      card.dataset.collapsed = collapsed ? "1" : "0";
      const btn = card.querySelector(".toggle-card");
      if (btn){
        btn.textContent = collapsed ? "展开" : "收起";
      }
    }

    function applyFilter(){
      const q = ($("#q").value || "").trim().toLowerCase();
      const onlyGpu = $("#onlyGpu").checked;
      const onlyPdf = $("#onlyPdf").checked;
      const onlyZh  = $("#onlyZh").checked;

      let shown = 0;
      $$(".paper").forEach(card => {
        const blob = (card.getAttribute("data-search") || "");
        const hasGpu = card.getAttribute("data-has-gpu") === "true";
        const hasPdf = card.getAttribute("data-has-pdf") === "true";
        const hasZh  = card.getAttribute("data-has-zh")  === "true";

        let ok = true;
        if (q && !blob.includes(q)) ok = false;
        if (onlyGpu && !hasGpu) ok = false;
        if (onlyPdf && !hasPdf) ok = false;
        if (onlyZh  && !hasZh)  ok = false;

        card.style.display = ok ? "" : "none";
        if (ok) shown++;
      });

      $("#shown").textContent = shown;
    }

    function clearAll(){
      $("#q").value = "";
      $("#onlyGpu").checked = false;
      $("#onlyPdf").checked = false;
      $("#onlyZh").checked  = false;
      applyFilter();
    }

    function collapseAll(){
      $$(".paper").forEach(card => setCollapsed(card, true));
    }

    function expandAll(){
      $$(".paper").forEach(card => setCollapsed(card, false));
    }

    window.addEventListener("DOMContentLoaded", () => {
      $("#q").addEventListener("input", applyFilter);
      $("#onlyGpu").addEventListener("change", applyFilter);
      $("#onlyPdf").addEventListener("change", applyFilter);
      $("#onlyZh").addEventListener("change", applyFilter);

      $("#clear").addEventListener("click", clearAll);
      $("#collapseAll").addEventListener("click", collapseAll);
      $("#expandAll").addEventListener("click", expandAll);

      $$(".paper .toggle-card").forEach(btn => {
        btn.addEventListener("click", (e) => {
          const card = e.target.closest(".paper");
          const collapsed = card.dataset.collapsed === "1";
          setCollapsed(card, !collapsed);
        });
      });

      applyFilter();
    });
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{esc(report_title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>{esc(report_title)}</h1>
        <div class="sub">
          Total: {stats["total"]} · With PDF: {stats["with_pdf"]} · GPU detected: {stats["gpu_detected"]} · With Chinese abstract: {stats["with_zh"]}
          · Showing: <span id="shown">{stats["total"]}</span>
        </div>
      </div>

      <div class="panel">
        <div class="controls">
          <input id="q" class="input" placeholder="搜索：标题 / venue / arXiv / DOI / 类别 / GPU / 关键词 …"/>
          <button id="clear" class="btn">清空</button>
          <button id="collapseAll" class="btn">全部收起（仅标题/会议/算力）</button>
          <button id="expandAll" class="btn">全部展开</button>
          <label class="chk"><input id="onlyGpu" type="checkbox"/> 仅 GPU</label>
          <label class="chk"><input id="onlyPdf" type="checkbox"/> 仅有 PDF</label>
          <label class="chk"><input id="onlyZh" type="checkbox"/> 仅有中文摘要</label>
        </div>
      </div>
    </div>

    <div class="list">
      {cards}
    </div>

    <div class="foot">
      Tip：如果你希望本地 PDF/MD 链接更稳定，推荐在 <code>./store/report</code> 下运行 <code>python -m http.server</code> 打开。
    </div>
  </div>

  <script>{js}</script>
</body>
</html>
"""


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Paper Report", help="Report title")

    ap.add_argument("--parsed_csv", default="./store/parse/papers.csv")
    ap.add_argument("--parsed_jsonl", default="./store/parse/papers.jsonl")

    ap.add_argument("--enriched_csv", default="./store/enrich/papers.enriched.csv")
    ap.add_argument("--enriched_jsonl", default="./store/enrich/papers.enriched.jsonl")

    ap.add_argument("--pages_jsonl", default="./store/ocr/papers.pages.jsonl")
    ap.add_argument("--compute_jsonl", default="./store/compute/gpu_compute.jsonl")
    ap.add_argument("--llm_jsonl", default="./store/llm/papers.llm.jsonl")

    ap.add_argument("--out_html", default="./store/report/report.html")
    ap.add_argument("--report_base_dir", default="./store", help="relativize links to this dir (recommended: ./store)")

    ap.add_argument("--sort", default="venue_year_title", choices=["venue_year_title", "title", "year", "none"])
    args = ap.parse_args()

    parsed_rows = read_csv(args.parsed_csv) or read_jsonl(args.parsed_jsonl)
    enriched_rows = read_csv(args.enriched_csv) or read_jsonl(args.enriched_jsonl)
    pages_rows = read_jsonl(args.pages_jsonl)
    compute_rows = read_jsonl(args.compute_jsonl)
    llm_rows = read_jsonl(args.llm_jsonl)

    parsed_idx = build_multi_index(parsed_rows)
    enriched_idx = build_multi_index(enriched_rows)
    pages_idx = build_multi_index(pages_rows)
    compute_idx = build_multi_index(compute_rows)
    llm_idx = build_multi_index(llm_rows)

    universe_keys = []
    for src in (llm_rows, enriched_rows, parsed_rows, pages_rows, compute_rows):
        for r in src:
            k = pick_key(r)
            if k:
                universe_keys.append(k)

    seen = set()
    keys = []
    for k in universe_keys:
        if k not in seen:
            seen.add(k)
            keys.append(k)

    papers: List[Dict[str, Any]] = []
    for k in keys:
        base: Dict[str, Any] = {}
        if k in parsed_idx:
            base = merge_dict(base, parsed_idx[k], prefer_src=True)
        if k in enriched_idx:
            base = merge_dict(base, enriched_idx[k], prefer_src=True)
        if k in pages_idx:
            base = merge_dict(base, pages_idx[k], prefer_src=True)
        if k in compute_idx:
            base = merge_dict(base, compute_idx[k], prefer_src=True)
        if k in llm_idx:
            base = merge_dict(base, llm_idx[k], prefer_src=True)

        if "gpu_detected" in base:
            base["gpu_detected"] = to_bool(base.get("gpu_detected"))

        papers.append(base)

    def sort_key(p: Dict[str, Any]) -> Tuple:
        venue = ensure_str(p.get("venue")).lower()
        year = ensure_str(p.get("year")).strip()
        try:
            year_i = int(year)
        except Exception:
            year_i = 0
        title = ensure_str(p.get("title")).lower()
        return (venue, -year_i, title)

    if args.sort == "venue_year_title":
        papers.sort(key=sort_key)
    elif args.sort == "title":
        papers.sort(key=lambda p: ensure_str(p.get("title")).lower())
    elif args.sort == "year":
        def yk(p):
            try:
                return -int(ensure_str(p.get("year")).strip())
            except Exception:
                return 0
        papers.sort(key=yk)

    out_dir = os.path.dirname(args.out_html)
    os.makedirs(out_dir, exist_ok=True)

    report_base_dir = args.report_base_dir.strip() if args.report_base_dir else None
    if report_base_dir:
        report_base_dir = os.path.abspath(report_base_dir)

    html_text = generate_html(papers, args.title, report_base_dir)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"OK: wrote {args.out_html}")
    print(f"Papers: {len(papers)}")
    if report_base_dir:
        print(f"Links relativized to: {report_base_dir}")


if __name__ == "__main__":
    main()