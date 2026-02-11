#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Condition
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests
from tqdm import tqdm

from paper_utils import (
    load_csv,
    load_jsonl,
    write_csv,
    write_jsonl,
    compute_paper_id,
    compute_source_hash,
    sha1_text,
)


# 章节名称映射表（支持模糊匹配）
SECTION_PATTERNS = {
    "abstract": [
        r"^abstract\s*$",
        r"^摘要\s*$",
    ],
    "introduction": [
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*introduction\s*$",
        r"^introduction\s*$",
        r"^引言\s*$",
    ],
    "related_work": [
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*related\s+works?\s*$",
        r"^related\s+works?\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*preliminaries\s*$",
        r"^preliminaries\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*preliminary\s*$",
        r"^preliminary\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*background\s*$",
        r"^background\s*$",
        r"^相关工作\s*$",
    ],
    "methods": [
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*methods?\s*$",
        r"^methods?\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*methodology\s*$",
        r"^methodology\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*approach\s*$",
        r"^approach\s*$",
        r"^方法\s*$",
    ],
    "experiments": [
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*experiments?\s*$",
        r"^experiments?\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*evaluation\s*$",
        r"^evaluation\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*results\s*$",
        r"^results\s*$",
        r"^实验\s*$",
    ],
    "conclusion": [
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*conclusions?\s*$",
        r"^conclusions?\s*$",
        r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*discussion\s*$",
        r"^discussion\s*$",
        r"^总结\s*$",
    ],
}

SECTION_LABELS = {
    "abstract": ["abstract", "摘要"],
    "introduction": ["introduction", "intro", "引言"],
    "related_work": [
        "related work",
        "related works",
        "preliminaries",
        "preliminary",
        "background",
        "相关工作",
    ],
    "methods": ["methods", "method", "methodology", "approach", "方法", "方法论"],
    "experiments": ["experiments", "experiment", "evaluation", "results", "实验", "结果"],
    "conclusion": ["conclusion", "conclusions", "discussion", "总结"],
}


def _normalize_heading_candidate(line: str) -> str:
    """
    将疑似标题行做轻量规范化，增强对 OCR/Markdown 噪声的鲁棒性。
    例：
      "**1. Introduction**" -> "1. Introduction"
      "**1** **Introduction**" -> "1 Introduction"
      "I. INTRODUCTION" -> "I. INTRODUCTION"
      "# 2 Related Work" -> "2 Related Work"
    """
    s = (line or "").strip()
    if not s:
        return ""

    # 常见前缀噪声：引用、列表、标题符号
    s = re.sub(r"^[>\-\+\u2022\s]+", "", s)
    s = s.lstrip("#").strip()

    # 去掉常见 Markdown 强调/代码样式符号（仅用于标题匹配）
    s = s.replace("*", "").replace("_", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[:：]\s*$", "", s).strip()
    return s


def _roman_to_int(roman: str) -> Optional[int]:
    roman = (roman or "").strip().lower()
    if not roman or not re.fullmatch(r"[ivxlcdm]+", roman):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(roman):
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total or None


def _is_probably_heading_line(raw_line: str, norm_line: str) -> bool:
    s = (raw_line or "").strip()
    if not s:
        return False
    if not norm_line:
        return False

    # 避免把正文长句当标题
    if len(norm_line) > 140:
        return False

    if s.startswith("#"):
        return True
    if (s.startswith("**") and s.endswith("**")) or (s.startswith("__") and s.endswith("__")):
        return True
    if norm_line.isupper() and len(norm_line) > 3:
        return True
    if re.match(r"^(?:\d+|[ivxlcdm]+)\s*[\.\)\-–—:]?\s*[A-Za-z]", norm_line, re.IGNORECASE):
        return True
    return False


def _matches_any_pattern(norm_line: str, pattern_strs: List[str]) -> bool:
    for p in pattern_strs:
        if re.match(p, norm_line, re.IGNORECASE):
            return True
    return False


def _matches_section_heading(raw_line: str, norm_line: str, section_name: str) -> Optional[str]:
    patterns = SECTION_PATTERNS.get(section_name, [])
    if _matches_any_pattern(norm_line, patterns):
        return "full"

    if not _is_probably_heading_line(raw_line, norm_line):
        return None

    labels = SECTION_LABELS.get(section_name, [])
    lower = norm_line.lower()
    for label in labels:
        if lower.startswith(label):
            tail = lower[len(label):]
            if not tail:
                return "full"
            if tail[0] in (" ", ".", ":", "：", "-", "–", "—", ")", "）"):
                return "inline"
    return None


@dataclass
class LLMUsage:
    prompt_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    raw: Dict[str, Any]


class LLMProvider:
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""

        usage_meta = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(usage_meta.get("prompt_tokens") or 0),
            output_tokens=int(usage_meta.get("completion_tokens") or 0),
            total_tokens=int(usage_meta.get("total_tokens") or 0),
        )
        return LLMResponse(text=text, model=self.model, usage=usage, raw=data)


class RateLimitController:
    def __init__(self, max_inflight: int, wait_s: int):
        self.default_max = max(1, max_inflight)
        self.max_inflight = self.default_max
        self.in_flight = 0
        self.pause_until = 0.0
        self.rate_limited = False
        self.wait_s = max(1, int(wait_s))
        self.waiting_logged = False
        self.cond = Condition()

    def acquire(self) -> None:
        with self.cond:
            while True:
                now = time.time()
                if self.pause_until and now < self.pause_until:
                    wait_s = self.pause_until - now
                    if not self.waiting_logged:
                        self.waiting_logged = True
                        tqdm.write(
                            f"[rate-limit] waiting {int(wait_s)}s, concurrency={self.max_inflight}"
                        )
                    self.cond.wait(timeout=wait_s)
                    continue
                if self.pause_until and now >= self.pause_until:
                    self.pause_until = 0.0
                    self.waiting_logged = False
                if self.in_flight < self.max_inflight:
                    self.in_flight += 1
                    return
                self.cond.wait()

    def release(self) -> None:
        with self.cond:
            self.in_flight = max(0, self.in_flight - 1)
            self.cond.notify_all()

    def on_rate_limit(self) -> None:
        with self.cond:
            self.rate_limited = True
            self.max_inflight = 1
            now = time.time()
            self.pause_until = max(self.pause_until, now + self.wait_s)
            self.waiting_logged = False
            self.cond.notify_all()
            tqdm.write(
                f"[rate-limit] 429 detected, pause {self.wait_s}s, "
                f"concurrency set to 1"
            )

    def on_success(self) -> None:
        with self.cond:
            if self.rate_limited and time.time() >= self.pause_until:
                self.rate_limited = False
                self.max_inflight = self.default_max
                self.cond.notify_all()
                tqdm.write(
                    f"[rate-limit] recovered, concurrency={self.default_max}"
                )


def normalize_section_name(section: str) -> str:
    """将用户输入的章节名称标准化"""
    section = section.strip().lower()
    # 直接匹配已知的章节类型
    for key in SECTION_PATTERNS.keys():
        if key == section or key.replace("_", " ") == section:
            return key
    # 模糊匹配
    if "abstract" in section:
        return "abstract"
    if "intro" in section:
        return "introduction"
    if "related" in section or "preliminary" in section or "background" in section:
        return "related_work"
    if "method" in section or "approach" in section:
        return "methods"
    if "experiment" in section or "evaluation" in section or "result" in section:
        return "experiments"
    if "conclusion" in section or "discussion" in section:
        return "conclusion"
    return section


def find_section_in_text(text: str, section_name: str) -> Optional[Tuple[int, int]]:
    """在文本中查找章节的起始和结束位置"""
    section_name = normalize_section_name(section_name)
    patterns = SECTION_PATTERNS.get(section_name, [])
    
    if not patterns:
        # 如果没有预定义模式，尝试直接搜索
        patterns = [rf"^{re.escape(section_name)}\s*$"]
    
    lines = text.split("\n")
    norm_lines = [_normalize_heading_candidate(l) for l in lines]
    section_start = None
    section_end = None
    
    # 查找章节开始位置
    for i, norm in enumerate(norm_lines):
        if not norm:
            continue
        match_type = _matches_section_heading(lines[i], norm, section_name)
        if match_type:
            section_start = i if match_type == "inline" else i + 1
            break
        if not SECTION_PATTERNS.get(section_name):
            if _matches_any_pattern(norm, patterns):
                section_start = i + 1  # 下一行开始
                break
    
    if section_start is None:
        return None
    
    # 查找章节结束位置（下一章节标题）
    # 1) 优先：匹配到任何其他已知章节标题
    # 2) 兜底（针对 introduction）：遇到主编号 > 1 的标题（如 "2. ..." 或 "II. ..."）即截断
    for i in range(section_start, len(lines)):
        norm = norm_lines[i]
        if not norm:
            continue
        if not _is_probably_heading_line(lines[i], norm):
            continue

        for other_section in SECTION_PATTERNS.keys():
            if other_section == section_name:
                continue
            match_type = _matches_section_heading(lines[i], norm, other_section)
            if match_type:
                section_end = i
                return (section_start, section_end)

        if section_name == "introduction":
            m_num = re.match(r"^(?P<n>\d+)(?:\.\d+)*\s*[\.\)\-–—:]?\s+\S", norm)
            if m_num:
                try:
                    if int(m_num.group("n")) > 1:
                        section_end = i
                        return (section_start, section_end)
                except ValueError:
                    pass
            m_rom = re.match(r"^(?P<r>[ivxlcdm]+)\s*[\.\)\-–—:]?\s+\S", norm, re.IGNORECASE)
            if m_rom:
                v = _roman_to_int(m_rom.group("r"))
                if v is not None and v > 1:
                    section_end = i
                    return (section_start, section_end)
    
    # 如果没有找到下一个章节，返回到文本末尾
    section_end = len(lines)
    return (section_start, section_end)


def extract_sections_from_paper(
    paper: Dict[str, Any],
    section_names: List[str],
) -> Dict[str, str]:
    """从论文中提取指定章节的内容"""
    result = {}

    def wants_full_text(names: List[str]) -> bool:
        for name in names:
            if not name:
                continue
            raw = str(name).strip()
            if not raw:
                continue
            lower = raw.lower()
            if lower in {"full_text", "full", "all", "*"}:
                return True
            if raw in {"全文", "全部"}:
                return True
        return False
    
    # 优先从abstract字段获取摘要
    if "abstract" in section_names:
        abstract = paper.get("abstract") or ""
        if abstract:
            result["abstract"] = abstract

    md_pages = paper.get("md_pages") or []
    md_text = (paper.get("md_text") or "").strip()
    full_text = "\n".join([page.get("text", "") for page in md_pages]).strip()

    if wants_full_text(section_names):
        if full_text:
            result["full_text"] = full_text
            return result
        if md_text:
            result["full_text"] = md_text
            return result
        # OCR 不可用时兜底，至少返回摘要避免 no_sections。
        abstract = (paper.get("abstract") or "").strip()
        if abstract:
            result["full_text"] = abstract
        return result

    # 从md_pages/md_text中提取其他章节
    if not full_text and not md_text:
        return result
    full_text = full_text or md_text
    
    for section_name in section_names:
        if section_name == "abstract" and "abstract" in result:
            continue  # 已经处理过了
        
        section_range = find_section_in_text(full_text, section_name)
        if section_range:
            start, end = section_range
            lines = full_text.split("\n")
            section_text = "\n".join(lines[start:end]).strip()
            if section_text:
                result[section_name] = section_text
    
    return result


def _parse_creators(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict) or isinstance(v, str)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [v for v in parsed if isinstance(v, dict) or isinstance(v, str)]
        except Exception:
            return []
    return []


def _format_creators(creators: Any) -> str:
    parsed = _parse_creators(creators)
    names: List[str] = []
    for c in parsed:
        if isinstance(c, str):
            name = c.strip()
            if name:
                names.append(name)
            continue
        first = (c.get("firstName") or c.get("first_name") or "").strip()
        last = (c.get("lastName") or c.get("last_name") or "").strip()
        name = " ".join([x for x in [first, last] if x]).strip()
        if not name:
            name = (c.get("name") or "").strip()
        if name:
            names.append(name)
    return ", ".join(names)


def _pick_venue(paper: Dict[str, Any]) -> str:
    for key in (
        "venue",
        "conference_name",
        "proceedings_title",
        "publication_title",
        "book_title",
        "series",
    ):
        val = (paper.get(key) or "").strip()
        if val:
            return val
    return ""


def _collect_body_text(
    paper: Dict[str, Any],
    section_names: List[str],
    max_body_length: int,
) -> Tuple[str, List[str]]:
    sections = extract_sections_from_paper(paper, section_names)
    if not sections:
        return "", []
    if "full_text" in sections:
        body = sections.get("full_text") or ""
        found = ["full_text"]
    else:
        ordered: List[str] = []
        for name in section_names:
            if name in sections and sections[name]:
                ordered.append(sections[name])
        if not ordered:
            ordered = [v for v in sections.values() if v]
        body = "\n\n".join(ordered).strip()
        found = [k for k in sections.keys()]

    if max_body_length and len(body) > max_body_length:
        body = body[:max_body_length] + "..."
    return body, found


def build_structured_prompt(
    question: str,
    papers: List[Dict[str, Any]],
) -> str:
    """构造结构化 Markdown prompt（不包含章节信息）"""
    lines: List[str] = []
    lines.append("# Query")
    lines.append(f"问题: {question}")
    lines.append("")
    lines.append("# Papers")
    for idx, p in enumerate(papers, 1):
        lines.append(f"## Paper {idx}")
        lines.append(f"标题: {p.get('title') or ''}")
        lines.append(f"作者: {p.get('authors') or ''}")
        lines.append(f"会议/期刊: {p.get('venue') or ''}")
        lines.append(f"年份: {p.get('year') or ''}")
        lines.append(f"DOI: {p.get('doi') or ''}")
        lines.append(f"URL: {p.get('url') or ''}")
        lines.append("正文:")
        lines.append(p.get("body") or "")
        lines.append("")
    lines.append("请基于上述论文内容回答问题。如果论文中没有相关信息，请说明。")
    return "\n".join(lines)


DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_CONCURRENCY = 5
DEFAULT_RETRY_ON_429 = False
DEFAULT_RETRY_WAIT_S = 300


def _load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _pick_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str) and value.strip() == "":
        return fallback
    return value


def _pick_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _pick_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except Exception:
        return fallback


def _pick_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


def main() -> None:
    ap = argparse.ArgumentParser(
        description="基于论文内容进行LLM查询",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查询abstract和introduction章节
  python query_papers.py --sections abstract,introduction --question "这篇论文的主要贡献是什么？"
  
  # 查询methods章节
  python query_papers.py --sections methods --question "这篇论文使用了什么方法？"
  
  # 限制只查询前5篇论文
  python query_papers.py --sections abstract --question "..." --top_k 5
        """
    )
    
    ap.add_argument("--pages_jsonl", default="./store/ocr/papers.pages.jsonl",
                    help="论文页面JSONL文件路径")
    ap.add_argument("--out_jsonl", default="papers.query.jsonl",
                    help="输出JSONL文件路径")
    ap.add_argument("--out_csv", default="papers.query.csv",
                    help="输出CSV文件路径")
    ap.add_argument("--out_md", default="papers.query.md",
                    help="输出Markdown文件路径")
    ap.add_argument("--issues_out", default="papers.query_issues.csv",
                    help="错误报告CSV文件路径")
    ap.add_argument("--progress_path", default="",
                    help="进度文件路径（可选，JSON格式，包含done/total）")
    ap.add_argument("--base_output_dir", default="./store/query",
                    help="输出根目录（相对路径会拼接到该目录）")
    
    ap.add_argument("--sections", required=True,
                    help="要查询的章节，用逗号分隔，如: abstract,introduction,methods")
    ap.add_argument("--question", required=True,
                    help="要查询的问题")
    ap.add_argument("--system_prompt", default="",
                    help="系统提示词（可选）")
    ap.add_argument("--max_section_length", type=int, default=8000,
                    help="正文最大长度（字符数，按每篇论文截断）")
    
    ap.add_argument("--config", default="",
                    help="LLM配置文件（JSON，可选）")
    ap.add_argument("--model", default=None,
                    help="LLM模型名称")
    ap.add_argument("--base_url", default=None,
                    help="API基础URL")
    ap.add_argument("--api_key", default=None,
                    help="API密钥（也可通过环境变量SILICONFLOW_API_KEY或OPENAI_API_KEY设置）")
    
    ap.add_argument("--temperature", type=float, default=None,
                    help="生成温度")
    ap.add_argument("--max_output_tokens", type=int, default=None,
                    help="最大输出token数")
    
    ap.add_argument("--resume", action="store_true", default=False,
                    help="从已有结果继续（跳过已处理的论文）")
    ap.add_argument("--resume_from", default="",
                    help="从指定文件继续（默认使用out_jsonl）")
    ap.add_argument("--filter_title", default="",
                    help="按标题关键字筛选论文（大小写不敏感，如：data）")
    ap.add_argument("--top_k", type=int, default=0,
                    help="只处理前K篇论文（0=全部）")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="并发请求数量")
    ap.add_argument("--retry_on_429", action="store_true", default=None,
                    help="遇到429时持续重试")
    ap.add_argument("--retry_wait_s", type=int, default=None,
                    help="429重试等待秒数")
    ap.add_argument("--query_mode", default="single",
                    help="查询模式：single 或 merge")
    
    args = ap.parse_args()
    
    def apply_base_dir(path: str, base_dir: str) -> str:
        if not base_dir or os.path.isabs(path):
            return path
        base_norm = os.path.normpath(base_dir)
        path_norm = os.path.normpath(path)
        if path_norm == base_norm or path_norm.startswith(base_norm + os.sep):
            return path
        return os.path.join(base_dir, path)
    
    if args.base_output_dir:
        os.makedirs(args.base_output_dir, exist_ok=True)
        args.out_jsonl = apply_base_dir(args.out_jsonl, args.base_output_dir)
        args.out_csv = apply_base_dir(args.out_csv, args.base_output_dir)
        args.out_md = apply_base_dir(args.out_md, args.base_output_dir)
        args.issues_out = apply_base_dir(args.issues_out, args.base_output_dir)
        if args.progress_path:
            args.progress_path = apply_base_dir(args.progress_path, args.base_output_dir)

    def write_progress(done: int, total: int):
        if not args.progress_path:
            return
        payload = {
            "done": done,
            "total": total,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(args.progress_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass
    
    config_path = args.config or os.environ.get("PAPERVIEW_LLM_CONFIG", "")
    config = _load_config(config_path)

    args.model = _pick_value(args.model, _pick_value(config.get("model"), DEFAULT_MODEL))
    args.base_url = _pick_value(args.base_url, _pick_value(config.get("base_url"), DEFAULT_BASE_URL))
    args.temperature = _pick_float(
        args.temperature, _pick_float(config.get("temperature"), DEFAULT_TEMPERATURE)
    )
    args.max_output_tokens = _pick_int(
        args.max_output_tokens,
        _pick_int(config.get("max_output_tokens"), DEFAULT_MAX_OUTPUT_TOKENS),
    )
    args.concurrency = _pick_int(
        args.concurrency, _pick_int(config.get("concurrency"), DEFAULT_CONCURRENCY)
    )
    args.retry_on_429 = _pick_bool(
        args.retry_on_429, _pick_bool(config.get("retry_on_429"), DEFAULT_RETRY_ON_429)
    )
    args.retry_wait_s = _pick_int(
        args.retry_wait_s, _pick_int(config.get("retry_wait_s"), DEFAULT_RETRY_WAIT_S)
    )

    # 解析章节列表
    section_names = [s.strip() for s in args.sections.split(",") if s.strip()]
    if not section_names:
        raise SystemExit("错误: 必须指定至少一个章节")

    query_mode = (args.query_mode or "single").strip().lower()
    if query_mode not in {"single", "merge"}:
        raise SystemExit("错误: query_mode 仅支持 single 或 merge")
    
    # 加载论文数据
    if not os.path.exists(args.pages_jsonl):
        raise SystemExit(f"错误: 文件不存在: {args.pages_jsonl}")
    
    papers = load_jsonl(args.pages_jsonl)
    if not papers:
        raise SystemExit(f"错误: 没有找到论文数据")
    
    # 按标题关键字筛选
    if args.filter_title:
        filter_keyword = args.filter_title.lower()
        original_count = len(papers)
        papers = [
            p for p in papers
            if filter_keyword in (p.get("title") or "").lower()
        ]
        print(f"标题筛选 '{args.filter_title}': {original_count} -> {len(papers)} 篇论文")
    
    if not papers:
        raise SystemExit("错误: 筛选后没有符合条件的论文")
    
    # 限制处理数量
    if args.top_k and args.top_k > 0:
        papers = papers[:args.top_k]
        print(f"限制处理前 {args.top_k} 篇论文")
    
    print(f"共 {len(papers)} 篇论文待处理")
    print(f"查询章节: {', '.join(section_names)}")
    print(f"查询问题: {args.question}")
    print(f"查询模式: {query_mode}")
    
    # 初始化LLM provider
    api_key = (
        _pick_value(args.api_key, _pick_value(config.get("api_key"), ""))
        or os.environ.get("SILICONFLOW_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        raise SystemExit("错误: 缺少API密钥，请设置SILICONFLOW_API_KEY/OPENAI_API_KEY环境变量或使用--api_key参数")
    
    provider = OpenAICompatProvider(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
    )
    
    # 加载已有结果（用于resume）
    existing_idx: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        resume_path = args.resume_from or args.out_jsonl
        if os.path.exists(resume_path):
            existing_papers = load_jsonl(resume_path)
            for p in existing_papers:
                pid = p.get("paper_id")
                if pid:
                    existing_idx[pid] = p
            print(f"从 {resume_path} 加载了 {len(existing_idx)} 条已有结果")
    
    rate_state = RateLimitController(args.concurrency, args.retry_wait_s)

    def build_paper_payload(
        paper: Dict[str, Any],
        section_names: List[str],
        max_body_length: int,
    ) -> Tuple[Optional[Dict[str, Any]], List[str], str]:
        body, found_sections = _collect_body_text(paper, section_names, max_body_length)
        if not body:
            return None, [], ""
        payload = {
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title", ""),
            "authors": _format_creators(paper.get("creators"))
            or (paper.get("authors") or paper.get("author") or ""),
            "venue": _pick_venue(paper),
            "year": paper.get("year") or "",
            "doi": paper.get("doi") or "",
            "url": paper.get("url") or paper.get("paper_url") or paper.get("page_url") or "",
            "body": body,
        }
        return payload, found_sections, body
    
    def process_one(idx: int, paper: Dict[str, Any]) -> Dict[str, Any]:
        paper_id = paper.get("paper_id")
        if not paper_id:
            return {
                "idx": idx,
                "data": {
                    "paper_id": None,
                    "title": paper.get("title", ""),
                    "query_status": "error",
                    "query_error": "缺少paper_id",
                }
            }
        
        # 检查是否已处理
        if args.resume and paper_id in existing_idx:
            existing = existing_idx[paper_id]
            # 检查是否已有成功的查询结果
            if existing.get("query_status") == "ok":
                return {"idx": idx, "data": existing}
        
        out = dict(paper)
        out["query_question"] = args.question
        out["query_sections"] = ",".join(section_names)
        out["query_status"] = "init"
        out["query_error"] = None
        out["query_response"] = None
        out["query_usage"] = None
        out["query_model"] = None
        out["query_elapsed_ms"] = None
        
        start_ts = time.time()
        try:
            payload, found_sections, body = build_paper_payload(
                paper, section_names, args.max_section_length
            )

            if not payload or not body:
                out["query_status"] = "no_sections"
                out["query_error"] = f"未找到指定章节: {', '.join(section_names)}"
                out["query_elapsed_ms"] = int((time.time() - start_ts) * 1000)
                return {"idx": idx, "data": out}
            
            # 构造结构化prompt
            prompt = build_structured_prompt(
                question=args.question,
                papers=[payload],
            )
            
            # 调用LLM
            retries = 0
            
            while True:
                rate_state.acquire()
                try:
                    resp = provider.generate(
                        prompt=prompt,
                        system=args.system_prompt if args.system_prompt else None,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                    rate_state.on_success()
                    break
                except requests.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 429 and args.retry_on_429:
                        retries += 1
                        rate_state.on_rate_limit()
                        continue
                    raise
                finally:
                    rate_state.release()
            
            elapsed_ms = int((time.time() - start_ts) * 1000)
            
            out["query_status"] = "ok"
            out["query_response"] = resp.text.strip()
            out["query_usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            out["query_model"] = resp.model
            out["query_elapsed_ms"] = elapsed_ms
            out["query_sections_found"] = found_sections
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_ts) * 1000)
            out["query_status"] = "error"
            out["query_error"] = str(e)
            out["query_elapsed_ms"] = elapsed_ms
        
        return {"idx": idx, "data": out}

    def process_merge(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        write_progress(0, len(papers))
        merged_papers: List[Dict[str, Any]] = []
        prompt_papers: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        found_union: List[str] = []

        for paper in papers:
            payload, found_sections, body = build_paper_payload(
                paper, section_names, args.max_section_length
            )
            if not payload or not body:
                skipped.append({
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title", ""),
                    "reason": "no_sections",
                })
                continue
            prompt_papers.append(payload)
            merged_papers.append({
                "paper_id": payload.get("paper_id"),
                "title": payload.get("title"),
                "authors": payload.get("authors"),
                "venue": payload.get("venue"),
                "year": payload.get("year"),
                "doi": payload.get("doi"),
                "url": payload.get("url"),
            })
            for s in found_sections:
                if s not in found_union:
                    found_union.append(s)

        out: Dict[str, Any] = {}
        merge_id = sha1_text(args.question + "|" + ",".join(sorted([
            str(p.get("paper_id") or "") for p in papers
        ])))
        out["paper_id"] = f"merge_{merge_id}"
        out["title"] = f"Concat Query ({len(merged_papers)} papers)"
        out["query_question"] = args.question
        out["query_sections"] = ",".join(section_names)
        out["query_status"] = "init"
        out["query_error"] = None
        out["query_response"] = None
        out["query_usage"] = None
        out["query_model"] = None
        out["query_elapsed_ms"] = None
        out["merged_papers"] = merged_papers
        out["merge_skipped"] = skipped

        if not prompt_papers:
            out["query_status"] = "no_sections"
            out["query_error"] = f"未找到指定章节: {', '.join(section_names)}"
            out["query_sections_found"] = []
            write_progress(len(papers), len(papers))
            return [out]

        start_ts = time.time()
        try:
            prompt = build_structured_prompt(
                question=args.question,
                papers=prompt_papers,
            )

            while True:
                rate_state.acquire()
                try:
                    resp = provider.generate(
                        prompt=prompt,
                        system=args.system_prompt if args.system_prompt else None,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                    rate_state.on_success()
                    break
                except requests.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 429 and args.retry_on_429:
                        rate_state.on_rate_limit()
                        continue
                    raise
                finally:
                    rate_state.release()

            elapsed_ms = int((time.time() - start_ts) * 1000)
            out["query_status"] = "ok"
            out["query_response"] = resp.text.strip()
            out["query_usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            out["query_model"] = resp.model
            out["query_elapsed_ms"] = elapsed_ms
            out["query_sections_found"] = found_union
        except Exception as e:
            elapsed_ms = int((time.time() - start_ts) * 1000)
            out["query_status"] = "error"
            out["query_error"] = str(e)
            out["query_elapsed_ms"] = elapsed_ms

        write_progress(len(papers), len(papers))
        return [out]

    # 处理所有论文
    if query_mode == "merge":
        results = process_merge(papers)
    else:
        results: List[Dict[str, Any]] = [None] * len(papers)
        write_progress(0, len(papers))
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futures = [ex.submit(process_one, i, paper) for i, paper in enumerate(papers)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="查询论文", unit="篇"):
                res = fut.result()
                results[res["idx"]] = res["data"]
                done = sum(1 for r in results if r is not None)
                write_progress(done, len(papers))
        write_progress(len(papers), len(papers))
    
    # 保存结果
    write_jsonl(args.out_jsonl, results)
    
    # 生成CSV摘要
    summary_rows = []
    issues = []
    
    for r in results:
        status = r.get("query_status", "unknown")
        if status not in {"ok", "no_sections"}:
            issues.append({
                "paper_id": r.get("paper_id"),
                "title": r.get("title", "")[:100],
                "query_status": status,
                "query_error": r.get("query_error", "")[:200],
            })
        
        usage = r.get("query_usage") or {}
        summary_rows.append({
            "paper_id": r.get("paper_id"),
            "title": r.get("title", "")[:200],
            "arxiv_id": r.get("arxiv_id", ""),
            "query_question": r.get("query_question", ""),
            "query_sections": r.get("query_sections", ""),
            "query_sections_found": ",".join(r.get("query_sections_found", [])),
            "query_response": (r.get("query_response") or "")[:1000],
            "query_status": status,
            "query_error": (r.get("query_error") or "")[:200],
            "query_model": r.get("query_model", ""),
            "query_elapsed_ms": r.get("query_elapsed_ms"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
    
    preferred = [
        "paper_id", "title", "arxiv_id",
        "query_question", "query_sections", "query_sections_found",
        "query_response", "query_status", "query_error",
        "query_model", "query_elapsed_ms",
        "prompt_tokens", "output_tokens", "total_tokens",
    ]
    write_csv(args.out_csv, summary_rows, preferred_fields=preferred)
    
    # 生成Markdown报告
    md_lines = []
    md_lines.append("# 论文查询结果\n\n")
    md_lines.append(f"**查询问题**: {args.question}\n\n")
    md_lines.append(f"**查询章节**: {', '.join(section_names)}\n\n")
    md_lines.append(f"**处理时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md_lines.append("---\n\n")
    
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        arxiv_id = r.get("arxiv_id", "")
        status = r.get("query_status", "unknown")
        response = r.get("query_response", "")
        error = r.get("query_error", "")
        sections_found = r.get("query_sections_found", [])
        usage = r.get("query_usage") or {}
        
        md_lines.append(f"## {i}. {title}\n\n")
        if arxiv_id:
            md_lines.append(f"**ArXiv ID**: {arxiv_id}\n\n")
        md_lines.append(f"**状态**: {status}\n\n")
        if sections_found:
            md_lines.append(f"**找到章节**: {', '.join(sections_found)}\n\n")
        
        if status == "ok" and response:
            md_lines.append("### 回答\n\n")
            md_lines.append(f"{response}\n\n")
        elif error:
            md_lines.append("### 错误\n\n")
            md_lines.append(f"```\n{error}\n```\n\n")
        
        if usage.get("total_tokens"):
            md_lines.append(f"**Token使用**: {usage.get('total_tokens')} (prompt: {usage.get('prompt_tokens')}, output: {usage.get('output_tokens')})\n\n")
        
        md_lines.append("---\n\n")
    
    # 写入Markdown文件
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    
    if issues:
        write_csv(args.issues_out, issues, preferred_fields=["paper_id", "title", "query_status", "query_error"])
    
    # 统计信息
    status_counts = {}
    total_tokens = 0
    for r in results:
        status = r.get("query_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        usage = r.get("query_usage") or {}
        total_tokens += usage.get("total_tokens", 0)
    
    print(f"\n处理完成!")
    print(f"结果JSONL: {args.out_jsonl}")
    print(f"结果CSV: {args.out_csv}")
    print(f"结果Markdown: {args.out_md}")
    if issues:
        print(f"错误报告: {args.issues_out}")
    print(f"\n状态统计:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\n总token消耗: {total_tokens:,}")


if __name__ == "__main__":
    main()
