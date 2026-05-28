# -*- coding: utf-8 -*-
import base64
import hashlib
import mimetypes
import shutil
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import List, Optional, Union, Iterable

# -------------------------
# 数据结构
# -------------------------
@dataclass
class Sample:
    text: Optional[str] = None
    audio: Optional[str] = None  # 本地路径或 URL

@dataclass
class Block:
    description: Optional[str] = None
    items: List[Union[Sample, str, dict]] = field(default_factory=list)
    # 说明：
    # - 推荐传 List[Sample]
    # - 也支持 List[str]（仅音频，文本为空）或 List[dict]（包含 "audio"/"text" 键）

@dataclass
class ReportData:
    notes: Optional[str] = None
    blocks: List[Block] = field(default_factory=list)
    page_title: str = "PromptTTS Inference Results"

# -------------------------
# 工具函数
# -------------------------
def _is_url(s: str) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return s.startswith(("http://", "https://", "data:", "blob:", "file:"))

def _guess_mime(path: Union[str, Path]) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "audio/wav"

def _sha1_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _to_data_uri(path: Path) -> str:
    mime = _guess_mime(path)
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _copy_with_hash(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    sha1 = _sha1_of_file(src)[:10]
    dst = dst_dir / f"{src.stem}.{sha1}{src.suffix}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst

def _media_src(
    path_or_url: Optional[str],
    out_html_path: Optional[Path],
    media_mode: str,
    assets_dirname: str,
) -> Optional[str]:
    """
    返回 <audio src="..."> 的值：
    - inline:   data:URI
    - copy:     复制到 assets 下并返回相对路径
    - link:     本地用 file:// 绝对 URI；URL 原样返回
    """
    if not path_or_url:
        return None
    if _is_url(path_or_url):
        return path_or_url.strip()

    src_path = Path(path_or_url).expanduser()
    if not src_path.exists():
        # 不存在则原样写入（可能是相对路径）
        return str(path_or_url)

    media_mode = media_mode.lower()
    if media_mode == "inline":
        return _to_data_uri(src_path)
    if media_mode == "copy":
        if out_html_path is None:
            # 未指定输出文件路径时无法复制，退化为绝对 URI
            return src_path.resolve().as_uri()
        out_dir = out_html_path.parent
        assets_dir = out_dir / assets_dirname
        copied = _copy_with_hash(src_path, assets_dir)
        return str(copied.relative_to(out_dir).as_posix())
    return src_path.resolve().as_uri()

def _coerce_items(items: Iterable[Union[Sample, str, dict]]) -> List[Sample]:
    """将输入统一为 List[Sample]。"""
    out: List[Sample] = []
    for x in items:
        if isinstance(x, Sample):
            out.append(x)
        elif isinstance(x, str):
            out.append(Sample(audio=x))
        elif isinstance(x, dict):
            out.append(Sample(text=x.get("text"), audio=x.get("audio")))
        else:
            raise TypeError(f"Unsupported item type: {type(x)}")
    return out

# -------------------------
# PromptTTS 主函数
# -------------------------
def generate_html_prompttts(
    data: ReportData,
    out_html: Union[str, Path, None] = None,
    media_mode: str = "link",       # "copy" | "inline" | "link"
    assets_dirname: str = "assets", # 仅 "copy" 模式使用
    per_row: int = 5
) -> Union[str, Path]:
    """
    生成适用于 PromptTTS 的静态 HTML 页面：
    - 不展示 reference 音频；
    - 每行展示 4 个样本；
    - 每个单元格内上方为音频播放器，下方为文本描述；
    - 文本框设定最大高度，超出内容可滚动。
    """
    out_html_path: Optional[Path] = None
    if out_html is not None:
        out_html_path = Path(out_html)
        out_html_path.parent.mkdir(parents=True, exist_ok=True)

    # 样式（纯展示，无交互）
    css = """
    :root {
      --bg: #f7f7f9; --fg: #222; --muted: #666;
      --card: #fff; --border: #e5e7eb; --radius: 10px;
    }
    html, body {
      margin: 0; padding: 0; background: var(--bg); color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC",
                   "Hiragino Sans GB", "Microsoft YaHei", system-ui, Arial, sans-serif;
      line-height: 1.5;
    }
    .page { max-width: 1200px; margin: 20px auto 60px; padding: 0 16px; }
    .notes {
      background: var(--card); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 12px; margin-bottom: 12px;
    }
    .notes .label { font-size: 13px; color: var(--muted); margin-bottom: 6px; }
    .notes .box {
      width: 100%; min-height: 120px; box-sizing: border-box;
      border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
      background: #fff; font-size: 14px; overflow: auto; white-space: pre-wrap; word-break: break-word;
    }
    .block {
      background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
      margin-top: 16px; padding: 12px; box-shadow: 0 1px 0 rgba(0,0,0,0.02);
    }
    .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    /*
    colgroup col { width: 20%; }
    */
    thead th {
      position: sticky; top: 0; background: #fafafa; border-bottom: 1px solid var(--border);
      font-weight: 600; font-size: 13px; padding: 8px; text-align: left;
    }
    tbody td { border-top: 1px solid var(--border); padding: 8px; vertical-align: top; }
    audio { width: 100%; max-width: 100%; height: 36px; }
    .item-cell {
      display: flex; flex-direction: column; gap: 8px;
    }
    .text-box {
      width: 100%;
      min-height: 120px;          /* 默认更大一些 */
      max-height: 280px;          /* 最大高度，超出滚动 */
      border: 1px solid var(--border); border-radius: 6px;
      padding: 8px; background: #fff; box-sizing: border-box;
      overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 14px;
    }
    .muted { color: var(--muted); font-size: 12px; }
    """

    def render_text_box(text: Optional[str]) -> str:
        content = "" if text is None else html_escape(str(text))
        return f'<div class="text-box" role="textbox" aria-readonly="true">{content}</div>'

    def render_item_cell(text: Optional[str], audio_src: Optional[str]) -> str:
        # 单元格：上一行音频播放器，下一行文本框
        parts = ['<div class="item-cell">']
        if audio_src:
            parts.append(f'<audio controls preload="none" src="{html_escape(audio_src)}"></audio>')
        else:
            parts.append('<div class="muted">（无音频）</div>')
        parts.append(render_text_box(text))
        parts.append('</div>')
        return "".join(parts)

    # 组装 HTML
    html_parts: List[str] = []
    ap = html_parts.append
    ap("<!DOCTYPE html>")
    ap('<html lang="zh-CN">')
    ap("<head>")
    ap('<meta charset="utf-8" />')
    ap('<meta name="viewport" content="width=device-width,initial-scale=1" />')
    ap(f'<title>{html_escape(data.page_title)}</title>')
    ap("<style>")
    ap(css)
    ap("</style>")
    ap("</head>")
    ap("<body>")
    ap('<div class="page">')

    # 备注
    if (data.notes or "").strip():
        notes_html = html_escape(data.notes or "")
        ap('<section class="notes">')
        ap('<div class="label">实验参数 / 备注</div>')
        ap(f'<div class="box">{notes_html}</div>')
        ap("</section>")

    # 各 Block（不展示 reference）
    for block_idx, block in enumerate(data.blocks):
        items = _coerce_items(block.items)
        ap('<section class="block">')
        ap(f'<div class="label">{block.description if block.description is not None else ""}</div>')

        # 表格（每行 4 个样本）
        ap('<div class="table-wrap">')
        ap("<table>")
        ap("<colgroup>" + "".join("<col />" for _ in range(per_row)) + "</colgroup>")
        ap("<thead><tr>" + "".join(f"<th>样本 {k}</th>" for k in range(1, per_row + 1)) + "</tr></thead>")
        ap("<tbody>")

        for i in range(0, len(items), per_row):
            ap("<tr>")
            for j in range(per_row):
                sample = items[i + j] if (i + j) < len(items) else None
                if sample:
                    audio_src = _media_src(sample.audio, out_html_path, media_mode, assets_dirname)
                    ap(f"<td>{render_item_cell(sample.text, audio_src)}</td>")
                else:
                    # 空单元格占位
                    ap("<td></td>")
            ap("</tr>")

        ap("</tbody></table></div>")  # .table-wrap
        ap("</section>")  # .block

    ap("</div>")  # .page
    ap("</body></html>")

    if out_html is not None:
        out_html_path.write_text("\n".join(html_parts), encoding="utf-8")
        return out_html_path
    return "\n".join(html_parts)

# -------------------------
# 示例
# -------------------------
if __name__ == "__main__":
    # 示例：PromptTTS（纯文本 -> 音频）
    blk = Block(
        items=[
            Sample(text="一个温柔女声，轻声读出：欢迎来到我们的节目。", audio="examples/out/prompt_case1.wav"),
            Sample(text="中年男声，带有少许沙哑，慢速朗读新闻标题。", audio="examples/out/prompt_case2.wav"),
            Sample(text="童声，活泼欢快，读数：一二三四五。", audio="examples/out/prompt_case3.wav"),
            Sample(text="旁白风格，平稳清晰，描述天气：今天局部地区有阵雨。", audio="examples/out/prompt_case4.wav"),
            Sample(text="充满激情的主持人风格，高能量语气，介绍比赛结果。", audio="examples/out/prompt_case5.wav"),
            Sample(text="轻柔的睡前故事语调，缓慢舒适。", audio="examples/out/prompt_case6.wav"),
            Sample(text="播客风格，中速自然，带有口语化停顿与连读。", audio="examples/out/prompt_case7.wav"),
            Sample(text="纯音乐哼唱风格（如果模型支持），轻快节奏。", audio="examples/out/prompt_case8.wav"),
        ],
    )

    report = ReportData(
        notes=(
            "模型: prompttts-v1\n"
            "采样率: 24000 Hz\n"
            "推理参数: top_p=0.9, temperature=0.7\n"
            "备注: 每行展示 4 个样本；单元格上方为音频播放器，下方为文本；文本框有最大高度。"
        ),
        blocks=[blk],
        page_title="PromptTTS Inference",
    )

    out_file = generate_html_prompttts(
        data=report,
        out_html=Path("report/prompttts_report.html"),
        media_mode="copy",        # "copy" | "inline" | "link"
        assets_dirname="assets",
    )
    print(f"Wrote: {out_file}")
