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
    ref_audio: Optional[str] = None
    items: List[Union[Sample, str, dict]] = field(default_factory=list)
    # 说明：
    # - 推荐传 List[Sample]
    # - 也支持 List[str]（仅音频，文本为空）或 List[dict]（包含 "audio"/"text" 键）

@dataclass
class ReportData:
    notes: Optional[str] = None
    blocks: List[Block] = field(default_factory=list)
    page_title: str = "TTS Inference Results"

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
    out_html_path: Path,
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
# 主函数
# -------------------------
def generate_html(
    data: ReportData,
    out_html: Union[str, Path, None] = None,
    media_mode: str = "link",       # "copy" | "inline" | "link"
    assets_dirname: str = "assets", # 仅 "copy" 模式使用
) -> Path:
    """
    生成静态 HTML 页面：
    - 每个 Block 的 items 两两配对生成表格行；
    - 奇数个样本时，最后一行的右侧两列为空。
    """
    out_html_path = None
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
    .ref {
      display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; align-items: center; margin-bottom: 10px;
    }
    .ref .label { font-size: 13px; color: var(--muted); white-space: nowrap; }
    audio { width: 320px; max-width: 100%; height: 32px; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    colgroup col { width: 25%; }
    thead th {
      position: sticky; top: 0; background: #fafafa; border-bottom: 1px solid var(--border);
      font-weight: 600; font-size: 13px; padding: 8px; text-align: left;
    }
    tbody td { border-top: 1px solid var(--border); padding: 8px; vertical-align: top; }
    .text-box {
      width: 100%; height: 100px; border: 1px solid var(--border); border-radius: 6px;
      padding: 8px; background: #fff; overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 14px;
    }
    .muted { color: var(--muted); font-size: 12px; }
    """

    def render_text_box(text: Optional[str]) -> str:
        content = "" if text is None else html_escape(str(text))
        return f'<div class="text-box" role="textbox" aria-readonly="true">{content}</div>'

    def render_audio_cell(src: Optional[str]) -> str:
        if src:
            return f'<audio controls preload="none" src="{html_escape(src)}"></audio>'
        # 右侧为空时，返回空内容（占位由单元格 padding/布局保证）
        return ""

    # 组装 HTML
    html_parts = []
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

    # 各 Block
    for block in data.blocks:
        items = _coerce_items(block.items)
        ap('<section class="block">')

        # Reference
        ref_src = _media_src(block.ref_audio, out_html_path, media_mode, assets_dirname)
        ap('<div class="ref">')
        ap('<div class="label">Reference WAV</div>')
        if ref_src:
            ap(f'<audio controls preload="none" src="{html_escape(ref_src)}"></audio>')
        else:
            ap('<div class="muted">—</div>')
        ap("</div>")  # .ref

        # 表格（两两配对）
        ap('<div class="table-wrap">')
        ap("<table>")
        ap("<colgroup>" + "".join("<col />" for _ in range(4)) + "</colgroup>")
        ap("<thead><tr><th>文本 A</th><th>音频 A</th><th>文本 B</th><th>音频 B</th></tr></thead>")
        ap("<tbody>")

        for i in range(0, len(items), 2):
            left = items[i]
            right = items[i + 1] if i + 1 < len(items) else None

            # 左
            left_audio = _media_src(left.audio, out_html_path, media_mode, assets_dirname) if left else None
            ap("<tr>")
            ap(f"<td>{render_text_box(left.text)}</td>")
            ap(f"<td>{render_audio_cell(left_audio)}</td>")

            # 右（可能为空）
            if right:
                right_audio = _media_src(right.audio, out_html_path, media_mode, assets_dirname)
                ap(f"<td>{render_text_box(right.text)}</td>")
                ap(f"<td>{render_audio_cell(right_audio)}</td>")
            else:
                # 空两列
                ap(f"<td>{render_text_box(None)}</td>")
                ap("<td></td>")

            ap("</tr>")

        ap("</tbody></table></div>")  # .table-wrap
        ap("</section>")  # .block

    ap("</div>")  # .page
    ap("</body></html>")

    if out_html is not None:
        Path(out_html).write_text("\n".join(html_parts), encoding="utf-8")
        return Path(out_html)
    
    return "\n".join(html_parts)


def generate_html_compare(
    data: ReportData,
    out_html: Union[str, Path, None] = None,
    media_mode: str = "link",       # "copy" | "inline" | "link"
    assets_dirname: str = "assets", # 仅 "copy" 模式使用
) -> Path:
    """
    生成静态 HTML 页面：
    - 每个 Block 的 items 两两配对生成表格行；
    - 奇数个样本时，最后一行的右侧两列为空。
    """
    out_html_path = None
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
    .ref {
      display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; align-items: center; margin-bottom: 10px;
    }
    .ref .label { font-size: 13px; color: var(--muted); white-space: nowrap; }
    audio { width: 320px; max-width: 100%; height: 32px; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    colgroup col { width: 16%; }
    thead th {
      position: sticky; top: 0; background: #fafafa; border-bottom: 1px solid var(--border);
      font-weight: 600; font-size: 13px; padding: 8px; text-align: left;
    }
    tbody td { border-top: 1px solid var(--border); padding: 8px; vertical-align: top; }
    .text-box {
      width: 100%; height: 100px; border: 1px solid var(--border); border-radius: 6px;
      padding: 8px; background: #fff; overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 14px;
    }
    .muted { color: var(--muted); font-size: 12px; }
    """

    def render_text_box(text: Optional[str]) -> str:
        content = "" if text is None else html_escape(str(text))
        return f'<div class="text-box" role="textbox" aria-readonly="true">{content}</div>'

    def render_audio_cell(src: Optional[str]) -> str:
        if src:
            return f'<audio controls preload="none" src="{html_escape(src)}"></audio>'
        # 右侧为空时，返回空内容（占位由单元格 padding/布局保证）
        return ""

    # 组装 HTML
    html_parts = []
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

    # 各 Block
    for block_idx, block in enumerate(data.blocks):
        # items = _coerce_items(block.items)
        items = block.items
        model_name = items[0]['model_name']
        model_name2 = items[0]['model_name2']
        ap('<section class="block">')

        # Reference
        ref_src = _media_src(block.ref_audio, out_html_path, media_mode, assets_dirname)
        ap('<div class="ref">')
        ap(f'<div class="label">{block_idx}. Reference WAV</div>')
        if ref_src:
            ap(f'<audio controls preload="none" src="{html_escape(ref_src)}"></audio>')
        else:
            ap('<div class="muted">—</div>')
        ap("</div>")  # .ref

        # 表格（两两配对）
        ap('<div class="table-wrap">')
        ap("<table>")
        ap("<colgroup>" + "".join("<col />" for _ in range(4)) + "</colgroup>")
        ap(f"<thead><tr><th>文本 A</th><th>{model_name}</th><th>{model_name2}</th><th>文本 B</th><th>{model_name}</th><th>{model_name2}</th></tr></thead>")
        ap("<tbody>")

        for i in range(0, len(items), 2):
            left = items[i]
            right = items[i + 1] if i + 1 < len(items) else None

            # 左
            left_audio = _media_src(left['audio'], out_html_path, media_mode, assets_dirname) if left else None
            left_audio2 = _media_src(left['audio2'], out_html_path, media_mode, assets_dirname) if left else None
            ap("<tr>")
            ap(f"<td>{render_text_box(left['text'])}</td>")
            ap(f"<td>{render_audio_cell(left_audio)}</td>")
            ap(f"<td>{render_audio_cell(left_audio2)}</td>")

            # 右（可能为空）
            if right:
                right_audio = _media_src(right['audio'], out_html_path, media_mode, assets_dirname)
                right_audio2 = _media_src(right['audio2'], out_html_path, media_mode, assets_dirname)
                ap(f"<td>{render_text_box(right['text'])}</td>")
                ap(f"<td>{render_audio_cell(right_audio)}</td>")
                ap(f"<td>{render_audio_cell(right_audio2)}</td>")
            else:
                # 空两列
                ap(f"<td>{render_text_box(None)}</td>")
                ap("<td></td>")

            ap("</tr>")

        ap("</tbody></table></div>")  # .table-wrap
        ap("</section>")  # .block

    ap("</div>")  # .page
    ap("</body></html>")

    if out_html is not None:
        Path(out_html).write_text("\n".join(html_parts), encoding="utf-8")
        return Path(out_html)
    
    return "\n".join(html_parts)


def generate_html_grid_search(
    page_title: str,
    notes: str,
    data: Union[ReportData, List[ReportData]],
    out_html: Union[str, Path, None] = None,
    media_mode: str = "link",       # "copy" | "inline" | "link"
    assets_dirname: str = "assets", # 仅 "copy" 模式使用
) -> Path:
    """
    生成静态 HTML 页面：
    - 每个 Block 的 items 两两配对生成表格行；
    - 奇数个样本时，最后一行的右侧两列为空。
    """
    out_html_path = None
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
    .ref {
      display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; align-items: center; margin-bottom: 10px;
    }
    .ref .label { font-size: 13px; color: var(--muted); white-space: nowrap; }
    audio { width: 320px; max-width: 100%; height: 32px; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    colgroup col { width: 25%; }
    thead th {
      position: sticky; top: 0; background: #fafafa; border-bottom: 1px solid var(--border);
      font-weight: 600; font-size: 13px; padding: 8px; text-align: left;
    }
    tbody td { border-top: 1px solid var(--border); padding: 8px; vertical-align: top; }
    .text-box {
      width: 100%; height: 100px; border: 1px solid var(--border); border-radius: 6px;
      padding: 8px; background: #fff; overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 14px;
    }
    .muted { color: var(--muted); font-size: 12px; }
    """

    def render_text_box(text: Optional[str]) -> str:
        content = "" if text is None else html_escape(str(text))
        return f'<div class="text-box" role="textbox" aria-readonly="true">{content}</div>'

    def render_audio_cell(src: Optional[str]) -> str:
        if src:
            return f'<audio controls preload="none" src="{html_escape(src)}"></audio>'
        # 右侧为空时，返回空内容（占位由单元格 padding/布局保证）
        return ""

    if not isinstance(data, list):
        data_lst = [data]
    else:
        data_lst = data

    # 组装 HTML
    html_parts = []
    ap = html_parts.append
    ap("<!DOCTYPE html>")
    ap('<html lang="zh-CN">')
    ap("<head>")
    ap('<meta charset="utf-8" />')
    ap('<meta name="viewport" content="width=device-width,initial-scale=1" />')
    ap(f'<title>{html_escape(page_title)}</title>')
    ap("<style>")
    ap(css)
    ap("</style>")
    ap("</head>")
    ap("<body>")
    ap('<div class="page">')

    if (notes or "").strip():
        notes_html = html_escape(notes or "")
        ap('<section class="notes">')
        ap('<div class="label">实验总参数</div>')
        ap(f'<div class="box">{notes_html}</div>')
        ap("</section>")

    for data_idx, data in enumerate(data_lst):

        if (data.notes or "").strip():
            notes_html = html_escape(data.notes or "")
            ap('<section class="notes">')
            ap(f'<div class="label">{data_idx}. 搜索参数</div>')
            ap(f'<div class="box">{notes_html}</div>')
            ap("</section>")

        # 各 Block
        for block_idx, block in enumerate(data.blocks):
            items = _coerce_items(block.items)
            ap('<section class="block">')

            # Reference
            ref_src = _media_src(block.ref_audio, out_html_path, media_mode, assets_dirname)
            ap('<div class="ref">')
            ap(f'<div class="label">{block_idx}. Reference WAV</div>')
            if ref_src:
                ap(f'<audio controls preload="none" src="{html_escape(ref_src)}"></audio>')
            else:
                ap('<div class="muted">—</div>')
            ap("</div>")  # .ref

            # 表格（两两配对）
            ap('<div class="table-wrap">')
            ap("<table>")
            ap("<colgroup>" + "".join("<col />" for _ in range(4)) + "</colgroup>")
            ap("<thead><tr><th>文本 A</th><th>音频 A</th><th>文本 B</th><th>音频 B</th></tr></thead>")
            ap("<tbody>")

            for i in range(0, len(items), 2):
                left = items[i]
                right = items[i + 1] if i + 1 < len(items) else None

                # 左
                left_audio = _media_src(left.audio, out_html_path, media_mode, assets_dirname) if left else None
                ap("<tr>")
                ap(f"<td>{render_text_box(left.text)}</td>")
                ap(f"<td>{render_audio_cell(left_audio)}</td>")

                # 右（可能为空）
                if right:
                    right_audio = _media_src(right.audio, out_html_path, media_mode, assets_dirname)
                    ap(f"<td>{render_text_box(right.text)}</td>")
                    ap(f"<td>{render_audio_cell(right_audio)}</td>")
                else:
                    # 空两列
                    ap(f"<td>{render_text_box(None)}</td>")
                    ap("<td></td>")

                ap("</tr>")

            ap("</tbody></table></div>")  # .table-wrap
            ap("</section>")  # .block

    ap("</div>")  # .page
    ap("</body></html>")

    if out_html is not None:
        Path(out_html).write_text("\n".join(html_parts), encoding="utf-8")
        return Path(out_html)
    
    return "\n".join(html_parts)


# -------------------------
# 示例
# -------------------------
if __name__ == "__main__":
    # 示例 1：完整的 text+audio
    blk1 = Block(
        ref_audio="examples/ref/ref1.wav",
        items=[
            Sample(text="今天天气不错，适合出去散步。", audio="examples/out/ref1_case1.wav"),
            Sample(text="更长的句子测试，包含停顿与连读。", audio="examples/out/ref1_case2.wav"),
            Sample(text="第三条示例（奇数个，右侧将留空）。", audio="examples/out/ref1_case3.wav"),
        ],
    )

    # 示例 2：只有音频（传 str 即可，文本自动为空）
    blk2 = Block(
        ref_audio="examples/ref/ref2.wav",
        items=[
            "examples/out/ref2_case1.wav",
            "examples/out/ref2_case2.wav",
        ],
    )

    report = ReportData(
        notes=(
            "模型: my-tts-zero-shot-v1\n"
            "采样率: 22050 Hz\n"
            "推理参数: top_p=0.9, temperature=0.7\n"
            "备注: 表格按样本两两在一行展示；若样本为奇数，右侧留空。"
        ),
        blocks=[blk1, blk2],
        page_title="Zero-shot TTS Inference",
    )

    out_file = generate_html(
        data=report,
        out_html=Path("report/tts_report.html"),
        media_mode="copy",        # "copy" | "inline" | "link"
        assets_dirname="assets",
    )
    print(f"Wrote: {out_file}")

