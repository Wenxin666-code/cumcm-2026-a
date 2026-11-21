"""验证任务共享的路径、数据校验与主模型接口。

数学算法分别位于各个验证文件。本文件只负责资源定位和运行记录，
通过普通模块导入复用 code/公共依赖.py 及 code/深度研究 中的实现。
"""

from __future__ import annotations

import hashlib
import importlib.util
import csv
import gzip
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Iterable

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT.parent
CODE = ROOT
DATA = CODE / "data/验证数据"
RESEARCH = CODE / "深度研究"
DATA_MANIFEST = CODE / "data/验证数据清单.json"
_modules: dict[str, ModuleType] = {}


def read_json(path: str | Path):
    """读取UTF-8 JSON，兼容带BOM的原始证据。"""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value) -> None:
    """保存格式化记录；非有限数值应由调用方明确处理。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows, *, columns=None, encoding="utf-8-sig") -> None:
    """写出字典行；字段按首次出现排序，也可显式指定完整列顺序。"""
    path = Path(path)
    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path, rows, *, ensure_ascii=True, separators=None) -> None:
    """逐行压缩写出JSON记录，支持生成器，避免整份数学日志驻留内存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=ensure_ascii, separators=separators, allow_nan=False)
                + "\n"
            )


def load_npz(path, keys=None) -> dict:
    """按需加载数值数组并关闭文件；返回值与压缩包的生命周期独立。"""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (archive.files if keys is None else keys)}


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_output(path: str | Path, *, allow_existing: bool = False) -> Path:
    """要求输出位于支撑包之外，以保护输入与既有证据。"""
    output = Path(path).expanduser().resolve()
    if output.is_relative_to(SUPPORT):
        raise ValueError("输出目录应位于整个支撑材料目录之外。")
    output.mkdir(parents=True, exist_ok=allow_existing)
    return output


def _load_module(name: str, path: Path) -> ModuleType:
    """按明确文件路径导入模块，避免工作目录影响模块选择。"""
    if name in _modules:
        return _modules[name]
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise RuntimeError(f"模块{name}已从其他目录装载，请使用独立进程运行验证。")
        _modules[name] = existing
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法装载模块：{path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    _modules[name] = module
    return module


def _research_adapter() -> ModuleType:
    if str(RESEARCH) not in sys.path:
        sys.path.insert(0, str(RESEARCH))
    adapter = _load_module("公共接口", RESEARCH / "公共接口.py")
    adapter.configure(code_dir=CODE, research_root=RESEARCH)
    return adapter


def science(name: str):
    """复用主程序的科学分区，如radial_core、moving_model。

    分区由已有研究适配器定位；该适配器检查主文件私有接口的形状。
    此处既不复制主模型，也不修改主模型中的函数或全局求解器钩子。
    """
    return _research_adapter().section(name)


def research(name: str) -> ModuleType:
    """取得已有研究实现，例如research('二维模型')。"""
    if Path(name).name != name or name.endswith(".py"):
        raise ValueError("请给出研究模块名，例如二维模型。")
    _research_adapter()
    return _load_module("_verification_research_" + name, RESEARCH / (name + ".py"))


def verify_data() -> dict:
    """按当前数据清单核对实际数据文件，不把来源源码身份当作现行代码。"""
    manifest = read_json(DATA_MANIFEST)
    expected = manifest["files"]
    actual = {p.relative_to(DATA).as_posix() for p in DATA.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError("验证数据文件集合与数据清单不符。")
    for name, digest in expected.items():
        if sha256(DATA / name) != digest:
            raise ValueError("验证数据校验失败：" + name)
    return {
        "passed": True,
        "files_checked": len(expected),
        "manifest_sha256": sha256(DATA_MANIFEST),
    }


def source_hashes(extra_files: Iterable[str | Path] = (), *, root=SUPPORT) -> dict:
    """记录包内科学源码和指定输入的哈希，包内路径统一相对支撑目录。"""
    root = Path(root).resolve()
    files = list((root / "code").glob("*.py"))
    files += list((root / "code/深度研究").glob("*.py"))
    files += [(root / p).resolve() for p in extra_files]
    identity = {}
    for path in sorted(set(files)):
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        identity[name] = sha256(path)
    return identity


def source_identity(extra_files: Iterable[str | Path] = ()) -> dict:
    """记录此次运行的科学源码、指定输入和验证数据清单身份。"""
    return {
        "schema": "integrated-verification-v1",
        "files": source_hashes(extra_files),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
    }


def audit_header(
    *, mode=None, started=None, identity_key="current_execution_identity", extra_files=(), **fields
) -> dict:
    """构造公共运行记录；通过与否及验证范围由各数学任务显式给出。"""
    record = dict(fields)
    record[identity_key] = source_identity(extra_files)
    if mode is not None:
        record["mode"] = mode
    if started is not None:
        record["elapsed_s"] = time.perf_counter() - started
    return record


def run_cli_modes(mode, handlers, *args, **kwargs):
    """执行分派表中的入口；返回值和异常直接交由各任务处理。"""
    try:
        handler = handlers[mode]
    except KeyError:
        raise ValueError(f"未知运行模式：{mode}；可选：{', '.join(handlers)}") from None
    return handler(*args, **kwargs)


def origin_source_hash(relative_path: str) -> str:
    """返回既有证据记录的来源源码身份，仅用于解释来源字段。"""
    index = read_json(DATA / "来源源码索引.json")
    return index["files"][relative_path]["sha256"]
