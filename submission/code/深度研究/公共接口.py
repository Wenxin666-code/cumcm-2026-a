"""直接复用用户原 code/公共依赖.py；此处只提供研究侧路径和私有接口适配。

_initialize_science 和 _sections 是原文件的私有实现，不是承诺稳定的公共 API。
首次 section() 会初始化原文件全部科学分区，包括导出器，因而也需要
XlsxWriter。不会改写原文件、复制其算法或绕过它的运行/验收约束。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_PACKAGE = Path(__file__).resolve().parents[2]
CODE_DIR = Path(os.environ.get('RESEARCH_CODE_DIR', _PACKAGE / 'code')).expanduser().resolve()
RESEARCH_ROOT = Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parent)).expanduser().resolve()
_PUBLIC = None


def configure(code_dir=None, research_root=None):
    """指定现有源码/数据目录；切换 code 会重新导入该文件而不会复制源码。"""
    global CODE_DIR, RESEARCH_ROOT, _PUBLIC
    code = Path(code_dir).expanduser().resolve() if code_dir is not None else CODE_DIR
    root = Path(research_root).expanduser().resolve() if research_root is not None else RESEARCH_ROOT
    if not (code / '公共依赖.py').is_file():
        raise FileNotFoundError(f'请指定包含公共依赖.py的 --code-dir：{code}')
    if not root.is_dir():
        raise FileNotFoundError(f'深度研究资源目录不存在：{root}')
    if code != CODE_DIR:
        _PUBLIC = None
    CODE_DIR, RESEARCH_ROOT = code, root
    return root


def public():
    """导入指定原文件；普通模块加载，不执行提取或拼装的源码字符串。"""
    global _PUBLIC
    if _PUBLIC is None:
        configure()
        source = CODE_DIR / '公共依赖.py'
        name = '_research_original_public_' + hashlib.sha256(str(source).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        if not callable(getattr(module, '_initialize_science', None)) or not isinstance(getattr(module, '_sections', None), dict):
            raise RuntimeError('原公共依赖的私有科学接口已改变；需要重新核对适配器。')
        _PUBLIC = module
    return _PUBLIC


def section(name):
    """惰性返回原分区对象；不对分区属性赋值，不改变原全局求解器钩子。"""
    module = public()
    if name not in module._sections:
        raise KeyError(f'原公共依赖没有分区 {name!r}；可用项：{", ".join(module._sections)}')
    try:
        module._initialize_science()
    except ModuleNotFoundError as error:
        raise RuntimeError(f'原公共依赖初始化缺少 {error.name}；请在所用 Python 环境提供此依赖。'
                           '该私有入口会同时初始化导出器，因此也需要 XlsxWriter。') from error
    return module._sections[name]


def paths():
    """定位共享配置、数据及研究附件；不依赖原电脑或旧研究目录。"""
    return {'root': RESEARCH_ROOT, 'code': CODE_DIR, 'inputs': RESEARCH_ROOT / 'inputs',
            'contract': CODE_DIR / 'config/model.json',
            'kinematics_contract': CODE_DIR / 'config/kinematics/model.json',
            'latent_table': CODE_DIR / 'data/latent',
            'near_critical_protocol': CODE_DIR / 'config/near_critical_protocol.json'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity():
    """记录本次实际使用的原源码与适配器；不继承任何旧正式验收结论。"""
    files = {'public_source': CODE_DIR / '公共依赖.py', 'adapter': Path(__file__).resolve()}
    return {'schema': 'research-direct-public-v1',
            'files': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in files.items()},
            'private_api': ['_initialize_science', '_sections'], 'historical_acceptance_claimed': False}
