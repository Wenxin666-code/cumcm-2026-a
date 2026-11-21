"""四问共享的可读源码、任务组织、输入核验和执行审计。

源码分为本文件与四个题号入口；科学函数在文件后部直接定义，按需初始化。
本地清单用于一致性检查，防篡改还需要独立可信的清单哈希和校验器。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata, util, import_module
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
import zipfile

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CUMCM_ROOT', CODE.parent)).resolve()
MANIFEST_NAME = 'MANIFEST.sha256.json'
SOURCE_SCHEMA = 'readable-five-file-v1'
SOURCE_NAMES = ('公共依赖.py', '问题1.py', '问题2.py', '问题3.py', '问题4.py')
VERIFICATION_NAMES = ('一维FV验证.py', '一维GLL验证.py', '二维验证.py', '时间界限验证.py',
                      '晚期区间验证.py', '输入与假设验证.py', '问题一舍入验证.py', '验证公共接口.py')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_ENV_DEFAULTS = dict(PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                     PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                     OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
EXPECTED = {
    1: '54f10f2fb90344b5dff6d0fb012d39c8c6c7c4093ad5561bfd27daba1a5b8010',
    2: 'c552dd167e163a20146aa47d8580a0157d98e61a5b502d93f3893de2b92b2560',
    3: '5dc924c273c142a98260880e2e11d2640ba858a477492df1c5c8a0f26a5a713e',
    4: '4ef7b16d4cb5d7f8262ef09a75a3e04ef116bfdae7367e92da33b51f50b5580d',
}
PINS = {
    'numpy': ('numpy', '2.5.3'), 'scipy': ('scipy', '1.18.1'),
    'openpyxl': ('openpyxl', '3.1.5'), 'XlsxWriter': ('xlsxwriter', '3.2.9'),
    'matplotlib': ('matplotlib', '3.11.1'), 'Pillow': ('PIL', '12.3.0'),
    'pypdf': ('pypdf', '6.18.0'), 'python-flint': ('flint', '0.9.0'),
}
PROFILES = {
    'core': ('numpy', 'scipy', 'openpyxl', 'XlsxWriter'),
    'export': ('numpy', 'openpyxl', 'XlsxWriter'),
    'figures': ('numpy', 'scipy', 'openpyxl', 'matplotlib', 'Pillow'),
    'certificates': ('numpy', 'scipy', 'openpyxl', 'python-flint'),
    'all': tuple(PINS),
}
PROFILES['run'] = PROFILES['q1'] = PROFILES['core']


def source_file():
    """The physical shared source file used by scientific source audits."""
    return Path(__file__).resolve()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'源码清单含重复键：{key}')
        value[key] = item
    return value


def _package_sources(code):
    code = Path(code).resolve()
    expected = set(SOURCE_NAMES + VERIFICATION_NAMES)
    # code根目录集中存放五份主流程源码和八份验证源码；研究文件位于子目录。
    actual = {path.relative_to(code).as_posix() for path in code.rglob('*.py')
              if not path.is_relative_to(code / '深度研究')}
    if actual != expected:
        raise ValueError(f'code目录的主流程与验证源码集合不符：missing={sorted(expected - actual)}, '
                         f'extra={sorted(actual - expected)}')
    if next(code.rglob('*.pyc'), None) is not None:
        raise ValueError('请移除旧字节码并使用 python -B，避免加载未核验的 .pyc')
    result = {}
    for name in SOURCE_NAMES + VERIFICATION_NAMES:
        path = code / name
        if path.is_symlink() or not path.resolve().is_relative_to(code):
            raise ValueError(f'源码不能是符号链接或越界路径：{name}')
        payload = path.read_bytes()
        compile(payload, str(path), 'exec')  # Check syntax only; never execute strings.
        result[f'code/{name}'] = hashlib.sha256(payload).hexdigest()
    return result


def verify_sources(package=None, code=None, expected_manifest_sha256=None):
    """核对主流程与验证源码的目录及清单绑定，返回五份主流程的源码身份。"""
    code = Path(code if code is not None else Path(package) / 'code' if package is not None else CODE).resolve()
    manifest = code.parent / MANIFEST_NAME
    payload = manifest.read_bytes()
    pin = expected_manifest_sha256 or os.environ.get('CUMCM_MANIFEST_SHA256')
    if pin is not None:
        if not _SHA256.fullmatch(pin) or hashlib.sha256(payload).hexdigest() != pin:
            raise ValueError('源码清单与外部可信 SHA-256 不匹配')
    document = json.loads(payload, object_pairs_hook=_unique_object)
    if (not isinstance(document, dict) or document.get('version') != 2
            or not isinstance(document.get('files'), dict)):
        raise ValueError('不支持的源码清单格式')
    recorded = document['files']
    required = {f'code/{name}' for name in SOURCE_NAMES}
    if set(document.get('main_sources', [])) != required or not required.issubset(recorded):
        raise ValueError('统一清单必须声明并包含五份主流程源码')
    if not {f'code/{name}' for name in VERIFICATION_NAMES}.issubset(recorded):
        raise ValueError('统一清单必须包含八份验证源码')
    for name, expected in recorded.items():
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError(f'非法源码 SHA-256：{name}')
    actual = _package_sources(code)
    changed = [name for name in actual if actual[name] != recorded[name]]
    if changed:
        raise ValueError(f'源码校验失败，文件发生变化：{changed}')
    return {name: actual[name] for name in actual if name in required}


def source_identity():
    """Scientific adapters share one explicit five-file identity scheme."""
    files = verify_sources(code=CODE)
    return {'schema': SOURCE_SCHEMA, 'files': {Path(name).name: value for name, value in files.items()}}


def check_environment(profile='core', *, probe_imports=False):
    """Read metadata first; load scientific libraries only for run/export preflight."""
    if profile not in PROFILES:
        raise ValueError(f'未知环境配置：{profile}')
    errors, dependencies, loaded = [], {}, {}
    if sys.version_info < (3, 11):
        errors.append('需要 Python >= 3.11')
    if profile == 'q1' and os.name != 'nt':
        errors.append('问题1当前资源检查需要 Windows 的 GlobalMemoryStatusEx')
    for name in PROFILES[profile]:
        module, required = PINS[name]
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        try:
            spec = util.find_spec(module)
        except (ImportError, ValueError, AttributeError) as error:
            spec = None
            errors.append(f'无法定位 {module}：{error}')
        dependencies[name] = {'required': required, 'installed': installed,
                              'module': module, 'origin': getattr(spec, 'origin', None),
                              'version_matches': installed == required, 'module_found': spec is not None}
        if installed != required:
            errors.append(f'{name} 需要 {required}，当前为 {installed or "未安装"}')
        if spec is None:
            errors.append(f'无法导入模块：{module}')
    if probe_imports and not errors:
        for name in PROFILES[profile]:
            module, required = PINS[name]
            try:
                loaded[module] = str(getattr(import_module(module), '__version__', 'unknown'))
                if loaded[module] != required:
                    errors.append(f'实际导入的 {module} 版本为 {loaded[module]}，要求 {required}')
            except (ImportError, OSError, RuntimeError) as error:
                errors.append(f'导入 {module} 失败：{error}')
    return {'ok': not errors, 'profile': profile, 'python': platform.python_version(),
            'python_executable': sys.executable, 'platform': platform.platform(),
            'dependencies': dependencies, 'errors': errors,
            'import_probe': {'requested': probe_imports, 'completed': bool(loaded), 'loaded_versions': loaded}}


def require_environment(profile='core'):
    report = check_environment(profile, probe_imports=True)
    if not report['ok']:
        raise EnvironmentError('运行环境检查失败：\n  - ' + '\n  - '.join(report['errors'])
                               + '\n请在包外独立环境安装交付要求的依赖，再重试。')
    return report


def _step_key(script_key):
    key = str(script_key)
    path = PurePosixPath(key)
    allowed = ('solvers/', 'extensions/q1/', 'extensions/endpoint/', 'exporters/')
    if (path.is_absolute() or '..' in path.parts or key != path.as_posix()
            or '\\' in key or ':' in key or not key.endswith('.py') or not key.startswith(allowed)):
        raise ValueError(f'非法计算步骤：{key}')
    return key


def dispatch_step(script_key, args, *, question=None, code=None):
    """Return an isolated worker command; no child source files are materialized."""
    key = _step_key(script_key)
    question = int(question if question is not None else os.environ.get('CUMCM_QUESTION', '1'))
    if question not in (1, 2, 3, 4):
        raise ValueError('题号必须为1至4')
    entry = Path(code if code is not None else CODE) / f'问题{question}.py'
    if not entry.is_file():
        raise ValueError(f'题号入口不存在：{entry}')
    return [sys.executable, '-B', str(entry.resolve()), '--worker', key, *map(str, args)]


def _run_worker(question, key, arguments):
    key = _step_key(key)
    os.environ.update(CUMCM_QUESTION=str(question), CUMCM_ROOT=str(ROOT), CUMCM_CODE=str(CODE))
    before = source_identity()
    profile = 'export' if key.startswith('exporters/') else 'q1' if key.startswith('extensions/q1/') else 'core'
    require_environment(profile)
    _initialize_science(key)
    if key not in _STEP_FUNCTIONS:
        raise ValueError(f'未实现的计算步骤：{key}')
    previous = sys.argv
    sys.argv = [str(source_file()), *map(str, arguments)]
    try:
        _STEP_FUNCTIONS[key]()
    finally:
        sys.argv = previous
    if source_identity() != before:
        raise ValueError('执行期间五份源码发生变化')
    return 0


@dataclass
class Job:
    name: str
    command: list
    products: list
    checks: tuple = ()
    artifact_roots: tuple = ()


class Context:
    def __init__(self, work, inputs, root=None, code=None, question=None):
        self.work = Path(work).resolve()
        self.inputs = Path(inputs).resolve()
        self.root = Path(root if root is not None else ROOT).resolve()
        self.code = Path(code if code is not None else CODE).resolve()
        self.question = int(question if question is not None else os.environ.get('CUMCM_QUESTION', '1'))
        self.contract = self.root / 'code/config/model.json'
        self.runs = self.work / 'runs'
        self.baseline = self.runs / 'production'
        self.endpoint = self.runs / 'endpoint'
        self.reports = self.work / 'reports'
        self.q1 = self.runs / 'q1_disk'
        self.results = self.work / 'results'

    @property
    def environment(self):
        search_paths = [str(self.code)]
        inherited = os.environ.get('PYTHONPATH')
        if inherited:
            search_paths.extend(str(Path(path).resolve()) for path in inherited.split(os.pathsep))
        return dict(os.environ, **_ENV_DEFAULTS, CUMCM_ROOT=str(self.root),
                    CUMCM_CODE=str(self.code), CUMCM_QUESTION=str(self.question),
                    PYTHONPATH=os.pathsep.join(dict.fromkeys(search_paths)))

    def job(self, name, script, args, products, checks=(), artifact_roots=()):
        return Job(name, dispatch_step(script, args, question=self.question, code=self.code),
                   products, checks, artifact_roots)

    def baseline_job(self, case):
        return self.job('baseline_' + case, 'solvers/solve_all.py',
                        ['--inputs', self.inputs, '--contract', self.contract,
                         '--out', self.baseline, '--question', case],
                        [self.baseline / f'run_{case}.json'],
                        [(self.baseline / f'run_{case}.json', 'original_inputs_unchanged')],
                        artifact_roots=[self.baseline / case])

    def endpoint_job(self, case):
        return self.job('endpoint_' + case, 'extensions/endpoint/reproduce_endpoint.py',
                        ['--base-code', self.code, '--inputs', self.inputs,
                         '--contract', self.contract, '--runs', self.baseline,
                         '--work', self.reports / case, '--scratch', self.endpoint / case,
                         '--stage', 'all', '--case', case],
                        [self.endpoint / case / 'finest_candidate' / case / 'endpoint.json'],
                        [(self.endpoint / case / 'finest_balance' / case / 'balance.json', 'passed'),
                         (self.endpoint / case / 'validation_metrics/validation_replay_comparison.json', 'completed')],
                        artifact_roots=[self.endpoint / case, self.reports / case])

    def output_jobs(self, question):
        source = ['--q1', self.q1 / 'full/solution.npz',
                  '--q23', self.endpoint / 'p23/finest_candidate/p23',
                  '--q4', self.endpoint / 'q4/finest_candidate/q4']
        return [
            self.job(f'export_q{question}', 'exporters/export_results.py',
                     source + ['--output-dir', self.results, '--problems', question],
                     [self.results / f'result{question}.xlsx']),
            self.job(f'verify_q{question}', 'exporters/verify_finest_outputs.py',
                     source + ['--results', self.results, '--problems', question,
                               '--out', self.reports / f'result{question}_verification.json'],
                     [self.reports / f'result{question}_verification.json'],
                     [(self.reports / f'result{question}_verification.json', 'passed')]),
        ]


def save(path, value):
    """Publish complete JSON in one same-directory atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def check_results(questions=(1, 2, 3, 4), root=None):
    root = Path(root) if root is not None else ROOT
    questions = list(questions)
    for question in questions:
        path = root / 'results' / f'result{question}.xlsx'
        if digest(path) != EXPECTED[question]:
            raise ValueError(f'问题{question}正式结果文件与绑定版本不同')
        with zipfile.ZipFile(path) as book:
            if book.testzip() is not None:
                raise ValueError(f'问题{question}工作簿损坏')
    return {'selected_results_passed': True, 'questions': questions, 'PDE_started': False}


@contextmanager
def _work_lock(work):
    """OS locks release on process exit; the persistent lock file is harmless."""
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = work / '.workflow.lock'
    with lock_path.open('a+b') as stream:
        if lock_path.stat().st_size == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'计算目录正在被其他进程使用：{work}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _snapshot(context, environment_report=None):
    sources = verify_sources(code=context.code)
    installed = {}
    for name in PINS:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    return {
        'version': 3,
        'source_schema': SOURCE_SCHEMA,
        'manifest_sha256': digest(context.code.parent / MANIFEST_NAME),
        'final_sources': sources,
        'inputs': {name: digest(path) for name, path in (
            ('input/附件1.xlsx', context.inputs / '附件1.xlsx'),
            ('input/附件2.xlsx', context.inputs / '附件2.xlsx'),
            ('code/config/model.json', context.contract))},
        'environment': installed,
        'python': {'executable': sys.executable, 'version': sys.version},
    }


def _definition(context, job):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', job.name):
        raise ValueError(f'非法任务名：{job.name!r}')
    paths = [Path(path).resolve() for path in [*job.products, *(path for path, _ in job.checks)]]
    roots = sorted({str(Path(path).resolve()) for path in job.artifact_roots})
    if job.name.startswith('export_q'):
        paths.append((context.results / f'{job.name}_manifest.json').resolve())
    for path in [*paths, *map(Path, roots)]:
        if not path.is_relative_to(context.work) or path == context.work:
            raise ValueError(f'任务产品必须位于计算目录中：{path}')
        if (path.is_relative_to(context.work / 'state') or path.is_relative_to(context.work / 'logs')
                or path == context.work / '.workflow.lock'):
            raise ValueError(f'任务产品不能覆盖接续记录：{path}')
    return {'command': list(map(str, job.command)), 'cwd': str(context.root),
            'products': sorted({str(path) for path in paths}),
            'artifact_roots': roots,
            'checks': [[str(Path(path).resolve()), field] for path, field in job.checks],
            'environment': {key: context.environment[key] for key in
                            (*_ENV_DEFAULTS, 'CUMCM_ROOT', 'CUMCM_CODE', 'CUMCM_QUESTION', 'PYTHONPATH')}}


def _product_paths(definition):
    paths = set(definition['products'])
    for root_name in definition.get('artifact_roots', []):
        root = Path(root_name)
        if not root.is_dir() or root.resolve() != root:
            raise ValueError(f'任务输出目录不存在或变为链接：{root}')
        for path in root.rglob('*'):
            if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                raise ValueError(f'任务输出目录不能包含链接：{path}')
            if path.is_file():
                paths.add(str(path))
    return sorted(paths)


def _verify_products(definition):
    products = {path: digest(path) for path in _product_paths(definition)}
    for path, field in definition['checks']:
        if json.loads(Path(path).read_text(encoding='utf-8')).get(field) is not True:
            raise ValueError(f'未满足完成条件：{path} 中的 {field}')
    return products


def _verify_completed(context, state, identity):
    """Validate upstream arrays even when this invocation only exports workbooks."""
    records = {}
    for marker in sorted(state.glob('*.json')):
        if marker.name == 'inputs_and_code.json':
            continue
        record = json.loads(marker.read_text(encoding='utf-8'))
        definition = record.get('definition', {})
        if record.get('status') != 'completed' or not definition.get('products'):
            raise ValueError(f'无效的已完成任务记录：{marker}')
        for name in [*definition['products'], *definition.get('artifact_roots', [])]:
            if not Path(name).resolve().is_relative_to(context.work):
                raise ValueError(f'已完成任务的产品路径越界：{name}')
        if (record.get('source_manifest_sha256') != identity['manifest_sha256']
                or record.get('products') != _verify_products(definition)):
            raise ValueError(f'{marker.stem} 的已完成结果或设置发生变化')
        records[marker.stem] = record
    return records


def _protect_completed_outputs(job_name, definition, records):
    for completed_name, record in records.items():
        if completed_name == job_name:
            continue
        previous = record['definition']
        for current in [*definition['products'], *definition.get('artifact_roots', [])]:
            for protected in [*previous['products'], *previous.get('artifact_roots', [])]:
                left, right = Path(current), Path(protected)
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError(f'{job_name} 的输出会覆盖已完成任务 {completed_name}：{left}')


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _run_job(context, job, definition, identity, state, environment_report):
    attempt = state / 'attempts' / job.name / uuid.uuid4().hex
    record = {'status': 'running', 'started_at': _timestamp(), 'definition': definition,
              'source_manifest_sha256': identity['manifest_sha256'],
              'environment_preflight': environment_report or {}}
    save(attempt / 'audit.json', record)
    try:
        # Preserve partial products from an earlier failure without accepting them
        # as newly produced output when a retried process exits successfully.
        old_paths = list(map(Path, definition['products'] + definition.get('artifact_roots', [])))
        if job.name.startswith('export_q'):
            old_paths.append(context.results / 'export_manifest.json')
        old_paths = sorted(set(old_paths), key=lambda path: len(path.parts))
        old_paths = [path for path in old_paths if not any(
            path != parent and path.is_relative_to(parent) for parent in old_paths)]
        for index, path in enumerate(old_paths):
            if not path.resolve().is_relative_to(context.work):
                raise ValueError(f'任务产品越出计算目录：{path}')
            if path.exists():
                stale = attempt / 'stale' / f'{index}_{path.name}'
                stale.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, stale)
        log = context.work / 'logs' / f'{job.name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'开始：{job.name}', flush=True)
        with log.open('ab') as output:
            subprocess.run(definition['command'], cwd=context.root, env=context.environment,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        after = _snapshot(context, environment_report)
        if after != identity:
            raise ValueError('执行期间源码、清单或输入发生变化，结果未标记为完成')
        _verify_completed(context, state, identity)
        if job.name.startswith('export_q'):
            os.replace(context.results / 'export_manifest.json',
                       context.results / f'{job.name}_manifest.json')
        products = _verify_products(definition)
        record.update(status='completed', finished_at=_timestamp(), products=products,
                      source_and_inputs_unchanged=True)
        save(attempt / 'audit.json', record)
        save(state / f'{job.name}.json', record)
    except BaseException as error:
        record.update(status='failed', finished_at=_timestamp(),
                      error=f'{type(error).__name__}: {error}',
                      returncode=getattr(error, 'returncode', None))
        try:
            record['source_and_inputs_unchanged'] = _snapshot(context, environment_report) == identity
        except Exception as verification_error:
            record['verification_error'] = str(verification_error)
        record['partial_products'] = {}
        try:
            existing = dict(definition, artifact_roots=[root for root in definition.get('artifact_roots', [])
                                                       if Path(root).is_dir()])
            record['partial_products'] = {
                path: digest(path) for path in _product_paths(existing) if Path(path).is_file()}
            save(attempt / 'audit.json', record)
        except Exception as audit_error:
            error.add_note(f'失败审计无法完整保存：{audit_error}')
        raise


def _same_definition(job_name, left, right):
    """Q2/Q3 share only two scientific steps; preserve every other audit field."""
    if left == right:
        return True
    shared = {'baseline_p23': 'solvers/solve_all.py',
              'endpoint_p23': 'extensions/endpoint/reproduce_endpoint.py'}
    if job_name not in shared:
        return False
    values = []
    for definition in (left, right):
        command = list(definition.get('command', []))
        environment = dict(definition.get('environment', {}))
        if len(command) < 5 or command[3:5] != ['--worker', shared[job_name]]:
            return False
        entry = Path(command[2])
        if entry.name not in ('问题2.py', '问题3.py'):
            return False
        question = entry.stem[-1]
        if environment.get('CUMCM_QUESTION') != question:
            return False
        command[2] = str(entry.with_name('问题2.py'))
        environment['CUMCM_QUESTION'] = '2'
        values.append(dict(definition, command=command, environment=environment))
    return values[0] == values[1]


def execute(context, jobs, environment_report=None):
    work = context.work
    for protected in (context.root, context.code.parent, context.inputs):
        if work == protected or work.is_relative_to(protected) or protected.is_relative_to(work):
            raise ValueError('计算目录必须位于源码包、数据根和输入目录以外，且不能是它们的父目录')
    jobs = list(jobs)
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError('同一计划中不能有重复任务名')
    definitions = [_definition(context, job) for job in jobs]
    with _work_lock(work):
        identity = _snapshot(context, environment_report)
        state = work / 'state'
        binding = state / 'inputs_and_code.json'
        if binding.exists():
            if json.loads(binding.read_text(encoding='utf-8')) != identity:
                raise ValueError('此计算目录的输入、代码或环境已变化，请使用新目录')
        elif any(path.name != '.workflow.lock' for path in work.iterdir()):
            raise FileExistsError('计算目录非空且没有对应的来源记录，请使用新目录')
        else:
            save(binding, identity)
        completed = _verify_completed(context, state, identity)
        for job, definition in zip(jobs, definitions):
            _protect_completed_outputs(job.name, definition, completed)
            if _snapshot(context, environment_report) != identity:
                raise ValueError('执行前源码、清单或输入发生变化')
            marker = state / f'{job.name}.json'
            if marker.exists():
                record = json.loads(marker.read_text(encoding='utf-8'))
                if (record.get('status') != 'completed'
                        or not _same_definition(job.name, record.get('definition', {}), definition)
                        or record.get('source_manifest_sha256') != identity['manifest_sha256']
                        or record.get('products') != _verify_products(definition)):
                    raise ValueError(f'{job.name} 的已完成结果或设置发生变化')
                print(f'复用已完成任务：{job.name}', flush=True)
                save(state / 'reuse' / job.name / f'{uuid.uuid4().hex}.json',
                     {'reused_at': _timestamp(), 'definition': definition,
                      'products': record['products'], 'source_and_inputs_unchanged': True,
                      'source_manifest_sha256': identity['manifest_sha256']})
                continue
            _run_job(context, job, definition, identity, state, environment_report)
            completed[job.name] = json.loads(marker.read_text(encoding='utf-8'))


def _shell_command(command):
    if os.name == 'nt':
        return '& ' + ' '.join("'" + str(part).replace("'", "''") + "'" for part in command)
    return shlex.join(map(str, command))



def _figure_inputs_identity(profile):
    """重用计算必须绑定当前源码、原始输入、模型及潜热表。"""
    import importlib.metadata
    paths = list(CODE.rglob('*.py')) + list((CODE/'config').rglob('*.json'))
    paths += list((CODE/'data/latent').glob('*'))
    paths += [ROOT/'inputs/附件'/f'附件{i}.xlsx' for i in (1, 2)]
    return {'profile': profile, 'python': platform.python_version(), 'files': {p.relative_to(ROOT).as_posix(): digest(p)
            for p in sorted(paths) if p.is_file()},
            'versions': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'scipy', 'openpyxl', 'matplotlib')}}


def _figure_research_module(name):
    """只导入本包研究源码，复用现有公共算子。"""
    folder = CODE/'深度研究'
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    interface = import_module('公共接口')
    interface.configure(CODE, folder)
    return import_module(name)


def _figure_primary_smoke(case, out):
    """低网格全时程接口测试；不作为论文生产数据。"""
    import numpy as np
    _initialize_science()
    core, moving = _sections['_vendor_host2_model'], _sections['moving_model']
    out.mkdir(parents=True, exist_ok=True)
    if case == 'q1':
        solver = _sections['solve_q1']
        # 使用与B1相同的原P1算子；零潜热即B0预热。
        return _figure_latent(out, N=40, scale=0.0)
    model = moving.build_model(ROOT/'inputs/附件', CODE/'config/model.json', N=40,
                               properties='P23' if case=='p23' else 'P4',
                               geometry='fixed' if case=='p23' else 'measured')
    settings = core.IntegrationSettings(rtol=1e-9, atol_temperature_K=1e-9,
                                        atol_moisture_C=1e-11, max_step_s=150.)
    trajectory = core.solve_to_event(model, settings)
    event = float(trajectory.event_time_s)
    report = (int(event/.36)+2)*.36
    y=trajectory.final_state
    segment=core._solve_segment(model,(event,report),y,settings,
                               right_of_breakpoint=True,detect_event=False)
    times=np.unique(np.r_[np.arange(0,event,60.),event,report])
    T,C=trajectory.evaluate_fields(times[:-1])
    last_T,last_C=model.unpack(segment.result.y[:,-1])
    T=np.vstack([T,last_T]); C=np.vstack([C,last_C])
    np.savez_compressed(out/'internal.npz',t_s=times,T_K=T,C=C,
                        xi=model.grid.r_m/model.grid.radius_m,R_m=model.geometry.radius(times))
    save(out/'endpoint.json',{'reported_time_h':report/3600,'reported_max_C':float(last_C.max())})
    save(out/'metadata.json',{'completed':True,'profile':'smoke','N':40})


def _figure_latent(out, N=10240, scale=1.0):
    """原P1均匀径向网格，先湿后热的Radau稠密解，表面附加潜热项。"""
    import numpy as np
    from scipy.integrate import solve_ivp
    _initialize_science()
    radial = _sections['radial_core']
    cfg=json.loads((CODE/'config/model.json').read_text(encoding='utf-8'))
    p,initial=cfg['property_sets']['P1'],cfg['initial_state']
    R,T0,C0=initial['radius_m'],initial['temperature_K'],initial['moisture_C']
    B=p['rho_offset']*p['cp_offset']; k=p['k_offset']
    h=cfg['fixed_domain_model']['h_W_m2_K']; hm=cfg['fixed_domain_model']['h_m_m_s']
    rhod0=p['rho_offset']/(1+C0)
    table_module=_figure_research_module('潜热矩阵')
    # 复用研究模块经来源校验的潜热表，不使用旧数值结果。
    table=table_module.build({'property_set':'P1','geometry':'fixed',
        'mass_mode':'common_wet820','latent_scale':scale},20)
    env=radial.Environment(ROOT/'inputs/附件',cfg)
    grid=radial.NodeRadialGrid(R,N)
    mat=radial.linear_diffusion_matrix(grid,k,h,B)
    gain=R*h/(grid.weights_m2[-1]*B)
    pattern=radial.tridiagonal_sparsity(N+1)
    def D(c):
        if np.any(c<=0) or not np.isfinite(c).all():
            raise ValueError('Nonpositive/nonfinite P1 moisture trial')
        return p['D_prefactor']*np.exp(-p['D_moisture_exponent']/c)
    def moisture_rhs(t,c):
        return radial.diffusion_rhs(grid,c,radial.face_arithmetic(D(c)),hm*(c[-1]-env(t)[1]))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    adapter=None
    if N>=1280:
        adapter=_sections['q1_disk_history_frozen'].DiskRadau(out/('scratch_'+uuid.uuid4().hex),out/'progress.json')
    def integrate(fun, span, y, **opts):
        if adapter:
            return adapter(fun,span,y,method='Radau',dense_output=True,**opts)
        result=solve_ivp(fun,span,y,method='Radau',dense_output=True,**opts)
        if not result.success:raise RuntimeError(result.message)
        return result
    first=min(.01,.1*(R/N)**2/float(D(np.array([C0]))[0]))
    times=np.arange(1801,dtype=float); idx=np.arange(0,N+1,N//20)
    T_out=np.full((1801,21),T0); C_out=np.full((1801,21),C0)
    theta=np.zeros(N+1); moisture=np.full(N+1,C0)
    profiles={}
    # 与论文B1主算例相同的容差；B0生产数据由原问题1主流程提供。
    rtol,atol_T,atol_C=2e-10,1e-11,2e-12
    cuts=env.breakpoints(0,1800)
    for left,right in zip(cuts[:-1],cuts[1:]):
        wet=integrate(moisture_rhs,(left,right),moisture,rtol=rtol,atol=atol_C,
                      jac_sparsity=pattern,first_step=first if left==0 else None)
        def increment(t,th):
            if scale==0:return 0.,0.
            L,Lp=map(float,table.values(float(th[-1]+T0)))
            jw=rhod0*hm*(float(wet.sol(t)[-1])-env(t)[1])
            factor=-R/grid.weights_m2[-1]*jw/B*scale
            return factor*L,factor*Lp
        def latent_heat_rhs(t,th):
            result=mat@th
            result[-1]+=gain*(env(t)[0]-T0)+increment(t,th)[0]
            return result
        def latent_heat_jac(t,th):
            matrix=mat.copy();matrix[-1,-1]+=increment(t,th)[1]
            return matrix
        heat=integrate(latent_heat_rhs,(left,right),theta,rtol=rtol,atol=atol_T,
                       jac=mat if scale==0 else latent_heat_jac)
        theta=heat.y[:,-1].copy();moisture=wet.y[:,-1].copy()
        rows=np.flatnonzero((times>left)&(times<=right));ts=times[rows]
        T_out[rows]=(heat.sol(ts)+T0)[idx].T;C_out[rows]=wet.sol(ts)[idx].T
        for t in (100.,600.,1800.):
            if left<t<=right:
                profiles[t]=(heat.sol(t)+T0,wet.sol(t))
        if adapter:adapter.cleanup()
        save(out/'progress.json',{'completed_until_s':float(right),'N':N,'latent_scale':scale})
    np.savez_compressed(out/'solution.npz',t_s=times,r_m=grid.r_m[idx],T_K=T_out,C=C_out)
    np.savez_compressed(out/'grid_profiles.npz',t_s=np.array(sorted(profiles)),r_m=grid.r_m,
                        T_K=np.array([profiles[t][0] for t in sorted(profiles)]),
                        C=np.array([profiles[t][1] for t in sorted(profiles)]))
    save(out/'summary.json',{'completed':True,'N':N,'latent_scale':scale,
         'settings':{'method':'Radau','rtol':rtol,'atol_T':atol_T,'atol_C':atol_C},
         'end_s':1800,'mass_mode':'common_wet820'})


def _figure_vapor(b0, b1, target):
    """由本次B0/B1表面轨迹按原气相诊断公式逐秒计算。"""
    import math
    import numpy as np
    from scipy.optimize import brentq
    _initialize_science()
    cfg=json.loads((CODE/'config/model.json').read_text(encoding='utf-8'))
    env=_sections['radial_core'].Environment(ROOT/'inputs/附件',cfg)
    def saturation(t):
        T=t+273.15
        if not 273.16<=T<=647.096:raise ValueError('Outside liquid-water saturation domain')
        tau=1-T/647.096
        a=(-7.85951783,1.84408259,-11.7866497,22.6807411,-15.9618719,1.80122502)
        powers=(1.,1.5,3.,3.5,4.,7.5)
        return 22.064e6*math.exp(647.096/T*sum(c*tau**k for c,k in zip(a,powers)))
    arrays={}; rhod=cfg['property_sets']['P1']['rho_offset']/(1+cfg['initial_state']['moisture_C'])
    hm=cfg['fixed_domain_model']['h_m_m_s']
    for case,path in [('B0_selected_N10240_tight',b0),('B1_N10240_main',b1)]:
        with np.load(path,allow_pickle=False) as z:
            times=z['t_s'];T=z['T_K'][:,-1]-273.15;C=z['C'][:,-1]
        if not np.array_equal(times,np.arange(1801)):raise ValueError('Vapor diagnostic requires integer 0..1800 s')
        room=np.array([env(float(t)) for t in times]); Tr=room[:,0]-273.15; Cr=room[:,1]
        for condition in ('dry_air_humidity_ratio','moist_air_specific_humidity'):
            W=Cr if condition=='dry_air_humidity_ratio' else Cr/(1-Cr)
            pv=101325.*W/(.621945+W)
            if np.any(pv>np.array([saturation(t) for t in Tr])):raise ValueError('Supersaturated bulk assumption')
            dew=np.array([brentq(lambda x:saturation(x)-v,273.16-273.15,tr,
                          xtol=1e-12,rtol=4*np.finfo(float).eps) for v,tr in zip(pv,Tr)])
            fields={'time_s':times,'T_surface_degC':T,'conditional_dewpoint_degC':dew,
                    'pv_bulk_over_psat_surface':pv/np.array([saturation(t) for t in T]),
                    'jw_model_outward_kg_m2_s':rhod*hm*(C-Cr)}
            arrays.update({f'{case}/{condition}/{k}':v for k,v in fields.items()})
    np.savez_compressed(target,**arrays)


def _figure_assumptions(out, N=640):
    """原图所用11个单因素情景与2个非对角算例，实际逐例求解。"""
    from dataclasses import replace
    _initialize_science()
    core,moving=_sections['_vendor_host2_model'],_sections['moving_model']
    cfg=json.loads((CODE/'config/model.json').read_text(encoding='utf-8'))
    ambient=core.AmbientProfile.from_xlsx(ROOT/'inputs/附件/附件1.xlsx')
    settings=core.IntegrationSettings(rtol=1e-9,atol_temperature_K=1e-9,atol_moisture_C=1e-11,max_step_s=150.)
    baseline={};cells=[]
    scenarios=[('ambient_mean_1h',1.,1.,1.,False),('alpha_0p5',.5,1.,1.,False),
               ('alpha_2p0',2.,1.,1.,False),('h_0p9',1.,.9,1.,False),('hm_0p9',1.,1.,.9,False)]
    def solve(prop,geometry,alpha,hfactor,hmfactor,external):
        air=replace(ambient,moisture=ambient.moisture*alpha,tail_moisture=ambient.tail_moisture*alpha)
        properties=core.MaterialProperties(**cfg['property_sets'][prop],name=prop)
        if geometry=='fixed':
            model=core.FixedRadialModel(core.RadialGrid.surface_stretched(N),air,properties,25.*hfactor,8e-7*hmfactor)
        else:
            radius=moving.RadiusProfile.from_xlsx(ROOT/'inputs/附件/附件2.xlsx')
            radius=replace(radius,tail_radius_m=float(radius.radius_m[-1]))
            model=moving.build_model(ROOT/'inputs/附件',CODE/'config/model.json',N=N,
                properties=properties,geometry=radius,ambient=air,h_W_m2_K=25.*hfactor,h_m_m_s=8e-7*hmfactor,
                moisture_boundary_multiplier=moving.initial_dry_density_boundary if external else moving.current_dry_density_boundary)
        trajectory=core.solve_to_event(model,settings,horizon_s=2592000.)
        event=float(trajectory.event_time_s)
        return {'event_root_s':event,'event_root_h':event/3600,'N':N,
                'event_max_C':float(model.unpack(trajectory.final_state)[1].max())}
    for prop,geometry in [('P23','fixed'),('P4','measured')]:
        cases=scenarios+([('boundary_initial_density',1.,1.,1.,True)] if prop=='P4' else [])
        for suffix,alpha,hfactor,hmfactor,external in cases:
            name=f'{prop}_{suffix}'
            print('计算绘图情景：'+name,flush=True)
            record=solve(prop,geometry,alpha,hfactor,hmfactor,external)
            # 目录标识与原绘图接口一致，实际N另存入记录。
            save(out/f'{name}__n640__main/record.json',record)
            if suffix=='ambient_mean_1h':baseline[prop]=record
        cells.append(dict(baseline[prop],property_set=prop,geometry=geometry))
    for prop,geometry in [('P4','fixed'),('P23','measured')]:
        print('计算绘图四格：'+prop+' '+geometry,flush=True)
        cells.append(dict(solve(prop,geometry,1.,1.,1.,False),property_set=prop,geometry=geometry))
    save(out/'factorial_report.json',{'cells':cells,'production_N':N})


def _figure_kinematics(out, N=2560):
    """调用保留的非仿射研究求解器，计算五组beta的事件和全径向最大值。"""
    import numpy as np
    module=_figure_research_module('收缩运动学')
    rows=[]
    for beta in (-1.,-.5,0.,.5,1.):
        folder=out/f'runs/beta_{beta:g}_N{N}'
        if (folder/'record.json').exists():
            record=json.loads((folder/'record.json').read_text(encoding='utf-8'))
            if not record.get('completed'):raise ValueError('Incomplete kinematics run')
        else:
            print(f'计算绘图运动学：beta={beta:g}, N={N}',flush=True)
            record=module.run_case(folder,beta=beta,N=N,inputs=ROOT/'inputs/附件',
                                   contract=CODE/'config/kinematics/model.json')
        with np.load(folder/'internal.npz',allow_pickle=False) as z:
            times=z['t_s'];maxima=z['C'].max(axis=1)
        with np.load(folder/'event_states.npz',allow_pickle=False) as z:
            state=z['root_state']; root_max=float(state[len(state)//2:].max())
        rows.append({'beta':beta,'N':N,'root_s':record['root_s'],
                     'root_h':record['root_h'],'root_max_C':root_max})
        path=out/f'tables/maximum_history_beta_{beta:g}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(path,np.column_stack([times,maxima]),delimiter=',',header='t_s,max_C',comments='')
    save(out/'summary.json',{'rows':rows})


def generate_paper_figure_data(work, question=None, profile='production'):
    """从附件生成绘图数据；只重用同一输入/源码身份下已完成的本次任务。"""
    import shutil
    work=Path(work).resolve()
    if profile not in ('production','smoke'):raise ValueError('Unknown figure profile')
    if work==ROOT or work.is_relative_to(ROOT):raise ValueError('绘图计算目录必须位于支撑包之外')
    if profile=='production':
        # 主计算和绘图共享工作目录，先校验或建立其来源身份，再写入绘图记录。
        execute(Context(work,ROOT/'inputs/附件',question=question or 1),[])
    resource=work/('paper_figures' if profile=='production' else 'paper_figures_smoke')
    resource.mkdir(parents=True,exist_ok=True)
    identity=_figure_inputs_identity(profile)
    identity_file=resource/'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text(encoding='utf-8'))!=identity:
        raise ValueError('绘图计算的源码、输入或环境已变化，请使用新的计算目录')
    save(identity_file,identity)
    executed_tasks=[]
    def task(name, action, products):
        marker=resource/'tasks'/f'{name}.json'
        if marker.exists():
            old=json.loads(marker.read_text(encoding='utf-8'))
            if old.get('identity')!=identity:raise ValueError('Figure task identity differs')
            if all(p.is_file() and digest(p)==old['files'].get(p.relative_to(work).as_posix()) for p in products):
                print('重用本次已完成绘图计算：'+name,flush=True);return
            raise ValueError('已完成绘图任务输出发生变化：'+name)
        print('执行绘图数据任务：'+name,flush=True)
        action()
        executed_tasks.append(name)
        if _figure_inputs_identity(profile)!=identity:raise ValueError('绘图计算期间输入或源码改变')
        save(marker,{'completed':True,'identity':identity,
             'files':{p.relative_to(work).as_posix():digest(p) for p in products}})
    primary={'q1','p23','q4'} if question in (None,1) else {'p23'} if question in (2,3) else {'q4'}
    cases={'q1':1,'p23':2,'q4':4}
    for case in sorted(primary):
        q=cases[case]
        dest=resource/'runs'/('q1/N10240' if case=='q1' else case)
        def run_primary(case=case,q=q,dest=dest):
            if profile=='smoke':
                _figure_primary_smoke(case,dest);return
            command=[sys.executable,'-B',str(CODE/f'问题{q}.py'),'--mode','run',
                     '--root',str(ROOT),'--work',str(work)]
            subprocess.run(command,check=True)
            source=work/'runs/q1_disk/full' if case=='q1' else work/f'runs/endpoint/{case}/finest_candidate/{case}'
            dest.mkdir(parents=True,exist_ok=True)
            for name in (['solution.npz','grid_profiles.npz'] if case=='q1' else ['internal.npz','endpoint.json','metadata.json']):
                shutil.copyfile(source/name,dest/name)
        products=[dest/name for name in (['solution.npz','grid_profiles.npz'] if case=='q1' else ['internal.npz','endpoint.json','metadata.json'])]
        task('primary_'+case,run_primary,products)
    if question in (None,1):
        b1=resource/'latent_B1'
        task('latent_B1',lambda:_figure_latent(b1,N=10240 if profile=='production' else 40),[b1/'solution.npz',b1/'summary.json'])
        task('vapor',lambda:_figure_vapor(resource/'runs/q1/N10240/solution.npz',b1/'solution.npz',resource/'vapor.npz'),[resource/'vapor.npz'])
    if question in (None,4):
        task('assumptions',lambda:_figure_assumptions(resource/'studies',N=640 if profile=='production' else 40),
             [resource/'studies/factorial_report.json'] + [resource/f'studies/{q}_{suffix}__n640__main/record.json'
              for q in ('P23','P4') for suffix in ('ambient_mean_1h','alpha_0p5','alpha_2p0','h_0p9','hm_0p9')]
             +[resource/'studies/P4_boundary_initial_density__n640__main/record.json'])
        task('kinematics',lambda:_figure_kinematics(resource/'kinematics',N=2560 if profile=='production' else 40),
             [resource/'kinematics/summary.json']+[resource/f'kinematics/tables/maximum_history_beta_{b:g}.csv' for b in (-1,-.5,0,.5,1)])
    # 绘图只能使用本函数完成并绑定过的当前目录；不存在随包旧结果回退。
    save(resource/'ready.json',{'completed':True,'question':question,'identity':identity,
         'raw_inputs_only':True,'PDE_solved_by_package':True,
         'profile':profile,'production_settings_used':profile=='production',
         'executed_tasks':executed_tasks,
         'files':{p.relative_to(resource).as_posix():digest(p) for p in resource.rglob('*')
                  if p.is_file() and p.name!='ready.json'}})
    return resource


def build_and_plot_paper(work, question=None, output=None, profile='production'):
    resource=generate_paper_figure_data(work,question,profile)
    result=plot_paper_figures(question,output or Path(work)/('figures' if profile=='production' else 'figures_smoke'),
                             ROOT,resource=resource)
    ready=json.loads((resource/'ready.json').read_text(encoding='utf-8'))
    result['executed_data_tasks']=ready['executed_tasks']
    result['PDE_started']=any(name!='vapor' for name in ready['executed_tasks'])
    return result



def plot_paper_figures(question=None, output=None, package=None, *, resource=None):
    """沿用原图绘制实现，读取本次计算生成并绑定的绘图数据。"""
    package=Path(package or ROOT).resolve()
    output=Path(output or package.parent/'论文原图重绘').resolve()
    if resource is None:
        raise ValueError('必须先由支撑程序计算绘图数据，不读取随包旧结果')
    resource=Path(resource).resolve()
    import json
    ready=json.loads((resource/'ready.json').read_text(encoding='utf-8'))
    if not ready.get('completed') or ready['identity']!=_figure_inputs_identity(ready['profile']):
        raise ValueError('绘图数据身份或完成记录不匹配')
    for name, expected in ready['files'].items():
        if digest(resource/name)!=expected:
            raise ValueError('绘图数据已变化：'+name)
    mapping={1:['f02_inputs','f03_q1','vapor_compatibility'],
             2:['f04_p23'],3:['f04_p23'],
             4:['f05_q4','f06_factorial','nonaffine_kinematics']}
    selected=(['f02_inputs','f03_q1','f04_p23','f05_q4','f06_factorial',
               'vapor_compatibility','nonaffine_kinematics'] if question is None else mapping[question])
    import argparse, hashlib, json, sys, warnings
    sys.dont_write_bytecode=True
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import NullFormatter
    from openpyxl import load_workbook

    BLUE='#245A81'; ORANGE='#A85D25'; GREEN='#437459'; GREY='#575757'
    COLORS=[BLUE,ORANGE,GREEN,GREY]
    STYLES=['-','--',':','-.']
    USED=[]

    def arguments(description):
        p=argparse.ArgumentParser(description=description)
        p.add_argument('--inputs',required=True,type=Path)
        p.add_argument('--runs',required=True,type=Path)
        p.add_argument('--studies',required=True,type=Path)
        p.add_argument('--end-effects',required=True,type=Path)
        p.add_argument('--out',required=True,type=Path)
        p.add_argument('--sources',type=Path,help='Source metadata directory; default is ../data/processed/figure_sources beside --out.')
        args=p.parse_args()
        if args.sources is None:args.sources=args.out.parent/'data'/'processed'/'figure_sources'
        return args

    def style():
        for path in font_manager.findSystemFonts():
            if Path(path).name.lower() in {'simhei.ttf','simsun.ttc'}:
                font_manager.fontManager.addfont(path)
        available={f.name for f in font_manager.fontManager.ttflist}
        family=next((n for n in ['SimHei','Noto Sans CJK SC','Source Han Sans SC','SimSun'] if n in available),None)
        if family is None:raise RuntimeError('A Chinese font is required: SimHei, SimSun or Noto Sans CJK SC.')
        plt.rcParams.update({'font.family':family,'font.size':9,'axes.labelsize':9,'axes.titlesize':9,
            'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8,'axes.unicode_minus':False,
            'lines.linewidth':1.15,'lines.markersize':4,'axes.linewidth':.65,
            'xtick.major.width':.6,'ytick.major.width':.6,'xtick.direction':'out','ytick.direction':'out',
            'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'figure.facecolor':'white',
            'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
            'figure.constrained_layout.use':True})
        return family

    FONT=style()

    def figure(n=1,height=2.85,ratios=None):
        f,a=plt.subplots(1,n,figsize=(6.1,height),squeeze=False,
            gridspec_kw={} if ratios is None else {'width_ratios':ratios})
        f.set_constrained_layout_pads(w_pad=.035,h_pad=.035,wspace=.12,hspace=.04)
        return f,a[0]

    def panel(ax,title,xlabel=None,ylabel=None):
        ax.set_title(title,loc='left',pad=5)
        if xlabel:ax.set_xlabel(xlabel,labelpad=3)
        if ylabel:ax.set_ylabel(ylabel,labelpad=3)
        ax.grid(axis='y',color='.85',linewidth=.4,zorder=0)
        ax.set_axisbelow(True)

    def log_ticks(ax,values,labels=None,axis='y'):
        """Plain glyph labels avoid font-dependent MathText minus substitution."""
        labels=[f'{v:g}' for v in values] if labels is None else labels
        if axis=='y':
            ax.set_yticks(values,labels);ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xticks(values,labels);ax.xaxis.set_minor_formatter(NullFormatter())

    def register(path):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError('Required new-run input is not ready: '+str(path))
        if path not in USED:USED.append(path)
        return path

    def readj(path):return json.loads(register(path).read_text(encoding='utf-8'))

    def npz(path):
        with np.load(register(path)) as d:return {k:d[k] for k in d.files}

    def excel(path,columns):
        book=load_workbook(register(path),read_only=True,data_only=True)
        try:return np.array(list(book['Sheet1'].iter_rows(min_row=2,max_col=columns,values_only=True)),dtype=float)
        finally:book.close()

    def input_file(args,name):
        direct=args.inputs/name
        return direct if direct.exists() else args.inputs/'附件'/name

    def representative_rows(d,wanted_h):
        idx=sorted(set(int(np.argmin(abs(d['t_s']-3600*h))) for h in wanted_h))
        return [(j,float(d['t_s'][j]/3600)) for j in idx]

    def save(fig,args,name,details):
        args.out.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            bounds=fig.bbox
            clipped=[]
            drawn_font_sizes=[]
            # Matplotlib creates ticks outside the view interval but does not draw them.
            # Exclude those labels from the canvas-boundary audit, not real text.
            omitted_tick_ids=set()
            for ax in fig.axes:
                for axis in [ax.xaxis,ax.yaxis]:
                    lo,hi=sorted(axis.get_view_interval())
                    eps=max(abs(hi-lo),1.)*1e-10
                    for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                        if tick.get_loc()<lo-eps or tick.get_loc()>hi+eps:
                            omitted_tick_ids.update([id(tick.label1),id(tick.label2)])
            for item in fig.findobj(match=matplotlib.text.Text):
                if id(item) in omitted_tick_ids or not item.get_visible() or not item.get_text():continue
                drawn_font_sizes.append(float(item.get_fontsize()))
                box=item.get_window_extent(renderer)
                if box.x0<bounds.x0-1 or box.y0<bounds.y0-1 or box.x1>bounds.x1+1 or box.y1>bounds.y1+1:
                    clipped.append(item.get_text())
            fig.savefig(args.out/(name+'.pdf'),format='pdf',metadata={'Title':name,'Author':'','Subject':'New-run data; vector Matplotlib paper figure'})
            fig.savefig(args.out/(name+'.png'),dpi=300)
        missing=[str(w.message) for w in caught if 'Glyph' in str(w.message) or 'constrained_layout' in str(w.message)]
        if missing:raise RuntimeError('Figure font/layout warning: '+'; '.join(missing))
        sources=[]
        for path in USED:
            label=path.name
            for role,base in [('inputs',args.inputs),('runs',args.runs),('studies',args.studies),('end_effects',args.end_effects)]:
                try:label=role+'/'+path.resolve().relative_to(base.resolve()).as_posix();break
                except ValueError:pass
            sources.append({'source':label,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        record={'figure':name,'size_inches':fig.get_size_inches().tolist(),'png_dpi':300,
            'minimum_configured_text_pt':8,'font':FONT,'matplotlib_version':matplotlib.__version__,
            'minimum_drawn_text_pt':min(drawn_font_sizes),
            'execution':{'entry_point':Path(sys.argv[0]).name,
                'integrated_in': 'code/公共依赖.py',
                'inputs_resolved_from_package': True},
            'source_data':sources,'details':details,'vector_pdf':True,'rasterized_artists':False,
            'font_or_layout_warnings':missing,
            'text_outside_figure_bbox':clipped,
            'output_sha256':{ext:hashlib.sha256((args.out/(name+'.'+ext)).read_bytes()).hexdigest() for ext in ['pdf','png']}}
        args.sources.mkdir(parents=True,exist_ok=True)
        (args.sources/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        plt.close(fig)
        print(json.dumps({'figure':name,'sources':len(sources),'font':FONT},ensure_ascii=False))


    def draw_f02_inputs():
        env=excel(input_file(args,'附件1.xlsx'),3);rad=excel(input_file(args,'附件2.xlsx'),2)
        endpoints={q:readj(args.runs/q/'endpoint.json') for q in ['p23','q4']}
        mask=env[:,0]>=10800
        tailT=np.trapezoid(env[mask,1],env[mask,0])/3600
        tailC=np.trapezoid(env[mask,2],env[mask,0])/3600
        fig,axs=figure(2,height=2.75);a,b=axs
        c=a.twinx();c.spines['right'].set_visible(True)
        panel(a,'(a) 环境观测与外推','t/h','T/°C');c.set_ylabel('环境水分/(kg/kg)',labelpad=3,fontsize=9)
        a.axvspan(4,6,color='.94',zorder=-2);a.axvline(4,color='.55',lw=.6)
        lineT,=a.plot(env[:,0]/3600,env[:,1],color=BLUE,label='温度')
        lineC,=c.plot(env[:,0]/3600,env[:,2],color=ORANGE,ls='--',label='环境水分')
        a.plot([4,6],[tailT,tailT],color=BLUE,ls=':')
        c.plot([4,6],[tailC,tailC],color=ORANGE,ls=':')
        a.text(4.12,.17,'末1 h均值\n外推示至6 h',transform=a.get_xaxis_transform(),fontsize=8,color='.25')
        a.set_xlim(0,6);a.set_xticks([0,2,4,6]);a.tick_params(axis='y',labelcolor=BLUE);c.tick_params(axis='y',labelcolor=ORANGE)
        a.set_ylim(float(env[:,1].min()-1),float(env[:,1].max()+5))
        c.set_ylim(max(0,float(env[:,2].min()-.005)),float(env[:,2].max()+.01))
        a.legend([lineT,lineC],['温度','环境水分'],loc='upper center',ncol=2,handlelength=1.2,columnspacing=.8)
        panel(b,'(b) 半径与主算例终点','t/h','R/cm')
        b.plot(rad[:,0]/3600,rad[:,1],color=GREY,label='半径观测')
        for i,q in enumerate(['q4','p23']):
            e=endpoints[q];t=e['reported_time_h'];rv=np.interp(t*3600,rad[:,0],rad[:,1])
            b.axvline(t,ymax=.70,color=COLORS[i],ls=STYLES[i+1],lw=.9,
                label=('Q4' if q=='q4' else 'Q23时刻')+f' {t:.4f} h')
            if q=='q4':b.plot(t,rv,'o',color=COLORS[i])
        b.set_xlim(0,72);b.set_xticks([0,24,48,72]);b.legend(loc='upper right',handlelength=1.4)
        save(fig,args,'f02_inputs',{'tail_rule':'Time-weighted last observed hour mean; plotted extension only to 6h.',
            'tail_temperature_C':float(tailT),'tail_ambient_C':float(tailC),
            'radius_panel_note':'Q23 line denotes only its event time; Q23 itself has fixed radius. The measured shrinkage curve is used by Q4.',
            'endpoint_reports_h':{q:e['reported_time_h'] for q,e in endpoints.items()}})


    def draw_f03_q1():
        base=args.runs/'q1'/'N10240'
        d=npz(base/'grid_profiles.npz');initial=npz(base/'solution.npz')['C'][0,0]
        fig,axs=figure(3,height=2.75)
        for a,title,label in zip(axs,['(a) 径向温度','(b) 径向含水率','(c) 外环净失水'],['T/°C','C/(kg/kg)','净失水占比/%']):panel(a,title,'r/cm',label)
        actual=[]
        for n,t in enumerate([100,600,1800]):
            j=int(np.flatnonzero(d['t_s']==t)[0]);actual.append(t)
            axs[0].plot(d['r_m']*100,d['T_K'][j]-273.15,color=COLORS[n],ls=STYLES[n],label=f'{t} s')
            axs[1].plot(d['r_m']*100,d['C'][j],color=COLORS[n],ls=STYLES[n],label=f'{t} s')
        for a in axs[:2]:a.set_xlim(0,2);a.set_xticks([0,1,2]);a.legend(loc='best',handlelength=1.5)
        rr=d['r_m'];loss=initial-d['C'][int(np.flatnonzero(d['t_s']==1800)[0])]
        edges=np.linspace(0,rr[-1],5);mass=[]
        for left,right in zip(edges[:-1],edges[1:]):
            r=np.r_[left,rr[(rr>left)&(rr<right)],right]
            mass.append(float(np.trapezoid(2*r*np.interp(r,rr,loss),r)/rr[-1]**2))
        shares=100*np.array(mass)/sum(mass)
        axs[2].bar(np.arange(4),shares,color='.83',edgecolor='.3',linewidth=.6,hatch='//',width=.66)
        axs[2].set_xticks(range(4),['0–0.5','0.5–1','1–1.5','1.5–2'],rotation=45,ha='right')
        axs[2].set_xlabel('环带半径/cm');axs[2].set_ylim(0,108)
        for i,value in enumerate(shares):axs[2].text(i,value+2,f'{value:.1f}',ha='center',fontsize=8)
        axs[2].text(.02,.96,'1800 s',transform=axs[2].transAxes,va='top',fontsize=8)
        save(fig,args,'f03_q1',{'profile_times_s':actual,'annular_edges_m':edges.tolist(),
            'annular_net_loss_per_initial_dry_mass':mass,'annular_loss_shares_percent':shares.tolist(),
            'integration':'Independent trapezoidal integral of 2r(C_initial-C_1800)/R0^2 on full new grid; bin boundaries inserted.'})


    def draw_f04_p23():
        d=npz(args.runs/'p23'/'internal.npz');e=readj(args.runs/'p23'/'endpoint.json')
        env=excel(input_file(args,'附件1.xlsx'),3)
        fig,axs=figure(3,height=2.8)
        a,b,c=axs;early=d['t_s']<=10800
        panel(a,'(a) 前3 h温度','t/h','T/°C')
        a.plot(d['t_s'][early]/3600,d['T_K'][early,0]-273.15,color=BLUE,label='中心')
        a.plot(d['t_s'][early]/3600,d['T_K'][early,-1]-273.15,color=ORANGE,ls='--',label='表面')
        em=env[:,0]<=10800;a.plot(env[em,0]/3600,env[em,1],color=GREY,ls=':',label='环境')
        a.set_xlim(0,3);a.set_xticks([0,1,2,3]);a.legend(loc='lower right',handlelength=1.4)
        panel(b,'(b) 全程达标判据','t/h','max C/(kg/kg)')
        b.semilogy(d['t_s']/3600,d['C'].max(axis=1),color=BLUE)
        b.axhline(.15,color='.2',ls='--',lw=.8);b.text(.03,.17,'阈值0.15',transform=b.transAxes,fontsize=8)
        b.plot(e['reported_time_h'],e['reported_max_C'],'o',color=BLUE)
        b.text(.07,.92,f'{e["reported_time_h"]:.4f} h',transform=b.transAxes,va='top',fontsize=8)
        b.set_xlim(0,e['reported_time_h']*1.035);b.set_ylim(.11,3.0);b.set_xticks([0,24,48])
        log_ticks(b,[.15,.5,1,2.5])
        panel(c,'(c) 含水率剖面','r/cm','C/(kg/kg)')
        selected=representative_rows(d,[.5,6,24,e['reported_time_h']])
        for n,(j,t) in enumerate(selected):c.plot(d['xi']*d['R_m'][j]*100,d['C'][j],color=COLORS[n],ls=STYLES[n],label=f'{t:.1f} h')
        c.set_xlim(0,2);c.set_xticks([0,1,2]);c.legend(loc='center left',bbox_to_anchor=(.02,.63),handlelength=1.4)
        save(fig,args,'f04_p23',{'profile_actual_times_h':[t for _,t in selected],
            'reported_time_h':e['reported_time_h'],'reported_max_C_unrounded':e['reported_max_C'],
            'early_plot_source_sampling_s':60,'threshold':.15})


    def draw_f05_q4():
        # The solver writes this completion record after its compressed arrays.
        # Do not open a partially written NPZ just because its name has appeared.
        readj(args.runs/'q4'/'metadata.json')
        d=npz(args.runs/'q4'/'internal.npz');e=readj(args.runs/'q4'/'endpoint.json')
        fig,axs=figure(3,height=3.0,ratios=[1.25,.85,1.0]);a,b,c=axs
        indices=np.unique(np.r_[np.linspace(0,len(d['t_s'])-1,240,dtype=int),len(d['t_s'])-1])
        time=d['t_s'][indices]/3600;r=np.linspace(0,d['R_m'][0],180)
        z=np.full((len(indices),len(r)),np.nan)
        for i,j in enumerate(indices):
            valid=r<=d['R_m'][j]
            z[i,valid]=np.interp(r[valid]/d['R_m'][j],d['xi'],d['C'][j])
        assert np.all(np.isnan(z[r[None,:]>d['R_m'][indices,None]]))
        cmap=plt.get_cmap('cividis').copy();cmap.set_bad('white')
        panel(a,'(a) 真实运动域','t/h','r/cm')
        mesh=a.pcolormesh(time,r*100,z.T,shading='nearest',cmap=cmap,vmin=0,vmax=float(d['C'][0].max()),rasterized=False,edgecolors='face',linewidth=.2,antialiased=False)
        a.fill_between(d['t_s']/3600,d['R_m']*100,d['R_m'][0]*100,color='white',zorder=4)
        a.plot(d['t_s']/3600,d['R_m']*100,color='.15',lw=.7,zorder=5)
        a.text(.65,.92,'域外',transform=a.transAxes,ha='center',fontsize=8,zorder=6)
        a.set_ylim(0,2);a.set_xlim(0,e['reported_time_h']);a.set_xticks([0,24,48])
        cb=fig.colorbar(mesh,ax=a,orientation='horizontal',pad=.10,fraction=.05,aspect=25)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor('face');cb.solids.set_linewidth(.2);cb.solids.set_antialiased(False)
        cb.set_label('C/(kg/kg)',fontsize=8,labelpad=2);cb.ax.tick_params(labelsize=8,length=2)
        panel(b,'(b) 半径收缩','t/h','R/cm')
        b.plot(d['t_s']/3600,d['R_m']*100,color=GREY)
        b.plot(d['t_s'][-1]/3600,d['R_m'][-1]*100,'s',color=GREY)
        b.text(.03,.94,f'终点R={d["R_m"][-1]*100:.3f} cm',transform=b.transAxes,va='top',fontsize=8)
        b.set_xlim(0,e['reported_time_h']*1.04);b.set_xticks([0,24,48])
        panel(c,'(c) 中心与表面','t/h','C/(kg/kg)')
        c.semilogy(d['t_s']/3600,d['C'][:,0],color=BLUE,label='中心')
        c.semilogy(d['t_s']/3600,d['C'][:,-1],color=ORANGE,ls='--',label='表面')
        c.axhline(.15,color='.35',ls=':',lw=.8)
        c.text(.97,.93,f'{e["reported_time_h"]:.4f} h',transform=c.transAxes,ha='right',va='top',fontsize=8)
        c.set_xlim(0,e['reported_time_h']*1.04);c.set_xticks([0,24,48])
        c.legend(loc='upper right',bbox_to_anchor=(1,.78),handlelength=1.4)
        log_ticks(c,[.05,.15,.5,1,2.5])
        save(fig,args,'f05_q4',{'display_field_shape':z.shape,'physical_coordinates':True,'exterior_is_nan_and_white':True,
            'pdf_mesh_edge_treatment':'Vector face-colored 0.2 pt edges and antialiasing off for mesh and colorbar; no data or color scale change.',
            'interpolation':'New internal field sampled linearly in xi at fixed physical radii; exterior mask plus exact saved radius outline.',
            'reported_time_h':e['reported_time_h'],'reported_max_C_unrounded':e['reported_max_C'],'final_radius_m':float(d['R_m'][-1])})


    def draw_f06_factorial():
        factorial=readj(args.studies/'factorial_report.json')
        # Individual immutable records are enough: no dependency on a sibling table
        # directory or the legacy study summary layout.
        def root(cid):return readj(args.studies/(cid+'__n640__main')/'record.json')['event_root_s']
        fig,axs=figure(2,height=2.9,ratios=[.95,1.3]);a,b=axs
        panel(a,'(a) 物性×几何','几何条件','等号穿越时间/h')
        plot_N=factorial.get('production_N',640)
        cells=[v for v in factorial['cells'] if v['N']==plot_N]
        for i,q in enumerate(['P23','P4']):
            y=[next(v['event_root_h'] for v in cells if v['property_set']==q and v['geometry']==g) for g in ['fixed','measured']]
            a.plot([0,1],y,color=COLORS[i],ls=STYLES[i],marker=['o','s'][i],label=q)
            for j,value in enumerate(y):a.annotate(f'{value:.2f}',(j,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        a.set_xticks([0,1],['固定半径','实测收缩']);a.set_xlim(-.18,1.18);a.set_ylim(0,max(v['event_root_h'] for v in cells)*1.2)
        a.legend(loc='upper right',handlelength=1.4);a.text(.02,.98,f'N={plot_N}',transform=a.transAxes,va='top',fontsize=8)
        panel(b,'(b) 关键假设响应','相对基准的时间变化/%',None)
        suffix=['alpha_0p5','alpha_2p0','h_0p9','hm_0p9','boundary_initial_density']
        labels=['Ceq × 0.5','Ceq × 2.0','h × 0.9','hm × 0.9','外通量 × J']
        displayed=[]
        for i,q in enumerate(['P23','P4']):
            base=root(q+'_ambient_mean_1h');vals=[];yy=[]
            for j,s in enumerate(suffix):
                if q=='P23' and s=='boundary_initial_density':continue
                value=100*(root(q+'_'+s)/base-1);vals.append(value);yy.append(j+(i-.5)*.32)
                displayed.append({'question':q,'scenario':s,'delta_percent':value})
            b.barh(yy,vals,height=.29,color=COLORS[i],hatch='' if i==0 else '//',edgecolor='.2',lw=.5,label=q)
            for y,value in zip(yy,vals):
                if abs(value)<.1:
                    b.text(value+.1,y,f'{value:+.3f}%',va='center',fontsize=8,color=COLORS[i])
        b.set_yticks(range(len(labels)),labels);b.invert_yaxis();b.axvline(0,color='.35',lw=.65)
        b.legend(loc='upper right',handlelength=1.4);b.grid(axis='x',color='.87',lw=.4)
        save(fig,args,'f06_factorial',{'factorial_grid_N':plot_N,'factorial_values_are_equality_roots':True,
            'scenario_values':displayed,'basis':'Each property/geometry baseline at matched N640, last-observed-hour environment; deterministic chosen changes, not uncertainty intervals.'})


    @dataclass(frozen=True)
    class RadiusProfile:
        t_s: np.ndarray
        radius_m: np.ndarray
        tail_radius_m: float

        @classmethod
        def from_xlsx(cls, path, tail_radius_m=0.01198):
            book = load_workbook(Path(path), read_only=True, data_only=True)
            try:
                values = np.asarray(list(book["Sheet1"].iter_rows(min_row=2, max_col=2, values_only=True)), dtype=float)
            finally:
                book.close()
            t, radius_cm = values.T
            if not np.isfinite(values).all() or t[0] != 0 or np.any(np.diff(t) <= 0):
                raise ValueError("Invalid radius observation times or values")
            radius = radius_cm * 0.01
            if np.any(radius <= 0) or np.any(np.diff(radius) > 0):
                raise ValueError("Measured radii must be positive and nonincreasing")
            return cls(t, radius, float(tail_radius_m))

        def radius(self, t_s):
            t = np.asarray(t_s, dtype=float)
            if np.any(~np.isfinite(t)) or np.any(t < 0):
                raise ValueError("Radius time must be finite and nonnegative")
            value = np.where(t > self.t_s[-1], self.tail_radius_m, np.interp(t, self.t_s, self.radius_m))
            return float(value) if t.ndim == 0 else value

        def breakpoints(self, start_s, end_s):
            return np.r_[start_s, self.t_s[(self.t_s > start_s) & (self.t_s < end_s)], end_s]

    def draw_nonaffine_kinematics():
        root=args.kinematics
        records=readj(root/"summary.json")["rows"]
        cfg=readj(args.contract);R0=cfg["initial_state"]["radius_m"]
        radius=RadiusProfile.from_xlsx(args.inputs/"附件2.xlsx",cfg["input_radius"]["post_observation"]["radius_m"])
        observed_end=float(cfg["input_radius"]["observed_interval_s"][1])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font=font_manager.FontProperties(fname=font_manager.findfont('SimHei',fallback_to_default=False))
        plt.rcParams.update({'font.family':font.get_name(),'font.size':8,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
        fig,axs=plt.subplots(2,2,figsize=(6.2,4.3),layout='constrained');ax=axs.ravel()
        colors=['#808080','#25689b','#111111','#be5a2a','#989898'];styles=[':','--','-','-.',(0,(4,1,1,1))]
        x=np.linspace(0,1,401);common_t=24*3600.;s=float(radius.radius(common_t))/R0
        handles=[]
        for r,color,style in zip(records,colors,styles):
            beta=r['beta'];b=beta*(1-s);q=1+b*(1-x*x)
            line,=ax[0].plot(x,s*x*q,color=color,ls=style,label=rf'$\beta={beta:g}$'+(' 扩展' if abs(beta)==1 else ''),lw=1.2 if abs(beta)!=1 else .9)
            handles.append(line)
            ax[1].plot(x,1000*s*R0*x*b*(1-x*x),color=color,ls=style,lw=1.2 if abs(beta)!=1 else .9)
            d=np.genfromtxt(root/'tables'/f"maximum_history_beta_{beta:g}.csv",delimiter=',',names=True,encoding='utf-8-sig')
            t=d['t_s']/3600;M=d['max_C'];observed=t<=observed_end/3600
            ax[2].semilogy(t[observed],M[observed],color=color,ls=style,lw=1.2 if abs(beta)!=1 else .9)
            if (~observed).any():
                tail=np.r_[np.flatnonzero(observed)[-1],np.flatnonzero(~observed)]
                ax[2].semilogy(t[tail],M[tail],color=color,ls='--',lw=1.1)
            ax[2].plot(r['root_h'],r['root_max_C'],'o',color=color,mfc='white' if abs(beta)==1 else color,ms=3.3)
            ax[3].plot(beta,r['root_h'],'o',color=color,mfc='white' if abs(beta)==1 else color,ms=5)
        ax[0].set(xlabel=r'$a/R_0$',ylabel=r'$\chi(a,24\,h)/R_0$');ax[0].set_title('(a) 同一表面轨迹的材料映射',fontsize=8)
        ax[0].legend(handles=handles,loc='upper left',frameon=False,handlelength=1.8,labelspacing=.15,borderaxespad=.1)
        ax[1].axhline(0,color='.8',lw=.5);ax[1].set(xlabel=r'$a/R_0$',ylabel='相对仿射位移 / mm');ax[1].set_title('(b) 24 h 材料位移',fontsize=8)
        ax[2].axhline(.15,color='.6',lw=.7,ls=':');ax[2].axvline(72,color='.6',lw=.7,ls=':')
        ax[2].text(72.8,.26,'半径尾部\n外推',fontsize=8,ha='left')
        ax[2].set(xlabel='时间 / h',ylabel=r'$M_C$');ax[2].set_title('(c) 全时域最大含水率',fontsize=8);ax[2].set_ylim(.13,3.)
        ax[2].set_yticks([.15,.2,.4,.6,1.,2.]);ax[2].set_yticklabels(['0.15','0.2','0.4','0.6','1','2']);ax[2].minorticks_off()
        ax[3].axhline(72,color='.65',lw=.7,ls=':');ax[3].text(.95,72.9,'72 h',ha='right',fontsize=8)
        ax[3].set(xlabel=r'$\beta$',ylabel='等号终点 / h',xticks=[-1,-.5,0,.5,1]);ax[3].set_title('(d) 终点对照：空心为扩展情景',fontsize=8)
        for a in ax:a.spines[['top','right']].set_visible(False);a.tick_params(direction='out',length=3,width=.6)
        figs=args.out;figs.mkdir(parents=True,exist_ok=True)
        fig.savefig(figs/'nonaffine_kinematics.pdf',metadata={'Author':None,'CreationDate':None,'ModDate':None});fig.savefig(figs/'nonaffine_kinematics.png',dpi=300);plt.close(fig)


    def draw_vapor_compatibility():
        from matplotlib.text import Text
        arrays=npz(args.vapor)
        data={}
        for case in ('B0_selected_N10240_tight','B1_N10240_main'):
            for condition in ('dry_air_humidity_ratio','moist_air_specific_humidity'):
                data[(case,condition)]={k:arrays[f'{case}/{condition}/{k}'] for k in
                    ('time_s','T_surface_degC','conditional_dewpoint_degC',
                     'pv_bulk_over_psat_surface','jw_model_outward_kg_m2_s')}
        fontpath=next((Path(p) for p in font_manager.findSystemFonts() if Path(p).name.lower()=='simhei.ttf'),None)
        if fontpath is None:raise RuntimeError('The verified SimHei font is required for identical figure typography')
        font_manager.fontManager.addfont(str(fontpath))
        plt.rcParams.update({'font.family':'SimHei','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
            'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.unicode_minus':False,
            'pdf.fonttype':42,'ps.fonttype':42,'path.simplify':False,'axes.linewidth':.6,
            'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
        fig,ax=plt.subplots(1,2,figsize=(157/25.4,2.25),layout='constrained')
        fig.set_constrained_layout_pads(w_pad=.035,h_pad=.025,wspace=.07)
        b0=data[('B0_selected_N10240_tight','dry_air_humidity_ratio')]
        dw=data[('B1_N10240_main','dry_air_humidity_ratio')];dq=data[('B1_N10240_main','moist_air_specific_humidity')];time=dw['time_s']/60
        blue='#245A81';orange='#A85D25';gray='#686868'
        ax[0].plot(time,b0['T_surface_degC'],color=gray,ls=':',lw=1.3,label='B0 表面')
        ax[0].plot(time,dw['T_surface_degC'],color='black',lw=1.4,label='B1 表面')
        ax[0].plot(time,dw['conditional_dewpoint_degC'],color=blue,ls='--',lw=1.25,label='条件露点（W）')
        ax[0].plot(time,dq['conditional_dewpoint_degC'],color=orange,ls='-.',lw=1.15,label='条件露点（q）')
        ax[0].set_title('(a) 表面温度与条件露点',loc='left',pad=5);ax[0].set_ylabel('温度 / °C',labelpad=3);ax[0].set_ylim(6,43);ax[0].set_yticks([10,20,30,40])
        ax[0].legend(loc='upper left',ncol=2,columnspacing=.7,handlelength=1.8,handletextpad=.4,borderaxespad=.1,labelspacing=.35)
        ax[1].plot(time,dw['pv_bulk_over_psat_surface'],color=blue,lw=1.3,label='房内列解释为 W')
        ax[1].plot(time,dq['pv_bulk_over_psat_surface'],color=orange,ls='--',lw=1.3,label='房内列解释为 q')
        ax[1].axhline(1,color='black',ls=':',lw=.9,label='比值 1')
        ax[1].set_title('(b) B1 蒸汽压比',loc='left',pad=5);ax[1].set_ylabel('体相分压 / 表面饱和压',labelpad=3);ax[1].set_ylim(.5,4.6);ax[1].set_yticks([1,2,3,4])
        ax[1].legend(loc='lower right',handlelength=2.,borderaxespad=.15,labelspacing=.4)
        for axis in ax:
            axis.set_xlim(0,30);axis.set_xticks([0,10,20,30]);axis.set_xlabel('时间 / min',labelpad=3)
            axis.grid(axis='y',color='.87',lw=.4);axis.set_axisbelow(True)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();B=fig.bbox;texts=[]
        for t in fig.findobj(Text):
            if not t.get_visible() or not t.get_text():continue
            box=t.get_window_extent(renderer);inside=(box.x0>=B.x0-.1 and box.y0>=B.y0-.1 and box.x1<=B.x1+.1 and box.y1<=B.y1+.1)
            texts.append({'text':t.get_text(),'font_size_pt':t.get_fontsize(),'inside_figure':bool(inside)})
            if t.get_fontsize()<8 or not inside:raise ValueError('Printed font/bbox check failed: '+t.get_text())
        for axis in ax:
            for line in axis.lines:
                if len(line.get_xdata())!=2:assert len(line.get_xdata())==1801
        figure=args.out/'vapor_compatibility.pdf';png=args.out/'vapor_compatibility.png'
        fig.savefig(figure,metadata={'Title':'条件气相边界诊断','Author':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None})
        fig.savefig(png,dpi=300,metadata={'Software':'Matplotlib'});plt.close(fig)


    from types import SimpleNamespace
    args=SimpleNamespace(inputs=package/'inputs/附件', runs=resource/'runs',
        studies=resource/'studies', end_effects=resource, out=output,
        sources=output/'绘图数据记录', contract=package/'code/config/model.json',
        kinematics=resource/'kinematics', vapor=resource/'vapor.npz')
    output.mkdir(parents=True,exist_ok=True)
    drawers={'f02_inputs':draw_f02_inputs,'f03_q1':draw_f03_q1,
             'f04_p23':draw_f04_p23,'f05_q4':draw_f05_q4,'f06_factorial':draw_f06_factorial,
             'vapor_compatibility':draw_vapor_compatibility,
             'nonaffine_kinematics':draw_nonaffine_kinematics}
    products=[]
    for name in selected:
        USED.clear()
        print('绘制论文原图：'+name, flush=True)
        with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
            if name.startswith('f0'):
                style()
            drawers[name]()
        products.extend(str(output/(name+suffix)) for suffix in ('.pdf','.png'))
    return {'figures':products,'PDE_started':False,'data_source':str(resource),'profile':ready['profile']}


def main(question, build_jobs, description, argv=None):
    """Run a question plan, or one isolated scientific step from the same file."""
    global ROOT, CODE
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['--worker']:
        if len(arguments) < 2:
            raise ValueError('--worker 需要指定计算步骤')
        return _run_worker(question, arguments[1], arguments[2:])
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mode', choices=['plan', 'run', 'export', 'check', 'plot'], default='plan')
    parser.add_argument('--root', type=Path, default=ROOT, help='支撑包根目录（code、inputs、results）')
    parser.add_argument('--code', type=Path, default=CODE, help='本次使用的五份源码目录')
    parser.add_argument('--work', type=Path, help='包外计算目录；问题2和3可共用轨迹')
    parser.add_argument('--inputs', type=Path, help='附件目录，默认 ROOT/inputs/附件')
    parser.add_argument('--task', action='append', help='只运行所选任务；可重复指定，保留计划顺序')
    parser.add_argument('--manifest-sha256', help='独立可信渠道取得的源码清单 SHA-256')
    parser.add_argument('--plot-out', type=Path, help='绘图输出目录；默认包外论文原图重绘目录')
    parser.add_argument('--figure-profile', choices=['production','smoke'], default='production', help='smoke仅用于低网格接口测试')
    args = parser.parse_args(arguments)
    try:
        if args.code.resolve() != source_file().parent:
            parser.error('--code 必须对应当前公共依赖文件；请从目标源码目录的题号入口启动')
        ROOT, CODE = args.root.resolve(), args.code.resolve()
        os.environ.update(CUMCM_ROOT=str(ROOT), CUMCM_CODE=str(CODE), CUMCM_QUESTION=str(question))
        if args.manifest_sha256 is not None:
            os.environ['CUMCM_MANIFEST_SHA256'] = args.manifest_sha256
        verify_sources(code=CODE, expected_manifest_sha256=args.manifest_sha256)
        if args.mode == 'plot':
            if args.task or (args.inputs and args.inputs.resolve()!=(ROOT/'inputs/附件').resolve()):
                parser.error('绘图流程使用--root下的原始附件并自动安排前置任务，不接受--task或其他--inputs目录')
            report = build_and_plot_paper(args.work or ROOT.parent/'A题计算结果', question, args.plot_out, args.figure_profile)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.mode == 'check':
            print(json.dumps(check_results([question], root=ROOT), ensure_ascii=False))
            return 0
        if args.mode != 'plan' and args.work is None:
            parser.error('运行或导出需要指定 --work')
        context = Context(args.work or ROOT.parent / 'A题计算结果',
                          args.inputs or ROOT / 'inputs/附件', ROOT, CODE, question)
        jobs = context.output_jobs(question) if args.mode == 'export' else build_jobs(context)
        if args.task:
            unknown = set(args.task) - {job.name for job in jobs}
            if unknown:
                parser.error('不存在的任务：' + ', '.join(sorted(unknown)))
            jobs = [job for job in jobs if job.name in args.task]
        if args.mode == 'plan':
            tasks = []
            for job in jobs:
                mode = 'export' if job.name.startswith(('export_q', 'verify_q')) else 'run'
                command = [sys.executable, '-B', str(CODE / f'问题{question}.py'),
                           '--mode', mode, '--root', str(ROOT), '--code', str(CODE),
                           '--work', str(context.work), '--inputs', str(context.inputs), '--task', job.name]
                tasks.append({'name': job.name, 'command': list(map(str, job.command)),
                              'run_command': command, 'shell_command': _shell_command(command),
                              'outputs': list(map(str, job.products))})
            print(json.dumps({'question': question, 'PDE_started': False, 'cwd': str(ROOT),
                              'source_schema': SOURCE_SCHEMA,
                              'environment': {key: context.environment[key] for key in
                                  (*_ENV_DEFAULTS, 'CUMCM_ROOT', 'CUMCM_CODE', 'CUMCM_QUESTION', 'PYTHONPATH')},
                              'tasks': tasks}, ensure_ascii=False, indent=2))
            return 0
        export_only = args.mode == 'export' or all(
            job.name.startswith(('export_q', 'verify_q')) for job in jobs)
        report = require_environment('export' if export_only else 'q1' if question == 1 else 'core')
        execute(context, jobs, environment_report=report)
        return 0
    except (OSError, ValueError, RuntimeError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'错误：{error}\n')


def public_main(argv=None):
    """Optional standalone source and dependency checks for 公共依赖.py."""
    parser = argparse.ArgumentParser(description='五份可读源码的校验与环境诊断')
    parser.add_argument('--check', action='store_true', help='校验五份源码（默认）')
    parser.add_argument('--manifest-sha256', help='独立可信渠道取得的清单 SHA-256')
    parser.add_argument('--environment', action='store_true', help='报告依赖环境')
    parser.add_argument('--profile', choices=PROFILES, default='core')
    parser.add_argument('--probe-imports', action='store_true')
    parser.add_argument('--plot-paper', action='store_true', help='从附件计算并绘制全部七张图')
    parser.add_argument('--work', type=Path, help='包外计算目录；自动重用同一源码与输入的本次任务')
    parser.add_argument('--figure-profile', choices=['production','smoke'], default='production')
    parser.add_argument('--plot-out', type=Path, help='论文原图输出目录')
    args = parser.parse_args(argv)
    files = verify_sources(expected_manifest_sha256=args.manifest_sha256)
    result = {'sources_passed': True, 'source_schema': SOURCE_SCHEMA, 'source_files': len(files),
              'code': str(CODE), 'root': str(ROOT), 'PDE_started': False,
              'manifest_sha256': digest(CODE.parent / MANIFEST_NAME)}
    if args.plot_paper:
        result['paper_figures'] = build_and_plot_paper(args.work or ROOT.parent/'A题计算结果', output=args.plot_out, profile=args.figure_profile)
        result['PDE_started'] = result['paper_figures']['PDE_started']
    if args.environment:
        result['environment'] = check_environment(args.profile, probe_imports=args.probe_imports)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('environment', {}).get('ok', True) else 1


class _AlgorithmSection:
    """Attribute views of functions in this file, retaining scoped solver hooks.

    These are ordinary objects, not imported or dynamically executed modules.
    All implementations are the readable definitions below. Setting a hook
    updates the corresponding real global used by those functions.
    """
    __slots__ = ('_prefix',)

    def __init__(self, name):
        object.__setattr__(self, '_prefix', '_' + name.lstrip('_') + '__')

    @property
    def __file__(self):
        return __file__

    def __getattr__(self, name):
        try:
            return globals()[self._prefix + name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        key = self._prefix + name
        if key not in globals():
            raise AttributeError(name)
        globals()[key] = value

_science_initialized = False
_exporters_initialized = False
_STEP_FUNCTIONS = {}

_sections = {name: _AlgorithmSection(name) for name in ['_vendor_host2_model', 'moving_model', 'radial_core', 'relative_temperature', 'temperature_reference', 'solve_q1', 'solve_all', 'absolute_temperature_check', 'q1_full_tolerance', 'q1_disk_history_frozen', 'portable_q1_disk', 'portable_q1_compare', 'endpoint_study', 'global_refinement', 'global_refinement_local_clock', 'finest_candidate', 'finest_balance_audit', 'validation_replay', 'compare_validation_replays', 'reproduce_endpoint', 'export_results', 'verify_finest_outputs']}

def _initialize_exporters():
    global _exporters_initialized
    global _export_results__Decimal, _export_results__HEADER, _export_results__MAX_CHANGE, _export_results__Path, _export_results__QUANTUM, _export_results__ROUND_HALF_UP, _export_results____doc__
    global _export_results__annotations, _export_results__argparse, _export_results__datetime, _export_results__export, _export_results__fixed_fields, _export_results__hashlib, _export_results__json
    global _export_results__load, _export_results__locate, _export_results__main, _export_results__np, _export_results__quantized, _export_results__sha256, _export_results__sources
    global _export_results__tempfile, _export_results__timezone, _export_results__xlsxwriter, _verify_finest_outputs__Decimal, _verify_finest_outputs__HEADER, _verify_finest_outputs__Path, _verify_finest_outputs__ROUND_HALF_UP
    global _verify_finest_outputs____doc__, _verify_finest_outputs__argparse, _verify_finest_outputs__json, _verify_finest_outputs__load_workbook, _verify_finest_outputs__main, _verify_finest_outputs__np, _verify_finest_outputs__q4
    global _verify_finest_outputs__sources
    if _exporters_initialized:
        return

    # ===== exporters/export_results.py: readable implementation =====
    _export_results____doc__ = 'Portable final XLSX export from reproduced NPZ arrays (no artifact-tool required).\n\nExample:\n  python export_results.py --q1 q1/solution.npz --q23 p23 --q4 q4 --output-dir results\n\nRequires numpy and XlsxWriter. Uses constant-memory row writing. Internal arrays\nare read without pickle and never changed. Four-decimal numeric output uses\nDecimal(str(float(value))) with ROUND_HALF_UP, never Python round().'
    import argparse as _export_results__argparse
    from datetime import datetime as _export_results__datetime, timezone as _export_results__timezone
    from decimal import Decimal as _export_results__Decimal, ROUND_HALF_UP as _export_results__ROUND_HALF_UP
    import hashlib as _export_results__hashlib
    import json as _export_results__json
    from pathlib import Path as _export_results__Path
    import tempfile as _export_results__tempfile
    import numpy as _export_results__np
    import xlsxwriter as _export_results__xlsxwriter
    _export_results__HEADER = '时间\\到药材中心的距离'
    _export_results__QUANTUM = _export_results__Decimal('0.0001')
    _export_results__MAX_CHANGE = _export_results__Decimal('0.00005')

    def _export_results__sha256(path: _export_results__Path) -> str:
        h = _export_results__hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    def _export_results__load(path: _export_results__Path) -> dict[str, _export_results__np.ndarray]:
        with _export_results__np.load(path, allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}

    def _export_results__quantized(value: float) -> tuple[float, _export_results__Decimal]:
        original = _export_results__Decimal(str(float(value)))
        rounded = original.quantize(_export_results__QUANTUM, rounding=_export_results__ROUND_HALF_UP)
        difference = abs(rounded - original)
        if difference > _export_results__MAX_CHANGE:
            raise ValueError('Decimal output change exceeds 0.00005')
        return (float(rounded), difference)

    def _export_results__locate(base: _export_results__Path, alternatives: tuple[str, ...]) -> _export_results__Path:
        if base.is_file():
            return base
        for rel in alternatives:
            candidate = base / rel
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f'No supported source below {base}; expected {alternatives}')

    def _export_results__fixed_fields(path: _export_results__Path, problem: int):
        d = _export_results__load(path)
        t = _export_results__np.asarray(d['t_s'], dtype=float)
        r = _export_results__np.asarray(d['r_m'], dtype=float)
        keep = t > 0
        if r.shape != (21,) or not _export_results__np.allclose(r, _export_results__np.arange(21) * 0.001, rtol=0, atol=1e-12):
            raise ValueError('Fixed-grid output must have 21 radii 0..0.02 m at 0.001 m steps')
        C = _export_results__np.asarray(d['C'], dtype=float)
        if C.shape != (len(t), 21) or not _export_results__np.isfinite(C).all():
            raise ValueError('C must be a finite (nt,21) array')
        if problem in (1, 2):
            T = _export_results__np.asarray(d['T_C'] if 'T_C' in d else d['T_K'] - 273.15, dtype=float)
            if T.shape != C.shape or not _export_results__np.isfinite(T).all():
                raise ValueError('Temperature must be a finite (nt,21) array')
            fields = {'温度': T[keep], '水分浓度': C[keep]}
        else:
            fields = {'Sheet1': C[keep]}
        return (t[keep], [float(v) for v in r * 100], fields, None)

    def _export_results__sources(problem: int, args):
        if problem == 1:
            path = _export_results__locate(args.q1, ('solution.npz', 'accepted/solution.npz', 'q1/solution.npz'))
            yield (path, _export_results__fixed_fields(path, 1))
        elif problem == 2:
            index_path = _export_results__locate(args.q23, ('data/q2_fixed_grid_1s_index.json', 'p23/data/q2_fixed_grid_1s_index.json'))
            root = index_path.parent.parent
            index = _export_results__json.loads(index_path.read_text(encoding='utf-8'))
            for chunk in index['chunks']:
                rel = _export_results__Path(chunk['file'])
                if rel.is_absolute():
                    raise ValueError('Q2 index chunk paths must be relative to the P23 output directory')
                path = (root / rel).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError('Q2 index path escapes the output directory')
                if not path.is_file():
                    raise FileNotFoundError(path)
                if 'sha256' in chunk and _export_results__sha256(path) != chunk['sha256']:
                    raise ValueError(f'Q2 chunk hash mismatch: {rel}')
                yield (path, _export_results__fixed_fields(path, 2))
        elif problem == 3:
            path = _export_results__locate(args.q23, ('data/q3_fixed_grid_60s.npz', 'p23/data/q3_fixed_grid_60s.npz'))
            yield (path, _export_results__fixed_fields(path, 3))
        elif problem == 4:
            path = _export_results__locate(args.q4, ('result4_candidate_data.npz', 'q4/result4_candidate_data.npz'))
            d = _export_results__load(path)
            t = _export_results__np.asarray(d['t_s'], dtype=float)
            r = _export_results__np.asarray(d['r_sample_m'], dtype=float)
            R = _export_results__np.asarray(d['R_m'], dtype=float)
            C = _export_results__np.asarray(d['C_sample'], dtype=float)
            Cs = _export_results__np.asarray(d['C_surface'], dtype=float)
            if r.shape != (20,) or not _export_results__np.allclose(r, _export_results__np.arange(20) * 0.001, rtol=0, atol=1e-12):
                raise ValueError('Q4 needs 20 fixed radii 0..0.019 m')
            mask = r[None, :] < R[:, None]
            if C.shape != (len(t), 20) or not _export_results__np.array_equal(mask, d['valid_interior']):
                raise ValueError('Q4 domain mask must equal r<R(t) exactly')
            if not _export_results__np.isfinite(C[mask]).all() or not _export_results__np.isnan(C[~mask]).all() or (not _export_results__np.isfinite(Cs).all()):
                raise ValueError('Q4 interior/surface must be finite; all other fixed positions must be NaN')
            if Cs.shape != t.shape or R.shape != t.shape or (not _export_results__np.isfinite(R).all()) or (R <= 0).any():
                raise ValueError('Invalid Q4 radius/surface shape or values')
            keep = t > 0
            yield (path, (t[keep], [float(v) for v in r * 100] + ['药材表面'], {'Sheet1': _export_results__np.c_[C[keep], Cs[keep]]}, R[keep]))

    def _export_results__export(problem: int, args) -> dict:
        destination = args.output_dir / f'result{problem}.xlsx'
        if destination.exists() and (not args.overwrite):
            raise FileExistsError(f'{destination} exists; use --overwrite explicitly')
        names = ['温度', '水分浓度'] if problem in (1, 2) else ['Sheet1']
        source_records = []
        row_number = 1
        previous_time = 0.0
        first_time = None
        final_unrounded_max_C = None
        changes = {name: _export_results__Decimal(0) for name in names}
        counts = {name: {'numeric_results': 0, 'blank_results': 0} for name in names}
        radius_rows = []
        with _export_results__tempfile.TemporaryDirectory(prefix='xlsx_rows_', dir=args.output_dir) as temporary:
            workbook = _export_results__xlsxwriter.Workbook(str(destination), {'constant_memory': True, 'tmpdir': temporary, 'use_zip64': True})
            workbook.set_properties({'author': '', 'title': '', 'company': '', 'comments': ''})
            style = {'font_name': '宋体', 'font_size': 10, 'align': 'center', 'valign': 'vcenter'}
            value_format = workbook.add_format({**style, 'num_format': '0.0000'})
            time_format = workbook.add_format({**style, 'num_format': 'General'})
            header_format = workbook.add_format({**style, 'num_format': '0.0'})
            worksheets = {name: workbook.add_worksheet(name) for name in names}
            for sheet in worksheets.values():
                sheet.set_column(0, 0, 19.625)
                sheet.set_column(1, 21, 9.625)
                sheet.set_default_row(14.1)
                sheet.freeze_panes(1, 1)
            try:
                for source, (times, radii, fields, current_radii) in _export_results__sources(problem, args):
                    before = _export_results__sha256(source)
                    if row_number == 1:
                        for sheet in worksheets.values():
                            sheet.write_row(0, 0, [_export_results__HEADER, *radii], header_format)
                    if not _export_results__np.isfinite(times).all() or not _export_results__np.all(_export_results__np.diff(times) > 0):
                        raise ValueError('Output times must be finite and strictly increasing')
                    for k, t in enumerate(times):
                        if float(t) <= previous_time:
                            raise ValueError('Overlapping or unordered time blocks')
                        previous_time = float(t)
                        first_time = previous_time if first_time is None else first_time
                        for name, sheet in worksheets.items():
                            sheet.write_number(row_number, 0, float(t), time_format)
                            for column, v in enumerate(fields[name][k], 1):
                                if _export_results__np.isnan(v):
                                    if problem != 4:
                                        raise ValueError('NaN only allowed outside Q4 current domain')
                                    sheet.write_blank(row_number, column, None, value_format)
                                    counts[name]['blank_results'] += 1
                                else:
                                    value, difference = _export_results__quantized(v)
                                    changes[name] = max(changes[name], difference)
                                    sheet.write_number(row_number, column, value, value_format)
                                    counts[name]['numeric_results'] += 1
                        if current_radii is not None:
                            radius_rows.append([float(t), float(current_radii[k] * 100)])
                        row_number += 1
                    concentration = fields['水分浓度'] if '水分浓度' in fields else fields['Sheet1']
                    if len(times):
                        final_unrounded_max_C = float(_export_results__np.nanmax(concentration[-1]))
                    if _export_results__sha256(source) != before:
                        raise ValueError('Source changed during export')
                    source_records.append({'file': source.name, 'sha256': before, 'rows': len(times)})
                    print(f'Q{problem}: {row_number - 1} rows written per sheet', flush=True)
            finally:
                workbook.close()
        if radius_rows:
            import csv as csv
            with (args.output_dir / 'surface_radius.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t_s', 'R_cm'])
                writer.writerows(radius_rows)
        return {'file': destination.name, 'sha256': _export_results__sha256(destination), 'bytes': destination.stat().st_size, 'rows_per_sheet': row_number - 1, 'sheet_names': names, 'first_time_s': first_time, 'last_time_s': previous_time, 'final_unrounded_max_C': final_unrounded_max_C, 'counts': counts, 'maximum_decimal_rounding_change': {k: str(v) for k, v in changes.items()}, 'sources': source_records}

    def _export_results__main():
        parser = _export_results__argparse.ArgumentParser(description=_export_results____doc__, formatter_class=_export_results__argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--q1', type=_export_results__Path, help='Q1 solution.npz, accepted directory, or reproduced q1 directory')
        parser.add_argument('--q23', '--p23', dest='q23', type=_export_results__Path, help='P23 directory containing data/, or its reproduced parent')
        parser.add_argument('--q4', type=_export_results__Path, help='Q4 result4_candidate_data.npz or reproduced q4 directory')
        parser.add_argument('--output-dir', type=_export_results__Path, required=True, help='Destination for result1.xlsx ... result4.xlsx')
        parser.add_argument('--problems', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
        parser.add_argument('--overwrite', action='store_true', help='Explicitly allow replacement of existing result files')
        args = parser.parse_args()
        if len(set(args.problems)) != len(args.problems):
            parser.error('Duplicate problem number')
        for problem in args.problems:
            required = args.q1 if problem == 1 else args.q23 if problem in (2, 3) else args.q4
            if required is None:
                parser.error(f'Missing input path for Q{problem}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        files = [_export_results__export(problem, args) for problem in args.problems]
        manifest = {'created_utc': _export_results__datetime.now(_export_results__timezone.utc).isoformat(), 'library': f'XlsxWriter {_export_results__xlsxwriter.__version__}', 'constant_memory': True, 'rounding': 'Decimal(str(float(binary64_value))).quantize(0.0001, ROUND_HALF_UP)', 'source_arrays_modified': False, 'endpoints_modified_for_display': False, 'files': files, 'note': 'Rounded 0.1500 does not prove equality of the unrounded endpoint. Strict drying is audited using model endpoint evidence. ZIP limit applies after including code and required evidence.'}
        (args.output_dir / 'export_manifest.json').write_text(_export_results__json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(_export_results__json.dumps({'files': [{k: f[k] for k in ['file', 'bytes', 'rows_per_sheet']} for f in files]}, ensure_ascii=False))

    # ===== exporters/verify_finest_outputs.py: readable implementation =====
    _verify_finest_outputs____doc__ = 'Read every delivered XLSX cell and compare with unrounded computed arrays.\n\nOutput formatting is checked independently of dry-event strictness. No source\nor workbook is changed. Openpyxl is used for read-only streaming inspection.'
    from pathlib import Path as _verify_finest_outputs__Path
    from decimal import Decimal as _verify_finest_outputs__Decimal, ROUND_HALF_UP as _verify_finest_outputs__ROUND_HALF_UP
    import argparse as _verify_finest_outputs__argparse
    import json as _verify_finest_outputs__json
    import numpy as _verify_finest_outputs__np
    from openpyxl import load_workbook as _verify_finest_outputs__load_workbook
    _verify_finest_outputs__sources = _sections['export_results'].sources
    _verify_finest_outputs__HEADER = _sections['export_results'].HEADER

    def _verify_finest_outputs__q4(value):
        return float(_verify_finest_outputs__Decimal(str(float(value))).quantize(_verify_finest_outputs__Decimal('0.0001'), rounding=_verify_finest_outputs__ROUND_HALF_UP))

    def _verify_finest_outputs__main():
        p = _verify_finest_outputs__argparse.ArgumentParser(description=_verify_finest_outputs____doc__)
        p.add_argument('--q1', type=_verify_finest_outputs__Path, required=True)
        p.add_argument('--q23', type=_verify_finest_outputs__Path, required=True)
        p.add_argument('--q4', type=_verify_finest_outputs__Path, required=True)
        p.add_argument('--results', type=_verify_finest_outputs__Path, required=True)
        p.add_argument('--out', type=_verify_finest_outputs__Path, required=True)
        p.add_argument('--problems', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
        a = p.parse_args()
        reports = []
        for q in a.problems:
            book = _verify_finest_outputs__load_workbook(a.results / f'result{q}.xlsx', read_only=True, data_only=False)
            names = ['温度', '水分浓度'] if q <= 2 else ['Sheet1']
            assert book.sheetnames == names, (q, book.sheetnames)
            iterators = {name: iter(book[name].iter_rows()) for name in names}
            n = 0
            numeric = 0
            blanks = 0
            prev = 0.0
            times_seen = []
            for _, (t, r, fields, R) in _verify_finest_outputs__sources(q, a):
                if n == 0:
                    for name in names:
                        head = next(iterators[name])
                        assert len(head) == 22
                        vals = [v.value for v in head]
                        assert vals[0] == _verify_finest_outputs__HEADER
                        for got, expected in zip(vals[1:], r):
                            assert got == expected if isinstance(expected, str) else abs(float(got) - expected) < 1e-12
                for i, ti in enumerate(t):
                    assert ti > prev
                    prev = float(ti)
                    times_seen.append(prev)
                    for name in names:
                        row = next(iterators[name])
                        assert len(row) == 22
                        assert row[0].value == float(ti), (q, name, n, 'time')
                        for j, (cell, v) in enumerate(zip(row[1:], fields[name][i])):
                            if _verify_finest_outputs__np.isnan(v):
                                assert q == 4 and cell.value is None, (q, name, n, j, 'outside')
                                blanks += 1
                            else:
                                assert cell.data_type == 'n' and cell.value == _verify_finest_outputs__q4(v), (q, name, n, j, cell.value, _verify_finest_outputs__q4(v))
                                assert cell.number_format == '0.0000', (q, name, n, j, 'format')
                                numeric += 1
                    n += 1
            for it in iterators.values():
                try:
                    next(it)
                except StopIteration:
                    pass
                else:
                    raise AssertionError(f'Q{q} unexpected trailing row')
            book.close()
            tt = _verify_finest_outputs__np.array(times_seen)
            step = 1.0 if q <= 2 else 60.0
            assert tt[0] == step
            if len(tt) > 2:
                assert _verify_finest_outputs__np.all(_verify_finest_outputs__np.diff(tt[:-1]) == step)
            assert 0 < tt[-1] - tt[-2] <= step + 1e-08
            if q == 1:
                assert tt[-1] == 1800.0
            else:
                endpoint = _verify_finest_outputs__json.loads(((a.q4 if q == 4 else a.q23) / 'endpoint.json').read_text(encoding='utf-8'))
                assert tt[-1] == endpoint['reported_time_s']
                assert endpoint['reported_max_C'] < 0.15 and endpoint['reported_time_s'] > endpoint['equality_root_s']
            reports.append({'question': q, 'rows_per_sheet': n, 'numeric_values_checked': numeric, 'outside_blanks_checked': blanks, 'first_time_s': tt[0], 'last_time_s': tt[-1], 'cell_values_and_four_decimal_format_passed': True})
            print(_verify_finest_outputs__json.dumps(reports[-1], ensure_ascii=False), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(_verify_finest_outputs__json.dumps({'passed': True, 'reports': reports, 'strictness_scope': 'Model endpoint metadata uses the full piecewise-linear spatial field, not rounded XLSX values.'}, ensure_ascii=False, indent=2), encoding='utf-8')

    _STEP_FUNCTIONS.update({
        'exporters/export_results.py': _export_results__main,
        'exporters/verify_finest_outputs.py': _verify_finest_outputs__main,
    })
    _exporters_initialized = True

def _initialize_science(step=None):
    global _science_initialized
    global _absolute_temperature_check__BDF, _absolute_temperature_check__StreamingBDF, _absolute_temperature_check____doc__, _absolute_temperature_check__core, _absolute_temperature_check__ctypes, _absolute_temperature_check__hashlib, _absolute_temperature_check__json
    global _absolute_temperature_check__memory_status, _absolute_temperature_check__moving, _absolute_temperature_check__np, _absolute_temperature_check__os, _absolute_temperature_check__perf_counter, _absolute_temperature_check__settings, _absolute_temperature_check__solve_event_equation
    global _absolute_temperature_check__writej, _compare_validation_replays__Path, _compare_validation_replays____doc__, _compare_validation_replays__argparse, _compare_validation_replays__gc, _compare_validation_replays__json, _compare_validation_replays__k
    global _compare_validation_replays__main, _compare_validation_replays__np, _compare_validation_replays__os, _compare_validation_replays__physical, _compare_validation_replays__profiles, _compare_validation_replays__read, _compare_validation_replays__sha
    global _compare_validation_replays__source_identity, _compare_validation_replays__sys, _compare_validation_replays__write, _endpoint_study__Path, _endpoint_study____doc__, _endpoint_study__hashlib, _endpoint_study__json
    global _endpoint_study__sha, _endpoint_study__write, _finest_balance_audit__Observer, _finest_balance_audit__Path, _finest_balance_audit____doc__, _finest_balance_audit__argparse, _finest_balance_audit__candidate
    global _finest_balance_audit__clock, _finest_balance_audit__json, _finest_balance_audit__k, _finest_balance_audit__leggauss, _finest_balance_audit__main, _finest_balance_audit__np, _finest_balance_audit__os
    global _finest_balance_audit__sha, _finest_balance_audit__source_file, _finest_balance_audit__source_identity, _finest_balance_audit__sys, _finest_balance_audit__weakref, _finest_balance_audit__write, _finest_candidate__CompleteRecorder
    global _finest_candidate__Decimal, _finest_candidate__Path, _finest_candidate__PchipInterpolator, _finest_candidate__ROUND_CEILING, _finest_candidate____doc__, _finest_candidate__argparse, _finest_candidate__clock
    global _finest_candidate__complete_class, _finest_candidate__gc, _finest_candidate__json, _finest_candidate__k, _finest_candidate__main, _finest_candidate__np, _finest_candidate__os
    global _finest_candidate__perf_counter, _finest_candidate__sha, _finest_candidate__source_identity, _finest_candidate__sys, _finest_candidate__ticks, _finest_candidate__write, _global_refinement__Path
    global _global_refinement__Recorder, _global_refinement____doc__, _global_refinement__hashlib, _global_refinement__json, _global_refinement__np, _global_refinement__sha, _global_refinement__write
    global _global_refinement_local_clock__BDF, _global_refinement_local_clock__Path, _global_refinement_local_clock__Recorder, _global_refinement_local_clock__SimpleNamespace, _global_refinement_local_clock____doc__, _global_refinement_local_clock__argparse, _global_refinement_local_clock__json
    global _global_refinement_local_clock__k, _global_refinement_local_clock__local_class, _global_refinement_local_clock__main, _global_refinement_local_clock__np, _global_refinement_local_clock__os, _global_refinement_local_clock__sha, _global_refinement_local_clock__solve_event_equation
    global _global_refinement_local_clock__source_identity, _global_refinement_local_clock__sys, _global_refinement_local_clock__write, _moving_model__AmbientProfile, _moving_model__AmbientValues, _moving_model__Callable, _moving_model__CallableRadius
    global _moving_model__ConstantAmbient, _moving_model__FixedRadius, _moving_model__GeometryAwareAmbient, _moving_model__IntegrationSettings, _moving_model__MaterialProperties, _moving_model__MovingRadialModel, _moving_model__Path
    global _moving_model__PchipInterpolator, _moving_model__PiecewiseTrajectory, _moving_model__RadialGrid, _moving_model__RadiusProfile, _moving_model____doc__, _moving_model__annotations, _moving_model__build_model
    global _moving_model__current_dry_density_boundary, _moving_model__dataclass, _moving_model__field, _moving_model__initial_dry_density_boundary, _moving_model__json, _moving_model__load_workbook, _moving_model__np
    global _moving_model__properties_from_contract, _moving_model__sample_physical, _moving_model__solve_moving_to_event, _moving_model__solve_moving_to_time, _moving_model__upstream, _portable_q1_compare__Path, _portable_q1_compare____doc__
    global _portable_q1_compare__all_true, _portable_q1_compare__argparse, _portable_q1_compare__comparison, _portable_q1_compare__hashlib, _portable_q1_compare__json, _portable_q1_compare__key, _portable_q1_compare__main
    global _portable_q1_compare__os, _portable_q1_compare__read, _portable_q1_compare__sha, _portable_q1_compare__sys, _portable_q1_compare__traceback, _portable_q1_compare__write, _portable_q1_disk__Path
    global _portable_q1_disk____doc__, _portable_q1_disk__argparse, _portable_q1_disk__disk, _portable_q1_disk__identities, _portable_q1_disk__inspect, _portable_q1_disk__invoke, _portable_q1_disk__json
    global _portable_q1_disk__key, _portable_q1_disk__main, _portable_q1_disk__module, _portable_q1_disk__np, _portable_q1_disk__os, _portable_q1_disk__protected_sources, _portable_q1_disk__run
    global _portable_q1_disk__scipy, _portable_q1_disk__sys, _portable_q1_disk__traceback, _portable_q1_disk__validate, _q1_disk_history_frozen__DiskInterpolant, _q1_disk_history_frozen__DiskRadau, _q1_disk_history_frozen__DiskStates
    global _q1_disk_history_frozen__OdeSolution, _q1_disk_history_frozen__Path, _q1_disk_history_frozen__Radau, _q1_disk_history_frozen__RadauDenseOutput, _q1_disk_history_frozen__SETTINGS, _q1_disk_history_frozen__SimpleNamespace, _q1_disk_history_frozen____doc__
    global _q1_disk_history_frozen__argparse, _q1_disk_history_frozen__array_sha, _q1_disk_history_frozen__ctypes, _q1_disk_history_frozen__datetime, _q1_disk_history_frozen__hashlib, _q1_disk_history_frozen__json, _q1_disk_history_frozen__key
    global _q1_disk_history_frozen__np, _q1_disk_history_frozen__os, _q1_disk_history_frozen__reference_capture, _q1_disk_history_frozen__scipy, _q1_disk_history_frozen__scipy_solve_ivp, _q1_disk_history_frozen__sha, _q1_disk_history_frozen__shutil
    global _q1_disk_history_frozen__sys, _q1_disk_history_frozen__time, _q1_disk_history_frozen__timezone, _q1_disk_history_frozen__traceback, _q1_disk_history_frozen__wait_for_resources, _q1_disk_history_frozen__write, _q1_full_tolerance__Decimal
    global _q1_full_tolerance__OdeSolution, _q1_full_tolerance__Path, _q1_full_tolerance__ROUND_HALF_UP, _q1_full_tolerance__Radau, _q1_full_tolerance__SETTINGS, _q1_full_tolerance____doc__, _q1_full_tolerance___key
    global _q1_full_tolerance__argparse, _q1_full_tolerance__compare, _q1_full_tolerance__csv, _q1_full_tolerance__datetime, _q1_full_tolerance__difference, _q1_full_tolerance__hashlib, _q1_full_tolerance__inspect
    global _q1_full_tolerance__json, _q1_full_tolerance__main, _q1_full_tolerance__np, _q1_full_tolerance__observe_progress, _q1_full_tolerance__os, _q1_full_tolerance__perf_counter, _q1_full_tolerance__progress_worker
    global _q1_full_tolerance__protected_sources, _q1_full_tolerance__read_temperature_data, _q1_full_tolerance__readj, _q1_full_tolerance__reference_temperature, _q1_full_tolerance__rounding_mismatches, _q1_full_tolerance__scipy_solve_ivp, _q1_full_tolerance__sha
    global _q1_full_tolerance__solve_q1, _q1_full_tolerance__source_identity, _q1_full_tolerance__sys, _q1_full_tolerance__threading, _q1_full_tolerance__timezone, _q1_full_tolerance__writej, _radial_core__Environment
    global _radial_core__NodeRadialGrid, _radial_core__Path, _radial_core__RadiusHistory, _radial_core____doc__, _radial_core___read_numeric_table, _radial_core__annotations, _radial_core__dataclass
    global _radial_core__diags, _radial_core__diffusion_rhs, _radial_core__face_arithmetic, _radial_core__hashlib, _radial_core__json, _radial_core__linear_diffusion_matrix, _radial_core__load_contract
    global _radial_core__maximum_moisture_event, _radial_core__np, _radial_core__openpyxl, _radial_core__property_values, _radial_core__sha256, _radial_core__tridiagonal_sparsity, _relative_temperature__RelativeTemperatureModel
    global _relative_temperature__RelativeTrajectory, _relative_temperature____doc__, _relative_temperature__dataclass, _relative_temperature__m, _relative_temperature__np, _relative_temperature__wrap, _reproduce_endpoint__Path
    global _reproduce_endpoint____doc__, _reproduce_endpoint__argparse, _reproduce_endpoint__dispatch_step, _reproduce_endpoint__json, _reproduce_endpoint__k, _reproduce_endpoint__main, _reproduce_endpoint__os
    global _reproduce_endpoint__source_identity, _reproduce_endpoint__subprocess, _reproduce_endpoint__sys, _solve_all__Decimal, _solve_all__Path, _solve_all__RelativeTemperatureModel, _solve_all____doc__
    global _solve_all__argparse, _solve_all__balance, _solve_all__compact_sample, _solve_all__csv, _solve_all__fields, _solve_all__hashlib, _solve_all__json
    global _solve_all__leggauss, _solve_all__m, _solve_all__main, _solve_all__math, _solve_all__np, _solve_all__reference_temperature, _solve_all__run_case
    global _solve_all__save_csv, _solve_all__settings, _solve_all__sha, _solve_all__simpson, _solve_all__solve_first, _solve_all__solve_long, _solve_all__solve_q1
    global _solve_all__source_identity, _solve_all__sys, _solve_all__time, _solve_all__times, _solve_all__wrap, _solve_all__write, _solve_q1__Environment
    global _solve_q1__NodeRadialGrid, _solve_q1__Path, _solve_q1____doc__, _solve_q1__annotations, _solve_q1__argparse, _solve_q1__args, _solve_q1__datetime
    global _solve_q1__diffusion_rhs, _solve_q1__face_arithmetic, _solve_q1__json, _solve_q1__leggauss, _solve_q1__linear_diffusion_matrix, _solve_q1__load_contract, _solve_q1__np
    global _solve_q1__parser, _solve_q1__platform, _solve_q1__scipy, _solve_q1__sha256, _solve_q1__solve_ivp, _solve_q1__solve_q1, _solve_q1__source_identity
    global _solve_q1__time, _solve_q1__timezone, _solve_q1__tridiagonal_sparsity, _temperature_reference__ALPHA_M2_S, _temperature_reference__BIOT, _temperature_reference__CP_J_KG_K, _temperature_reference__H_W_M2_K
    global _temperature_reference__INITIAL_T_K, _temperature_reference__K_W_M_K, _temperature_reference__Path, _temperature_reference__RADIUS_M, _temperature_reference__RHO_KG_M3, _temperature_reference____doc__, _temperature_reference___difference_summary
    global _temperature_reference___modal_basis, _temperature_reference___self_checks, _temperature_reference___sha256, _temperature_reference___vector, _temperature_reference__annotations, _temperature_reference__argparse, _temperature_reference__brentq
    global _temperature_reference__datetime, _temperature_reference__hashlib, _temperature_reference__j0, _temperature_reference__j1, _temperature_reference__jn_zeros, _temperature_reference__json, _temperature_reference__lru_cache
    global _temperature_reference__main, _temperature_reference__np, _temperature_reference__openpyxl, _temperature_reference__perf_counter, _temperature_reference__platform, _temperature_reference__read_temperature_data, _temperature_reference__reference_temperature
    global _temperature_reference__scipy, _temperature_reference__source_identity, _temperature_reference__sys, _temperature_reference__timezone, _validation_replay__Path, _validation_replay__Recorder, _validation_replay____doc__
    global _validation_replay__argparse, _validation_replay__gc, _validation_replay__json, _validation_replay__k, _validation_replay__local_class, _validation_replay__main, _validation_replay__np
    global _validation_replay__os, _validation_replay__sha, _validation_replay__source_identity, _validation_replay__sys, _validation_replay__write, _vendor_host2_model__AmbientProfile, _vendor_host2_model__AmbientValues
    global _vendor_host2_model__ArrayLike, _vendor_host2_model__Callable, _vendor_host2_model__ConstantAmbient, _vendor_host2_model__FixedRadialModel, _vendor_host2_model__FloatArray, _vendor_host2_model__IntegrationSettings, _vendor_host2_model__Iterable
    global _vendor_host2_model__MaterialProperties, _vendor_host2_model__NDArray, _vendor_host2_model__Path, _vendor_host2_model__PiecewiseTrajectory, _vendor_host2_model__RadialGrid, _vendor_host2_model__SegmentSolution, _vendor_host2_model__Sequence
    global _vendor_host2_model____doc__, _vendor_host2_model___event_function, _vendor_host2_model___solve_segment, _vendor_host2_model__annotations, _vendor_host2_model__arithmetic_mean, _vendor_host2_model__csr_matrix, _vendor_host2_model__dataclass
    global _vendor_host2_model__harmonic_mean, _vendor_host2_model__lil_matrix, _vendor_host2_model__load_workbook, _vendor_host2_model__np, _vendor_host2_model__refine_event_bracket, _vendor_host2_model__requested_time_grid, _vendor_host2_model__sample_radii
    global _vendor_host2_model__selected_times_with_final, _vendor_host2_model__solve_ivp, _vendor_host2_model__solve_to_event, _vendor_host2_model__solve_to_time
    _initialize_exporters()
    if step and step.startswith("exporters/"):
        return
    if _science_initialized:
        return

    # ===== solvers/_vendor_host2_model.py: readable implementation =====
    _vendor_host2_model____doc__ = 'Conservative radial finite-volume model for CUMCM 2026 problem A, Q2/Q3.\n\nThe implementation follows baseline model contract v0.1.  Unknowns are nodal\nvalues on a uniform radial grid including the centre and the physical surface.\nControl-volume weights are exact integrals of ``r dr`` around those nodes.\nVariable diffusivity and conductivity are kept inside the divergence operator,\nwith arithmetic face values (matching the host-1 common API) and Robin fluxes\napplied at the outer face.'
    from dataclasses import dataclass as _vendor_host2_model__dataclass
    from pathlib import Path as _vendor_host2_model__Path
    from typing import Callable as _vendor_host2_model__Callable, Iterable as _vendor_host2_model__Iterable, Sequence as _vendor_host2_model__Sequence
    import numpy as _vendor_host2_model__np
    from numpy.typing import ArrayLike as _vendor_host2_model__ArrayLike, NDArray as _vendor_host2_model__NDArray
    from openpyxl import load_workbook as _vendor_host2_model__load_workbook
    from scipy.integrate import solve_ivp as _vendor_host2_model__solve_ivp
    from scipy.sparse import csr_matrix as _vendor_host2_model__csr_matrix, lil_matrix as _vendor_host2_model__lil_matrix
    _vendor_host2_model__FloatArray = _vendor_host2_model__NDArray[_vendor_host2_model__np.float64]

    @_vendor_host2_model__dataclass(frozen=True)
    class _vendor_host2_model__AmbientValues:
        temperature_K: float | FloatArray
        moisture_eq: float | FloatArray
        extrapolated: bool | NDArray[np.bool_]

    @_vendor_host2_model__dataclass(frozen=True)
    class _vendor_host2_model__AmbientProfile:
        """Measured piecewise-linear environment plus the registered right tail."""
        t_s: FloatArray
        temperature_C: FloatArray
        moisture: FloatArray
        tail_temperature_C: float
        tail_moisture: float

        @classmethod
        def from_xlsx(cls, path: str | Path) -> 'AmbientProfile':
            workbook = _vendor_host2_model__load_workbook(filename=_vendor_host2_model__Path(path), read_only=True, data_only=True)
            try:
                sheet = workbook['Sheet1']
                rows = list(sheet.iter_rows(min_row=2, max_col=3, values_only=True))
            finally:
                workbook.close()
            values = _vendor_host2_model__np.asarray(rows, dtype=_vendor_host2_model__np.float64)
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(f'Unexpected environment table shape: {values.shape}')
            t_s, temperature_C, moisture = values.T
            if not _vendor_host2_model__np.all(_vendor_host2_model__np.isfinite(values)):
                raise ValueError('Environment input contains non-finite values')
            if not _vendor_host2_model__np.all(_vendor_host2_model__np.diff(t_s) > 0.0):
                raise ValueError('Environment times must be strictly increasing')
            if t_s[0] != 0.0 or t_s[-1] != 14400.0:
                raise ValueError('Environment observations must cover 0..14400 s')
            tail_mask = t_s >= 10800.0
            tail_temperature_C = float(_vendor_host2_model__np.trapezoid(temperature_C[tail_mask], t_s[tail_mask]) / 3600.0)
            tail_moisture = float(_vendor_host2_model__np.trapezoid(moisture[tail_mask], t_s[tail_mask]) / 3600.0)
            return cls(t_s=t_s, temperature_C=temperature_C, moisture=moisture, tail_temperature_C=tail_temperature_C, tail_moisture=tail_moisture)

        @property
        def breakpoint_s(self) -> float:
            return float(self.t_s[-1])

        def values(self, t_s: ArrayLike, *, right_of_breakpoint: bool=False) -> AmbientValues:
            query = _vendor_host2_model__np.asarray(t_s, dtype=_vendor_host2_model__np.float64)
            if _vendor_host2_model__np.any(~_vendor_host2_model__np.isfinite(query)):
                raise ValueError('Environment query time must be finite')
            observed = query <= self.breakpoint_s
            if right_of_breakpoint:
                observed = query < self.breakpoint_s
            temperature_C = _vendor_host2_model__np.interp(query, self.t_s, self.temperature_C)
            moisture_eq = _vendor_host2_model__np.interp(query, self.t_s, self.moisture)
            temperature_C = _vendor_host2_model__np.where(observed, temperature_C, self.tail_temperature_C)
            moisture_eq = _vendor_host2_model__np.where(observed, moisture_eq, self.tail_moisture)
            extrapolated = ~observed
            if query.ndim == 0:
                return _vendor_host2_model__AmbientValues(temperature_K=float(temperature_C + 273.15), moisture_eq=float(moisture_eq), extrapolated=bool(extrapolated))
            return _vendor_host2_model__AmbientValues(temperature_K=_vendor_host2_model__np.asarray(temperature_C + 273.15, dtype=_vendor_host2_model__np.float64), moisture_eq=_vendor_host2_model__np.asarray(moisture_eq, dtype=_vendor_host2_model__np.float64), extrapolated=_vendor_host2_model__np.asarray(extrapolated, dtype=bool))

        def breakpoints(self, start_s: float, end_s: float) -> FloatArray:
            inner = self.t_s[(self.t_s > start_s) & (self.t_s < end_s)]
            return _vendor_host2_model__np.concatenate([_vendor_host2_model__np.asarray([start_s], dtype=_vendor_host2_model__np.float64), inner, _vendor_host2_model__np.asarray([end_s], dtype=_vendor_host2_model__np.float64)])

    @_vendor_host2_model__dataclass(frozen=True)
    class _vendor_host2_model__ConstantAmbient:
        temperature_K: float
        moisture_eq: float
        breakpoint_s: float = _vendor_host2_model__np.inf

        def values(self, t_s: ArrayLike, *, right_of_breakpoint: bool=False) -> AmbientValues:
            query = _vendor_host2_model__np.asarray(t_s, dtype=_vendor_host2_model__np.float64)
            if query.ndim == 0:
                return _vendor_host2_model__AmbientValues(self.temperature_K, self.moisture_eq, False)
            shape = query.shape
            return _vendor_host2_model__AmbientValues(_vendor_host2_model__np.full(shape, self.temperature_K, dtype=_vendor_host2_model__np.float64), _vendor_host2_model__np.full(shape, self.moisture_eq, dtype=_vendor_host2_model__np.float64), _vendor_host2_model__np.zeros(shape, dtype=bool))

        def breakpoints(self, start_s: float, end_s: float) -> FloatArray:
            return _vendor_host2_model__np.asarray([start_s, end_s], dtype=_vendor_host2_model__np.float64)

    @_vendor_host2_model__dataclass(frozen=True)
    class _vendor_host2_model__MaterialProperties:
        rho_offset: float
        rho_C_coefficient: float
        cp_offset: float
        cp_fraction_coefficient: float
        k_offset: float
        k_fraction_coefficient: float
        D_prefactor: float
        D_moisture_exponent: float
        D_temperature_exponent_K: float
        name: str = 'custom'

        @classmethod
        def p23(cls) -> 'MaterialProperties':
            return cls(rho_offset=650.0, rho_C_coefficient=128.0, cp_offset=1450.0, cp_fraction_coefficient=2736.0, k_offset=0.21, k_fraction_coefficient=0.38, D_prefactor=0.0024, D_moisture_exponent=0.45, D_temperature_exponent_K=3850.0, name='P23')

        @classmethod
        def constant(cls, *, rho: float, cp: float, k: float, diffusivity: float, name: str='constant') -> 'MaterialProperties':
            return cls(rho, 0.0, cp, 0.0, k, 0.0, diffusivity, 0.0, 0.0, name)

        def evaluate(self, C: ArrayLike, T_K: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
            moisture = _vendor_host2_model__np.asarray(C, dtype=_vendor_host2_model__np.float64)
            temperature = _vendor_host2_model__np.asarray(T_K, dtype=_vendor_host2_model__np.float64)
            if _vendor_host2_model__np.any(~_vendor_host2_model__np.isfinite(moisture)) or _vendor_host2_model__np.any(~_vendor_host2_model__np.isfinite(temperature)):
                raise FloatingPointError('State contains non-finite values')
            if _vendor_host2_model__np.any(moisture < 0.0):
                minimum = float(_vendor_host2_model__np.min(moisture))
                raise FloatingPointError(f'Negative moisture encountered: min(C)={minimum:.17g}')
            rho = self.rho_offset + self.rho_C_coefficient * moisture
            cp = self.cp_offset + self.cp_fraction_coefficient * moisture / (1.0 + moisture)
            conductivity = self.k_offset + self.k_fraction_coefficient * moisture / (1.0 + moisture)
            diffusivity = _vendor_host2_model__np.zeros_like(moisture, dtype=_vendor_host2_model__np.float64)
            if self.D_moisture_exponent == 0.0:
                diffusivity[...] = self.D_prefactor * _vendor_host2_model__np.exp(-self.D_temperature_exponent_K / temperature)
            else:
                positive = moisture > 0.0
                if _vendor_host2_model__np.any(positive):
                    diffusivity[positive] = self.D_prefactor * _vendor_host2_model__np.exp(-self.D_moisture_exponent / moisture[positive] - self.D_temperature_exponent_K / temperature[positive])
            if _vendor_host2_model__np.any(rho <= 0.0) or _vendor_host2_model__np.any(cp <= 0.0) or _vendor_host2_model__np.any(conductivity <= 0.0):
                raise FloatingPointError('A thermal material property is non-positive')
            if _vendor_host2_model__np.any(diffusivity < 0.0) or _vendor_host2_model__np.any(~_vendor_host2_model__np.isfinite(diffusivity)):
                raise FloatingPointError('Diffusivity is invalid')
            return (rho, cp, conductivity, diffusivity)

    @_vendor_host2_model__dataclass(frozen=True)
    class _vendor_host2_model__RadialGrid:
        intervals: int
        radius_m: float
        r_m: FloatArray
        node_spacing_m: FloatArray
        face_r_m: FloatArray
        volume_weights_m2: FloatArray

        @classmethod
        def uniform(cls, intervals: int, radius_m: float=0.02) -> 'RadialGrid':
            if intervals < 4:
                raise ValueError('At least four radial intervals are required')
            if radius_m <= 0.0:
                raise ValueError('Radius must be positive')
            r_m = _vendor_host2_model__np.linspace(0.0, radius_m, intervals + 1, dtype=_vendor_host2_model__np.float64)
            node_spacing_m = _vendor_host2_model__np.diff(r_m)
            face_r_m = (r_m[:-1] + r_m[1:]) / 2.0
            left = _vendor_host2_model__np.empty_like(r_m)
            right = _vendor_host2_model__np.empty_like(r_m)
            left[0] = 0.0
            left[1:] = face_r_m
            right[:-1] = face_r_m
            right[-1] = radius_m
            volume_weights_m2 = 0.5 * (right * right - left * left)
            if not _vendor_host2_model__np.isclose(_vendor_host2_model__np.sum(volume_weights_m2), 0.5 * radius_m * radius_m):
                raise AssertionError('Radial control-volume weights do not tile the domain')
            return cls(intervals, radius_m, r_m, node_spacing_m, face_r_m, volume_weights_m2)

        @classmethod
        def surface_stretched(cls, intervals: int, radius_m: float=0.02, *, power: float=2.0) -> 'RadialGrid':
            """Power-law grid clustered near the Robin surface.

            ``r/R = 1 - (1-xi)**power`` with uniform ``xi``.  Power two is the
            production default because it resolves the micrometre-scale dry surface
            layer while retaining a moderate centre spacing.
            """
            if intervals < 4:
                raise ValueError('At least four radial intervals are required')
            if radius_m <= 0.0 or power <= 1.0:
                raise ValueError('A positive radius and stretch power > 1 are required')
            xi = _vendor_host2_model__np.linspace(0.0, 1.0, intervals + 1, dtype=_vendor_host2_model__np.float64)
            r_m = radius_m * (1.0 - (1.0 - xi) ** power)
            r_m[-1] = radius_m
            node_spacing_m = _vendor_host2_model__np.diff(r_m)
            if _vendor_host2_model__np.any(node_spacing_m <= 0.0):
                raise ValueError('Stretched grid nodes are not strictly increasing')
            face_r_m = (r_m[:-1] + r_m[1:]) / 2.0
            left = _vendor_host2_model__np.empty_like(r_m)
            right = _vendor_host2_model__np.empty_like(r_m)
            left[0] = 0.0
            left[1:] = face_r_m
            right[:-1] = face_r_m
            right[-1] = radius_m
            volume_weights_m2 = 0.5 * (right * right - left * left)
            if not _vendor_host2_model__np.isclose(_vendor_host2_model__np.sum(volume_weights_m2), 0.5 * radius_m * radius_m):
                raise AssertionError('Radial control-volume weights do not tile the domain')
            return cls(intervals, radius_m, r_m, node_spacing_m, face_r_m, volume_weights_m2)

        @property
        def minimum_spacing_m(self) -> float:
            return float(_vendor_host2_model__np.min(self.node_spacing_m))

        @property
        def maximum_spacing_m(self) -> float:
            return float(_vendor_host2_model__np.max(self.node_spacing_m))

    def _vendor_host2_model__harmonic_mean(left: _vendor_host2_model__FloatArray, right: _vendor_host2_model__FloatArray) -> _vendor_host2_model__FloatArray:
        denominator = left + right
        result = _vendor_host2_model__np.zeros_like(denominator, dtype=_vendor_host2_model__np.float64)
        positive = denominator > 0.0
        result[positive] = 2.0 * left[positive] * right[positive] / denominator[positive]
        return result

    def _vendor_host2_model__arithmetic_mean(left: _vendor_host2_model__FloatArray, right: _vendor_host2_model__FloatArray) -> _vendor_host2_model__FloatArray:
        return 0.5 * (left + right)

    @_vendor_host2_model__dataclass
    class _vendor_host2_model__FixedRadialModel:
        grid: RadialGrid
        ambient: AmbientProfile | ConstantAmbient
        properties: MaterialProperties
        h_W_m2_K: float = 25.0
        h_m_m_s: float = 8e-07

        @property
        def nr(self) -> int:
            return self.grid.r_m.size

        def initial_state(self, temperature_K: float=301.15, moisture_C: float=2.55) -> FloatArray:
            return _vendor_host2_model__np.concatenate([_vendor_host2_model__np.full(self.nr, temperature_K, dtype=_vendor_host2_model__np.float64), _vendor_host2_model__np.full(self.nr, moisture_C, dtype=_vendor_host2_model__np.float64)])

        def unpack(self, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
            state = _vendor_host2_model__np.asarray(y, dtype=_vendor_host2_model__np.float64)
            if state.shape != (2 * self.nr,):
                raise ValueError(f'Expected state shape {(2 * self.nr,)}, got {state.shape}')
            return (state[:self.nr], state[self.nr:])

        def _face_divergence(self, field: FloatArray, coefficient: FloatArray, outer_r_coefficient_gradient: float) -> FloatArray:
            face_coefficient = _vendor_host2_model__arithmetic_mean(coefficient[:-1], coefficient[1:])
            face_terms = self.grid.face_r_m * face_coefficient * _vendor_host2_model__np.diff(field) / self.grid.node_spacing_m
            divergence_integrals = _vendor_host2_model__np.empty_like(field)
            divergence_integrals[0] = face_terms[0]
            divergence_integrals[1:-1] = face_terms[1:] - face_terms[:-1]
            divergence_integrals[-1] = outer_r_coefficient_gradient - face_terms[-1]
            return divergence_integrals / self.grid.volume_weights_m2

        def rhs(self, t_s: float, y: FloatArray, *, right_of_breakpoint: bool=False) -> FloatArray:
            T_K, C = self.unpack(y)
            rho, cp, conductivity, diffusivity = self.properties.evaluate(C, T_K)
            ambient = self.ambient.values(t_s, right_of_breakpoint=right_of_breakpoint)
            radius_m = self.grid.radius_m
            moisture_outer = -radius_m * self.h_m_m_s * (C[-1] - float(ambient.moisture_eq))
            thermal_outer = -radius_m * self.h_W_m2_K * (T_K[-1] - float(ambient.temperature_K))
            dC_dt = self._face_divergence(C, diffusivity, moisture_outer)
            thermal_divergence = self._face_divergence(T_K, conductivity, thermal_outer)
            dT_dt = thermal_divergence / (rho * cp)
            result = _vendor_host2_model__np.concatenate([dT_dt, dC_dt])
            if _vendor_host2_model__np.any(~_vendor_host2_model__np.isfinite(result)):
                raise FloatingPointError('RHS contains non-finite values')
            return result

        def jacobian_sparsity(self) -> csr_matrix:
            size = 2 * self.nr
            pattern = _vendor_host2_model__lil_matrix((size, size), dtype=_vendor_host2_model__np.int8)
            for output_block in (0, 1):
                for i in range(self.nr):
                    row = output_block * self.nr + i
                    for input_block in (0, 1):
                        for j in range(max(0, i - 1), min(self.nr, i + 2)):
                            pattern[row, input_block * self.nr + j] = 1
            return pattern.tocsr()

        def audit(self, t_s: float, y: FloatArray, *, right_of_breakpoint: bool=False) -> dict[str, float]:
            T_K, C = self.unpack(y)
            rho, cp, conductivity, diffusivity = self.properties.evaluate(C, T_K)
            ambient = self.ambient.values(t_s, right_of_breakpoint=right_of_breakpoint)
            derivative = self.rhs(t_s, y, right_of_breakpoint=right_of_breakpoint)
            dT_dt, dC_dt = self.unpack(derivative)
            radius_m = self.grid.radius_m
            moisture_boundary = radius_m * self.h_m_m_s * (C[-1] - float(ambient.moisture_eq))
            thermal_boundary = radius_m * self.h_W_m2_K * (T_K[-1] - float(ambient.temperature_K))
            moisture_storage = float(_vendor_host2_model__np.dot(self.grid.volume_weights_m2, dC_dt))
            thermal_storage = float(_vendor_host2_model__np.dot(self.grid.volume_weights_m2, rho * cp * dT_dt))
            moisture_residual = moisture_storage + moisture_boundary
            thermal_residual = thermal_storage + thermal_boundary
            moisture_scale = max(abs(moisture_storage), abs(moisture_boundary), _vendor_host2_model__np.finfo(float).tiny)
            thermal_scale = max(abs(thermal_storage), abs(thermal_boundary), _vendor_host2_model__np.finfo(float).tiny)
            moisture_gradient = float(_vendor_host2_model__np.gradient(C, self.grid.r_m, edge_order=2)[-1])
            thermal_gradient = float(_vendor_host2_model__np.gradient(T_K, self.grid.r_m, edge_order=2)[-1])
            moisture_robin = -diffusivity[-1] * moisture_gradient - self.h_m_m_s * (C[-1] - float(ambient.moisture_eq))
            thermal_robin = -conductivity[-1] * thermal_gradient - self.h_W_m2_K * (T_K[-1] - float(ambient.temperature_K))
            moisture_robin_scale = max(abs(diffusivity[-1] * moisture_gradient), abs(self.h_m_m_s * (C[-1] - float(ambient.moisture_eq))), _vendor_host2_model__np.finfo(float).tiny)
            thermal_robin_scale = max(abs(conductivity[-1] * thermal_gradient), abs(self.h_W_m2_K * (T_K[-1] - float(ambient.temperature_K))), _vendor_host2_model__np.finfo(float).tiny)
            return {'moisture_balance_abs': float(abs(moisture_residual)), 'moisture_balance_relative': float(abs(moisture_residual) / moisture_scale), 'thermal_balance_abs': float(abs(thermal_residual)), 'thermal_balance_relative': float(abs(thermal_residual) / thermal_scale), 'moisture_robin_strong_relative': float(abs(moisture_robin) / moisture_robin_scale), 'thermal_robin_strong_relative': float(abs(thermal_robin) / thermal_robin_scale), 'minimum_C': float(_vendor_host2_model__np.min(C)), 'maximum_C': float(_vendor_host2_model__np.max(C)), 'maximum_C_radius_m': float(self.grid.r_m[int(_vendor_host2_model__np.argmax(C))]), 'minimum_T_K': float(_vendor_host2_model__np.min(T_K)), 'maximum_T_K': float(_vendor_host2_model__np.max(T_K)), 'minimum_D_m2_s': float(_vendor_host2_model__np.min(diffusivity)), 'maximum_D_m2_s': float(_vendor_host2_model__np.max(diffusivity))}

    @_vendor_host2_model__dataclass(frozen=True)
    class _vendor_host2_model__IntegrationSettings:
        method: str = 'BDF'
        rtol: float = 1e-08
        atol_temperature_K: float = 1e-07
        atol_moisture_C: float = 1e-10
        max_step_s: float = 600.0
        limit_initial_step: bool = True

        def atol_vector(self, nr: int) -> FloatArray:
            return _vendor_host2_model__np.concatenate([_vendor_host2_model__np.full(nr, self.atol_temperature_K, dtype=_vendor_host2_model__np.float64), _vendor_host2_model__np.full(nr, self.atol_moisture_C, dtype=_vendor_host2_model__np.float64)])

    @_vendor_host2_model__dataclass
    class _vendor_host2_model__SegmentSolution:
        t_start_s: float
        t_end_s: float
        right_of_breakpoint: bool
        first_step_s: float | None
        result: object

    @_vendor_host2_model__dataclass
    class _vendor_host2_model__PiecewiseTrajectory:
        model: FixedRadialModel
        segments: list[SegmentSolution]
        event_time_s: float | None = None
        event_step_bracket_s: tuple[float, float] | None = None

        @property
        def final_time_s(self) -> float:
            return float(self.segments[-1].t_end_s)

        @property
        def final_state(self) -> FloatArray:
            return _vendor_host2_model__np.asarray(self.segments[-1].result.y[:, -1], dtype=_vendor_host2_model__np.float64)

        @property
        def accepted_steps(self) -> int:
            return int(sum((len(segment.result.t) - 1 for segment in self.segments)))

        @property
        def function_evaluations(self) -> int:
            return int(sum((segment.result.nfev for segment in self.segments)))

        def evaluate_state(self, times_s: ArrayLike) -> FloatArray:
            query = _vendor_host2_model__np.asarray(times_s, dtype=_vendor_host2_model__np.float64)
            scalar = query.ndim == 0
            flat = _vendor_host2_model__np.atleast_1d(query)
            if _vendor_host2_model__np.any(flat < self.segments[0].t_start_s - 1e-09) or _vendor_host2_model__np.any(flat > self.final_time_s + 1e-07):
                raise ValueError('Requested time lies outside trajectory')
            output = _vendor_host2_model__np.empty((2 * self.model.nr, flat.size), dtype=_vendor_host2_model__np.float64)
            assigned = _vendor_host2_model__np.zeros(flat.size, dtype=bool)
            for index, segment in enumerate(self.segments):
                if index == 0:
                    mask = (flat >= segment.t_start_s - 1e-09) & (flat <= segment.t_end_s + 1e-09)
                else:
                    mask = (flat > segment.t_start_s + 1e-09) & (flat <= segment.t_end_s + 1e-09)
                if _vendor_host2_model__np.any(mask):
                    output[:, mask] = segment.result.sol(flat[mask])
                    assigned[mask] = True
            if not _vendor_host2_model__np.all(assigned):
                missing = flat[~assigned]
                raise RuntimeError(f'Could not evaluate times: {missing[:5]}')
            if scalar:
                return output[:, 0]
            return output

        def evaluate_fields(self, times_s: ArrayLike) -> tuple[FloatArray, FloatArray]:
            states = self.evaluate_state(times_s)
            if states.ndim == 1:
                return (states[:self.model.nr], states[self.model.nr:])
            return (states[:self.model.nr, :].T, states[self.model.nr:, :].T)

    def _vendor_host2_model___event_function(model: _vendor_host2_model__FixedRadialModel) -> _vendor_host2_model__Callable[[float, _vendor_host2_model__FloatArray], float]:

        def event(_t_s: float, y: FloatArray) -> float:
            _T_K, C = model.unpack(y)
            return float(_vendor_host2_model__np.max(C) - 0.15)
        event.direction = -1.0
        event.terminal = True
        return event

    def _vendor_host2_model___solve_segment(model: _vendor_host2_model__FixedRadialModel, t_span: tuple[float, float], y0: _vendor_host2_model__FloatArray, settings: _vendor_host2_model__IntegrationSettings, *, right_of_breakpoint: bool, detect_event: bool) -> _vendor_host2_model__SegmentSolution:
        rhs = lambda t, y: model.rhs(t, y, right_of_breakpoint=right_of_breakpoint)
        events = _vendor_host2_model___event_function(model) if detect_event else None
        first_step_s: float | None = None
        if settings.limit_initial_step and t_span[0] == 0.0:
            T0, C0 = model.unpack(y0)
            _rho, _cp, _k, D0 = model.properties.evaluate(C0, T0)
            maximum_diffusivity = float(_vendor_host2_model__np.max(D0))
            if maximum_diffusivity > 0.0:
                first_step_s = min(0.01, 0.1 * model.grid.minimum_spacing_m ** 2 / maximum_diffusivity, t_span[1] - t_span[0])
        solve_options = {}
        if first_step_s is not None:
            solve_options['first_step'] = first_step_s
        result = _vendor_host2_model__solve_ivp(rhs, t_span, y0, method=settings.method, rtol=settings.rtol, atol=settings.atol_vector(model.nr), max_step=settings.max_step_s, jac_sparsity=model.jacobian_sparsity(), dense_output=True, events=events, **solve_options)
        if not result.success:
            raise RuntimeError(f'Integration failed on {t_span}: {result.message}')
        return _vendor_host2_model__SegmentSolution(t_start_s=float(t_span[0]), t_end_s=float(result.t[-1]), right_of_breakpoint=right_of_breakpoint, first_step_s=first_step_s, result=result)

    def _vendor_host2_model__solve_to_event(model: _vendor_host2_model__FixedRadialModel, settings: _vendor_host2_model__IntegrationSettings, *, horizon_s: float=2592000.0, initial_temperature_K: float=301.15, initial_moisture_C: float=2.55) -> _vendor_host2_model__PiecewiseTrajectory:
        if horizon_s <= 0.0:
            raise ValueError('Event horizon must be positive')
        y = model.initial_state(initial_temperature_K, initial_moisture_C)
        segments: list[SegmentSolution] = []
        points = model.ambient.breakpoints(0.0, horizon_s)
        spans = [(float(points[i]), float(points[i + 1]), bool(points[i] >= model.ambient.breakpoint_s and _vendor_host2_model__np.isfinite(model.ambient.breakpoint_s))) for i in range(points.size - 1)]
        for start_s, end_s, right_side in spans:
            segment = _vendor_host2_model___solve_segment(model, (start_s, end_s), y, settings, right_of_breakpoint=right_side, detect_event=True)
            segments.append(segment)
            event_times = segment.result.t_events[0]
            if event_times.size:
                root = float(event_times[0])
                times = _vendor_host2_model__np.asarray(segment.result.t, dtype=_vendor_host2_model__np.float64)
                before = times[times < root - 1e-09]
                left = float(before[-1]) if before.size else float(start_s)
                return _vendor_host2_model__PiecewiseTrajectory(model=model, segments=segments, event_time_s=root, event_step_bracket_s=(left, root))
            y = _vendor_host2_model__np.asarray(segment.result.y[:, -1], dtype=_vendor_host2_model__np.float64)
        trajectory = _vendor_host2_model__PiecewiseTrajectory(model=model, segments=segments)
        maximum = float(_vendor_host2_model__np.max(model.unpack(trajectory.final_state)[1]))
        raise RuntimeError(f'Drying event not reached by {horizon_s:.6g} s; final max(C)={maximum:.12g}')

    def _vendor_host2_model__solve_to_time(model: _vendor_host2_model__FixedRadialModel, settings: _vendor_host2_model__IntegrationSettings, end_time_s: float, *, initial_temperature_K: float=301.15, initial_moisture_C: float=2.55) -> _vendor_host2_model__PiecewiseTrajectory:
        if end_time_s <= 0.0:
            raise ValueError('End time must be positive')
        y = model.initial_state(initial_temperature_K, initial_moisture_C)
        segments: list[SegmentSolution] = []
        points = model.ambient.breakpoints(0.0, end_time_s)
        spans = [(float(points[i]), float(points[i + 1]), bool(points[i] >= model.ambient.breakpoint_s and _vendor_host2_model__np.isfinite(model.ambient.breakpoint_s))) for i in range(points.size - 1)]
        for start_s, stop_s, right_side in spans:
            segment = _vendor_host2_model___solve_segment(model, (start_s, stop_s), y, settings, right_of_breakpoint=right_side, detect_event=False)
            segments.append(segment)
            y = _vendor_host2_model__np.asarray(segment.result.y[:, -1], dtype=_vendor_host2_model__np.float64)
        return _vendor_host2_model__PiecewiseTrajectory(model=model, segments=segments)

    def _vendor_host2_model__sample_radii(fields: _vendor_host2_model__FloatArray, source_r_m: _vendor_host2_model__FloatArray, target_r_m: _vendor_host2_model__ArrayLike) -> _vendor_host2_model__FloatArray:
        values = _vendor_host2_model__np.asarray(fields, dtype=_vendor_host2_model__np.float64)
        targets = _vendor_host2_model__np.asarray(target_r_m, dtype=_vendor_host2_model__np.float64)
        if values.ndim == 1:
            return _vendor_host2_model__np.interp(targets, source_r_m, values)
        return _vendor_host2_model__np.vstack([_vendor_host2_model__np.interp(targets, source_r_m, row) for row in values])

    def _vendor_host2_model__refine_event_bracket(trajectory: _vendor_host2_model__PiecewiseTrajectory, guess_s: float, *, half_width_s: float=2.0, tolerance_s: float=1e-07) -> dict[str, float]:
        lower = max(0.0, guess_s - half_width_s)
        upper = min(trajectory.final_time_s, guess_s + half_width_s)

        def g(time_s: float) -> float:
            _T, C = trajectory.evaluate_fields(time_s)
            return float(_vendor_host2_model__np.max(C) - 0.15)
        g_lower = g(lower)
        g_upper = g(upper)
        expansion = half_width_s
        while g_lower < 0.0 and lower > 0.0:
            expansion *= 2.0
            lower = max(0.0, guess_s - expansion)
            g_lower = g(lower)
        while g_upper >= 0.0 and upper < trajectory.final_time_s:
            expansion *= 2.0
            upper = min(trajectory.final_time_s, guess_s + expansion)
            g_upper = g(upper)
        if g_lower < 0.0 or g_upper >= 0.0:
            raise RuntimeError(f'Could not bracket event: g(lower)={g_lower}, g(upper)={g_upper}')
        while upper - lower > tolerance_s:
            middle = 0.5 * (lower + upper)
            g_middle = g(middle)
            if g_middle >= 0.0:
                lower, g_lower = (middle, g_middle)
            else:
                upper, g_upper = (middle, g_middle)
        return {'lower_s': float(lower), 'upper_s': float(upper), 'width_s': float(upper - lower), 'g_lower': float(g_lower), 'g_upper': float(g_upper)}

    def _vendor_host2_model__requested_time_grid(start_s: float, step_s: float, end_s: float) -> _vendor_host2_model__FloatArray:
        if start_s <= 0.0 or step_s <= 0.0 or end_s < start_s:
            raise ValueError('Invalid requested output time grid')
        last_regular = _vendor_host2_model__np.floor((end_s - start_s) / step_s + 1e-12)
        count = int(last_regular) + 1
        times = start_s + step_s * _vendor_host2_model__np.arange(count, dtype=_vendor_host2_model__np.float64)
        if end_s - times[-1] > 1e-08:
            times = _vendor_host2_model__np.concatenate([times, _vendor_host2_model__np.asarray([end_s], dtype=_vendor_host2_model__np.float64)])
        elif abs(end_s - times[-1]) <= 1e-08:
            times[-1] = end_s
        return times

    def _vendor_host2_model__selected_times_with_final(step_s: float, final_s: float) -> _vendor_host2_model__FloatArray:
        regular = _vendor_host2_model__np.arange(step_s, final_s, step_s, dtype=_vendor_host2_model__np.float64)
        if regular.size and abs(regular[-1] - final_s) <= 1e-08:
            regular[-1] = final_s
            return regular
        return _vendor_host2_model__np.concatenate([regular, _vendor_host2_model__np.asarray([final_s], dtype=_vendor_host2_model__np.float64)])

    # ===== solvers/moving_model.py: readable implementation =====
    _moving_model____doc__ = 'P4 material-coordinate adapter of the received host2 fixed-domain toolkit.\n\nState order is [T_K, C]. The grid stores initial/reference physical coordinate\na in [0,R0]. Every divergence is scaled by (R0/R(t))**2; the outer reference\nflux is -R(t)*h*(field-ambient). This implements the documented H3-01 closure,\nnot an inference that conservation uniquely determines that physical closure.\nNo I/O occurs at import, and no received source files are modified.'
    from dataclasses import dataclass as _moving_model__dataclass, field as _moving_model__field
    from pathlib import Path as _moving_model__Path
    from typing import Callable as _moving_model__Callable
    import json as _moving_model__json
    import numpy as _moving_model__np
    from openpyxl import load_workbook as _moving_model__load_workbook
    from scipy.interpolate import PchipInterpolator as _moving_model__PchipInterpolator
    try:
        _moving_model__upstream = _sections['_vendor_host2_model']
    except ImportError:
        _moving_model__upstream = _sections['_vendor_host2_model']
    _moving_model__AmbientProfile = _moving_model__upstream.AmbientProfile
    _moving_model__AmbientValues = _moving_model__upstream.AmbientValues
    _moving_model__ConstantAmbient = _moving_model__upstream.ConstantAmbient
    _moving_model__MaterialProperties = _moving_model__upstream.MaterialProperties
    _moving_model__RadialGrid = _moving_model__upstream.RadialGrid
    _moving_model__IntegrationSettings = _moving_model__upstream.IntegrationSettings
    _moving_model__PiecewiseTrajectory = _moving_model__upstream.PiecewiseTrajectory

    @_moving_model__dataclass(frozen=True)
    class _moving_model__RadiusProfile:
        t_s: np.ndarray
        radius_m: np.ndarray
        tail_radius_m: float

        @classmethod
        def from_xlsx(cls, path, tail_radius_m=0.01198):
            book = _moving_model__load_workbook(_moving_model__Path(path), read_only=True, data_only=True)
            try:
                values = _moving_model__np.asarray(list(book['Sheet1'].iter_rows(min_row=2, max_col=2, values_only=True)), dtype=float)
            finally:
                book.close()
            t, radius_cm = values.T
            if not _moving_model__np.isfinite(values).all() or t[0] != 0 or _moving_model__np.any(_moving_model__np.diff(t) <= 0):
                raise ValueError('Invalid radius observation times or values')
            radius = radius_cm * 0.01
            if _moving_model__np.any(radius <= 0) or _moving_model__np.any(_moving_model__np.diff(radius) > 0):
                raise ValueError('Measured radii must be positive and nonincreasing')
            return cls(t, radius, float(tail_radius_m))

        def radius(self, t_s):
            t = _moving_model__np.asarray(t_s, dtype=float)
            if _moving_model__np.any(~_moving_model__np.isfinite(t)) or _moving_model__np.any(t < 0):
                raise ValueError('Radius time must be finite and nonnegative')
            value = _moving_model__np.where(t > self.t_s[-1], self.tail_radius_m, _moving_model__np.interp(t, self.t_s, self.radius_m))
            return float(value) if t.ndim == 0 else value

        def breakpoints(self, start_s, end_s):
            return _moving_model__np.r_[start_s, self.t_s[(self.t_s > start_s) & (self.t_s < end_s)], end_s]

    @_moving_model__dataclass(frozen=True)
    class _moving_model__FixedRadius:
        radius_m: float = 0.02

        def radius(self, t_s):
            t = _moving_model__np.asarray(t_s, dtype=float)
            if _moving_model__np.any(~_moving_model__np.isfinite(t)) or _moving_model__np.any(t < 0):
                raise ValueError('Radius time must be finite and nonnegative')
            if not _moving_model__np.isfinite(self.radius_m) or self.radius_m <= 0:
                raise ValueError('Fixed radius must be finite and positive')
            return self.radius_m if t.ndim == 0 else _moving_model__np.full(t.shape, self.radius_m)

        def breakpoints(self, start_s, end_s):
            return _moving_model__np.array([start_s, end_s], dtype=float)

    @_moving_model__dataclass(frozen=True)
    class _moving_model__CallableRadius:
        function: Callable
        knots_s: tuple = ()

        def radius(self, t_s):
            t = _moving_model__np.asarray(t_s, dtype=float)
            if _moving_model__np.any(~_moving_model__np.isfinite(t)) or _moving_model__np.any(t < 0):
                raise ValueError('Radius time must be finite and nonnegative')
            result = _moving_model__np.asarray(self.function(t), dtype=float)
            result = _moving_model__np.broadcast_to(result, t.shape)
            if _moving_model__np.any(~_moving_model__np.isfinite(result)) or _moving_model__np.any(result <= 0):
                raise ValueError('Radius must be finite and positive')
            return float(result) if t.ndim == 0 else result

        def breakpoints(self, start_s, end_s):
            knots = _moving_model__np.asarray(self.knots_s, dtype=float)
            return _moving_model__np.r_[start_s, knots[(knots > start_s) & (knots < end_s)], end_s]

    @_moving_model__dataclass
    class _moving_model__GeometryAwareAmbient:
        base: object
        geometry: object

        @property
        def breakpoint_s(self):
            return self.base.breakpoint_s

        def values(self, t_s, *, right_of_breakpoint=False):
            return self.base.values(t_s, right_of_breakpoint=right_of_breakpoint)

        def breakpoints(self, start_s, end_s):
            return _moving_model__np.unique(_moving_model__np.r_[self.base.breakpoints(start_s, end_s), self.geometry.breakpoints(start_s, end_s)])

    def _moving_model__current_dry_density_boundary(_t_s, _R_m, _R0_m):
        return 1.0

    def _moving_model__initial_dry_density_boundary(_t_s, R_m, R0_m):
        """Alternative external flux normalization only; internal flux unchanged."""
        return (R_m / R0_m) ** 2

    @_moving_model__dataclass
    class _moving_model__MovingRadialModel(_moving_model__upstream.FixedRadialModel):
        geometry: object = _moving_model__field(default_factory=_moving_model__FixedRadius)
        moisture_boundary_multiplier: Callable = _moving_model__current_dry_density_boundary

        def __post_init__(self):
            if isinstance(self.ambient, _moving_model__GeometryAwareAmbient):
                self.ambient = _moving_model__GeometryAwareAmbient(self.ambient.base, self.geometry)
            else:
                self.ambient = _moving_model__GeometryAwareAmbient(self.ambient, self.geometry)

        @property
        def xi(self):
            return self.grid.r_m / self.grid.radius_m

        def rhs(self, t_s, y, *, right_of_breakpoint=False):
            T_K, C = self.unpack(y)
            if _moving_model__np.any(T_K <= 0):
                raise FloatingPointError('Absolute temperature must be positive')
            rho, cp, conductivity, diffusivity = self.properties.evaluate(C, T_K)
            ambient = self.ambient.values(t_s, right_of_breakpoint=right_of_breakpoint)
            R = float(self.geometry.radius(t_s))
            R0 = self.grid.radius_m
            scale = (R0 / R) ** 2
            multiplier = float(self.moisture_boundary_multiplier(t_s, R, R0))
            if not _moving_model__np.isfinite(multiplier) or multiplier < 0:
                raise ValueError('Boundary multiplier must be finite and nonnegative')
            C_outer = -R * self.h_m_m_s * multiplier * (C[-1] - float(ambient.moisture_eq))
            T_outer = -R * self.h_W_m2_K * (T_K[-1] - float(ambient.temperature_K))
            dC = scale * self._face_divergence(C, diffusivity, C_outer)
            dT = scale * self._face_divergence(T_K, conductivity, T_outer) / (rho * cp)
            result = _moving_model__np.r_[dT, dC]
            if not _moving_model__np.isfinite(result).all():
                raise FloatingPointError('Moving-domain RHS is nonfinite')
            return result

        def audit(self, t_s, y, *, right_of_breakpoint=False):
            T, C = self.unpack(y)
            rho, cp, _, _ = self.properties.evaluate(C, T)
            dT, dC = self.unpack(self.rhs(t_s, y, right_of_breakpoint=right_of_breakpoint))
            ambient = self.ambient.values(t_s, right_of_breakpoint=right_of_breakpoint)
            R = float(self.geometry.radius(t_s))
            R0 = self.grid.radius_m
            weights = self.grid.volume_weights_m2 / R0 ** 2
            multiplier = float(self.moisture_boundary_multiplier(t_s, R, R0))
            c_store = float(2 * weights @ dC)
            c_flux = 2 * self.h_m_m_s * multiplier / R * (C[-1] - float(ambient.moisture_eq))
            t_store = float(2 * weights @ (rho * cp * dT))
            t_flux = 2 * self.h_W_m2_K / R * (T[-1] - float(ambient.temperature_K))
            return {'moisture_mean_rate_residual': float(c_store + c_flux), 'thermal_current_volume_rate_residual': float(t_store + t_flux), 'moisture_scaled_residual': float(abs(c_store + c_flux) / max(abs(c_store), abs(c_flux), 1e-300)), 'thermal_scaled_residual': float(abs(t_store + t_flux) / max(abs(t_store), abs(t_flux), 1e-300)), 'mean_C': float(2 * weights @ C), 'minimum_C': float(C.min()), 'maximum_C': float(C.max()), 'maximum_C_xi': float(self.xi[_moving_model__np.argmax(C)]), 'radius_m': R, 'boundary_multiplier': multiplier}

    def _moving_model__properties_from_contract(contract, property_set='P4'):
        return _moving_model__MaterialProperties(**contract['property_sets'][property_set], name=property_set)

    def _moving_model__build_model(input_dir, contract_path, N=640, power=2.0, properties='P4', geometry='measured', ambient=None, h_W_m2_K=None, h_m_m_s=None, moisture_boundary_multiplier=_moving_model__current_dry_density_boundary):
        """Build measured or fixed geometry using the original data and contract.

        geometry may be "measured", "fixed", or an object exposing radius(t) and
        breakpoints(start,end). properties may be a contract group or a
        MaterialProperties instance. ambient may be any host2-compatible object.
        """
        contract = _moving_model__json.loads(_moving_model__Path(contract_path).read_text(encoding='utf-8'))
        input_dir = _moving_model__Path(input_dir)
        attachments = input_dir if (input_dir / '附件1.xlsx').exists() else input_dir / '附件'
        R0 = contract['initial_state']['radius_m']
        if isinstance(geometry, str):
            if geometry == 'measured':
                geometry = _moving_model__RadiusProfile.from_xlsx(attachments / '附件2.xlsx', contract['input_radius']['post_observation']['radius_m'])
            elif geometry == 'fixed':
                geometry = _moving_model__FixedRadius(R0)
            else:
                raise ValueError('Unknown geometry')
        if ambient is None:
            ambient = _moving_model__AmbientProfile.from_xlsx(attachments / '附件1.xlsx')
        material = _moving_model__properties_from_contract(contract, properties) if isinstance(properties, str) else properties
        grid = _moving_model__RadialGrid.uniform(N, R0) if power == 1 else _moving_model__RadialGrid.surface_stretched(N, R0, power=power)
        return _moving_model__MovingRadialModel(grid, ambient, material, contract['fixed_domain_model']['h_W_m2_K'] if h_W_m2_K is None else h_W_m2_K, contract['fixed_domain_model']['h_m_m_s'] if h_m_m_s is None else h_m_m_s, geometry, moisture_boundary_multiplier)

    def _moving_model__solve_moving_to_event(model, settings=None, horizon_s=432000.0, post_event_s=10.0, initial_temperature_K=301.15, initial_moisture_C=2.55):
        if settings is None:
            settings = _moving_model__IntegrationSettings(rtol=1e-09, atol_temperature_K=1e-08, atol_moisture_C=1e-11)
        trajectory = _moving_model__upstream.solve_to_event(model, settings, horizon_s=horizon_s, initial_temperature_K=initial_temperature_K, initial_moisture_C=initial_moisture_C)
        if post_event_s < 0:
            raise ValueError('post_event_s must be nonnegative')
        if post_event_s:
            end = trajectory.event_time_s + post_event_s
            state = trajectory.final_state
            points = model.ambient.breakpoints(trajectory.final_time_s, end)
            for start, stop in zip(points[:-1], points[1:]):
                segment = _moving_model__upstream._solve_segment(model, (float(start), float(stop)), state, settings, right_of_breakpoint=bool(start >= model.ambient.breakpoint_s and _moving_model__np.isfinite(model.ambient.breakpoint_s)), detect_event=False)
                trajectory.segments.append(segment)
                state = _moving_model__np.asarray(segment.result.y[:, -1], dtype=float)
        return trajectory

    def _moving_model__solve_moving_to_time(model, settings, end_time_s, **initial):
        return _moving_model__upstream.solve_to_time(model, settings, end_time_s, **initial)

    def _moving_model__sample_physical(trajectory, times_s, radii_m, method='linear'):
        if method not in ('linear', 'pchip'):
            raise ValueError('Sampling method must be linear or pchip')
        times = _moving_model__np.atleast_1d(_moving_model__np.asarray(times_s, dtype=float))
        radii = _moving_model__np.atleast_1d(_moving_model__np.asarray(radii_m, dtype=float))
        if _moving_model__np.any(radii < 0) or _moving_model__np.any(~_moving_model__np.isfinite(radii)):
            raise ValueError('Sampling radii must be finite and nonnegative')
        T, C = trajectory.evaluate_fields(times)
        R = _moving_model__np.asarray(trajectory.model.geometry.radius(times), dtype=float)
        valid = radii[None, :] < R[:, None]
        sampled_T = _moving_model__np.full(valid.shape, _moving_model__np.nan)
        sampled_C = _moving_model__np.full(valid.shape, _moving_model__np.nan)
        xi = trajectory.model.xi
        for index, radius in enumerate(R):
            targets = radii[valid[index]] / radius
            if method == 'linear':
                sampled_T[index, valid[index]] = _moving_model__np.interp(targets, xi, T[index])
                sampled_C[index, valid[index]] = _moving_model__np.interp(targets, xi, C[index])
            else:
                sampled_T[index, valid[index]] = _moving_model__PchipInterpolator(xi, T[index], extrapolate=False)(targets)
                sampled_C[index, valid[index]] = _moving_model__PchipInterpolator(xi, C[index], extrapolate=False)(targets)
        return {'t_s': times, 'r_sample_m': radii, 'R_m': R, 'valid_interior': valid, 'T_K_sample': sampled_T, 'C_sample': sampled_C, 'T_K_surface': T[:, -1], 'C_surface': C[:, -1]}

    # ===== solvers/radial_core.py: readable implementation =====
    _radial_core____doc__ = 'Shared fixed-radius nodal finite-volume operators; SI units throughout.\n\nNode values approximate point values. Half-distance control volumes supply\npositive lumped weights, including the centre and the true surface nodes.\nThis module does not implement a moving-domain conservation law.'
    from dataclasses import dataclass as _radial_core__dataclass
    from pathlib import Path as _radial_core__Path
    import hashlib as _radial_core__hashlib
    import json as _radial_core__json
    import numpy as _radial_core__np
    from scipy.sparse import diags as _radial_core__diags
    import openpyxl as _radial_core__openpyxl

    @_radial_core__dataclass(frozen=True)
    class _radial_core__NodeRadialGrid:
        radius_m: float
        intervals: int

        def __post_init__(self):
            if not _radial_core__np.isfinite(self.radius_m) or self.radius_m <= 0 or self.intervals < 2 or (int(self.intervals) != self.intervals):
                raise ValueError('Positive radius and integer intervals >= 2 required')
            r = _radial_core__np.linspace(0.0, self.radius_m, self.intervals + 1)
            faces = _radial_core__np.concatenate(([0.0], (r[:-1] + r[1:]) / 2, [self.radius_m]))
            object.__setattr__(self, 'r_m', r)
            object.__setattr__(self, 'faces_m', faces)
            object.__setattr__(self, 'weights_m2', _radial_core__np.diff(faces ** 2) / 2)

    def _radial_core__face_arithmetic(values):
        a = _radial_core__np.asarray(values, dtype=float)
        return 0.5 * (a[:-1] + a[1:])

    def _radial_core__diffusion_rhs(grid, u, face_coefficient, surface_flux, capacity=1.0, source=None):
        """capacity*u_t = div(a grad(u)); source is added to u_t.

        surface_flux is positive OUTWARD, in the same flux units as -a*u_r.
        Omitting the common factor 2*pi*length does not change this radial balance.
        """
        u = _radial_core__np.asarray(u)
        internal_flow = -grid.faces_m[1:-1] * face_coefficient * _radial_core__np.diff(u) / _radial_core__np.diff(grid.r_m)
        face_flow = _radial_core__np.concatenate(([0.0], internal_flow, [grid.radius_m * surface_flux]))
        result = -_radial_core__np.diff(face_flow) / (grid.weights_m2 * capacity)
        if source is not None:
            result = result + source
        return result

    def _radial_core__linear_diffusion_matrix(grid, coefficient, surface_exchange, capacity=1.0):
        """Sparse matrix for a constant coefficient and q_s=beta*(u_s-g)."""
        conductance = grid.faces_m[1:-1] * coefficient / _radial_core__np.diff(grid.r_m)
        denominator = grid.weights_m2 * capacity
        diagonal = -(_radial_core__np.r_[0.0, conductance] + _radial_core__np.r_[conductance, grid.radius_m * surface_exchange]) / denominator
        lower = conductance / denominator[1:]
        upper = conductance / denominator[:-1]
        return _radial_core__diags([lower, diagonal, upper], [-1, 0, 1], format='csc')

    def _radial_core__tridiagonal_sparsity(size):
        return _radial_core__diags([_radial_core__np.ones(size - 1), _radial_core__np.ones(size), _radial_core__np.ones(size - 1)], [-1, 0, 1], format='csc')

    def _radial_core__load_contract(path):
        return _radial_core__json.loads(_radial_core__Path(path).read_text(encoding='utf-8'))

    def _radial_core__sha256(path):
        return _radial_core__hashlib.sha256(_radial_core__Path(path).read_bytes()).hexdigest()

    def _radial_core__property_values(contract, property_set, C, T_K):
        """Empirical properties; rho is the T04 thermal coefficient convention."""
        p = contract['property_sets'][property_set]
        C, T_K = _radial_core__np.broadcast_arrays(_radial_core__np.asarray(C, dtype=float), _radial_core__np.asarray(T_K, dtype=float))
        if _radial_core__np.any(~_radial_core__np.isfinite(C)) or _radial_core__np.any(C < 0) or _radial_core__np.any(~_radial_core__np.isfinite(T_K)) or _radial_core__np.any(T_K <= 0):
            raise ValueError('Nonfinite/negative moisture or nonpositive Kelvin temperature')
        fraction = C / (1.0 + C)
        rho = p['rho_offset'] + p['rho_C_coefficient'] * C
        cp = p['cp_offset'] + p['cp_fraction_coefficient'] * fraction
        k = p['k_offset'] + p['k_fraction_coefficient'] * fraction
        exponent = _radial_core__np.full(C.shape, -_radial_core__np.inf)
        _radial_core__np.divide(-p['D_moisture_exponent'], C, out=exponent, where=C > 0)
        D = p['D_prefactor'] * _radial_core__np.exp(exponent - p['D_temperature_exponent_K'] / T_K)
        return {'rho': rho, 'cp': cp, 'k': k, 'D': D}

    def _radial_core___read_numeric_table(path, expected_columns):
        book = _radial_core__openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = list(book['Sheet1'].iter_rows(min_row=2, values_only=True))
            data = _radial_core__np.asarray([row[:expected_columns] for row in rows], dtype=float)
        finally:
            book.close()
        if data.ndim != 2 or data.shape[1] != expected_columns or (not _radial_core__np.all(_radial_core__np.isfinite(data))):
            raise ValueError(f'Invalid finite numeric table: {path}')
        if data[0, 0] != 0 or _radial_core__np.any(_radial_core__np.diff(data[:, 0]) <= 0):
            raise ValueError('Time must start at zero and increase strictly')
        return data

    class _radial_core__Environment:

        def __init__(self, input_dir, contract):
            self.path = _radial_core__Path(input_dir) / contract['input_environment']['file']
            table = _radial_core___read_numeric_table(self.path, 3)
            self.t_s, temperature_C, self.C_eq = table.T
            self.T_K = temperature_C + 273.15
            tail = contract['input_environment']['post_observation']
            self.tail_T_K = tail['temperature_degC'] + 273.15
            self.tail_C = tail['ambient_moisture']
            self.source_sha256 = _radial_core__sha256(self.path)
            if (len(table), self.t_s[-1]) != (241, 14400):
                raise ValueError('Environment coverage differs from the audited T04 input')

        def __call__(self, t_s, right_limit=False):
            t = _radial_core__np.asarray(t_s, dtype=float)
            if _radial_core__np.any(t < 0) or not _radial_core__np.all(_radial_core__np.isfinite(t)):
                raise ValueError('Time must be finite and nonnegative')
            outside = t > self.t_s[-1]
            if right_limit:
                outside |= t == self.t_s[-1]
            T = _radial_core__np.where(outside, self.tail_T_K, _radial_core__np.interp(t, self.t_s, self.T_K))
            C = _radial_core__np.where(outside, self.tail_C, _radial_core__np.interp(t, self.t_s, self.C_eq))
            return (T, C)

        def breakpoints(self, start_s, end_s):
            inner = self.t_s[(self.t_s > start_s) & (self.t_s < end_s)]
            return _radial_core__np.r_[start_s, inner, end_s]

    class _radial_core__RadiusHistory:
        """Input geometry only: this is not a moving-domain PDE implementation."""

        def __init__(self, input_dir, contract):
            self.path = _radial_core__Path(input_dir) / contract['input_radius']['file']
            table = _radial_core___read_numeric_table(self.path, 2)
            self.t_s = table[:, 0]
            self.R_m = table[:, 1] * 0.01
            self.tail_R_m = contract['input_radius']['post_observation']['radius_m']
            self.source_sha256 = _radial_core__sha256(self.path)
            if len(table) != 145 or self.t_s[-1] != 259200 or _radial_core__np.any(self.R_m <= 0) or _radial_core__np.any(_radial_core__np.diff(self.R_m) > 0):
                raise ValueError('Radius coverage/monotonicity differs from the audited input')

        def __call__(self, t_s):
            t = _radial_core__np.asarray(t_s, dtype=float)
            if _radial_core__np.any(t < 0) or not _radial_core__np.all(_radial_core__np.isfinite(t)):
                raise ValueError('Time must be finite and nonnegative')
            return _radial_core__np.where(t > self.t_s[-1], self.tail_R_m, _radial_core__np.interp(t, self.t_s, self.R_m))

        def derivative(self, t_s):
            """Right derivative at knots; constant extension after the final knot."""
            t = _radial_core__np.asarray(t_s, dtype=float)
            if _radial_core__np.any(t < 0) or not _radial_core__np.all(_radial_core__np.isfinite(t)):
                raise ValueError('Time must be finite and nonnegative')
            indices = _radial_core__np.clip(_radial_core__np.searchsorted(self.t_s, t, side='right') - 1, 0, len(self.t_s) - 2)
            slope = _radial_core__np.diff(self.R_m) / _radial_core__np.diff(self.t_s)
            return _radial_core__np.where(t >= self.t_s[-1], 0.0, slope[indices])

    def _radial_core__maximum_moisture_event(threshold=0.15, terminal=True):
        """Factory for a C-only state. T07 must adapt composite/moving states."""

        def event(t, C):
            return float(_radial_core__np.max(C) - threshold)
        event.direction = -1
        event.terminal = terminal
        return event

    # ===== solvers/relative_temperature.py: readable implementation =====
    _relative_temperature____doc__ = 'Shifted-temperature representation of the identical finite-volume model.\n\nThe state stores theta=T-Tref with Tref equal to the long-time ambient\ntemperature. This avoids subtracting nearly equal ~323 K values when the\nhighly refined surface cells approach thermal equilibrium. Arrhenius terms\nstill receive absolute kelvins. No physical coefficient or equation changes.'
    from dataclasses import dataclass as _relative_temperature__dataclass
    import numpy as _relative_temperature__np
    _relative_temperature__m = _sections['moving_model']

    @_relative_temperature__dataclass
    class _relative_temperature__RelativeTemperatureModel(_relative_temperature__m.MovingRadialModel):
        temperature_reference_K: float = 323.14591666666666

        @classmethod
        def from_model(cls, model):
            tail = float(model.ambient.values(100000000.0, right_of_breakpoint=True).temperature_K)
            return cls(model.grid, model.ambient, model.properties, model.h_W_m2_K, model.h_m_m_s, model.geometry, model.moisture_boundary_multiplier, tail)

        def initial_state(self, temperature_K=301.15, moisture_C=2.55):
            return _relative_temperature__np.r_[_relative_temperature__np.full(self.nr, temperature_K - self.temperature_reference_K), _relative_temperature__np.full(self.nr, moisture_C)]

        def unpack(self, y):
            y = _relative_temperature__np.asarray(y, dtype=float)
            if y.shape != (2 * self.nr,):
                raise ValueError('Unexpected shifted state shape')
            return (y[:self.nr] + self.temperature_reference_K, y[self.nr:])

        def rhs(self, t_s, y, *, right_of_breakpoint=False):
            T, C = self.unpack(y)
            theta = _relative_temperature__np.asarray(y)[:self.nr]
            rho, cp, k, D = self.properties.evaluate(C, T)
            ambient = self.ambient.values(t_s, right_of_breakpoint=right_of_breakpoint)
            R = float(self.geometry.radius(t_s))
            R0 = self.grid.radius_m
            scale = (R0 / R) ** 2
            mult = float(self.moisture_boundary_multiplier(t_s, R, R0))
            outer_C = -R * self.h_m_m_s * mult * (C[-1] - float(ambient.moisture_eq))
            outer_T = -R * self.h_W_m2_K * (theta[-1] - (float(ambient.temperature_K) - self.temperature_reference_K))
            dC = scale * self._face_divergence(C, D, outer_C)
            dT = scale * self._face_divergence(theta, k, outer_T) / (rho * cp)
            result = _relative_temperature__np.r_[dT, dC]
            if not _relative_temperature__np.isfinite(result).all():
                raise FloatingPointError('Nonfinite shifted-model derivative')
            return result

        def audit(self, t_s, y, *, right_of_breakpoint=False):
            T, C = self.unpack(y)
            derivative = self.rhs(t_s, y, right_of_breakpoint=right_of_breakpoint)
            dT, dC = (derivative[:self.nr], derivative[self.nr:])
            rho, cp, _, _ = self.properties.evaluate(C, T)
            ambient = self.ambient.values(t_s, right_of_breakpoint=right_of_breakpoint)
            R = float(self.geometry.radius(t_s))
            R0 = self.grid.radius_m
            w = 2 * self.grid.volume_weights_m2 / R0 ** 2
            mult = float(self.moisture_boundary_multiplier(t_s, R, R0))
            heat_difference = float(_relative_temperature__np.asarray(y)[self.nr - 1]) - (float(ambient.temperature_K) - self.temperature_reference_K)
            return {'moisture_mean_rate_residual': float(w @ dC + 2 * self.h_m_m_s * mult / R * (C[-1] - float(ambient.moisture_eq))), 'thermal_current_volume_rate_residual': float(w @ (rho * cp * dT) + 2 * self.h_W_m2_K / R * heat_difference), 'mean_C': float(w @ C), 'minimum_C': float(C.min()), 'maximum_C': float(C.max()), 'maximum_C_xi': float(self.xi[_relative_temperature__np.argmax(C)]), 'radius_m': R, 'boundary_multiplier': mult}

    class _relative_temperature__RelativeTrajectory(_relative_temperature__m.PiecewiseTrajectory):

        def evaluate_fields(self, times_s):
            T, C = super().evaluate_fields(times_s)
            return (T + self.model.temperature_reference_K, C)

    def _relative_temperature__wrap(trajectory):
        return _relative_temperature__RelativeTrajectory(trajectory.model, trajectory.segments, trajectory.event_time_s, trajectory.event_step_bracket_s)

    # ===== solvers/temperature_reference.py: readable implementation =====
    _temperature_reference____doc__ = 'Independent P1 cylindrical Robin heat reference with affine forcing.\n\nThe spatial basis is the continuous Bessel eigenbasis, not the FVM operator.\nEvery piecewise-affine environmental-temperature interval is advanced exactly\nin modal time. Finite-mode truncation and floating-point error remain.\n\nUsage:\n    python temperature_reference.py --input-dir PATH --output-dir PATH\n\nGeneral Bessel identities: https://dlmf.nist.gov/10.6 and /10.22.\nThe Robin eigenproblem and modal forcing recurrence are derived in the local\ndocument "问题1独立验证设计.md", sections 3 and 4.'
    import argparse as _temperature_reference__argparse
    from datetime import datetime as _temperature_reference__datetime, timezone as _temperature_reference__timezone
    from functools import lru_cache as _temperature_reference__lru_cache
    import hashlib as _temperature_reference__hashlib
    import json as _temperature_reference__json
    from pathlib import Path as _temperature_reference__Path
    import platform as _temperature_reference__platform
    _temperature_reference__source_identity = source_identity
    import sys as _temperature_reference__sys
    from time import perf_counter as _temperature_reference__perf_counter
    import numpy as _temperature_reference__np
    import openpyxl as _temperature_reference__openpyxl
    import scipy as _temperature_reference__scipy
    from scipy.optimize import brentq as _temperature_reference__brentq
    from scipy.special import j0 as _temperature_reference__j0, j1 as _temperature_reference__j1, jn_zeros as _temperature_reference__jn_zeros
    _temperature_reference__RADIUS_M = 0.02
    _temperature_reference__RHO_KG_M3 = 820.0
    _temperature_reference__CP_J_KG_K = 2600.0
    _temperature_reference__K_W_M_K = 0.36
    _temperature_reference__H_W_M2_K = 25.0
    _temperature_reference__INITIAL_T_K = 301.15
    _temperature_reference__ALPHA_M2_S = _temperature_reference__K_W_M_K / (_temperature_reference__RHO_KG_M3 * _temperature_reference__CP_J_KG_K)
    _temperature_reference__BIOT = _temperature_reference__H_W_M2_K * _temperature_reference__RADIUS_M / _temperature_reference__K_W_M_K

    def _temperature_reference___vector(values, label: str) -> _temperature_reference__np.ndarray:
        result = _temperature_reference__np.asarray(values, dtype=_temperature_reference__np.float64)
        if result.ndim != 1 or result.size == 0:
            raise ValueError(f'{label} must be a nonempty one-dimensional array')
        if not _temperature_reference__np.isfinite(result).all():
            raise ValueError(f'{label} must contain only finite values')
        return result

    @_temperature_reference__lru_cache(maxsize=12)
    def _temperature_reference___modal_basis(modes: int) -> tuple[_temperature_reference__np.ndarray, _temperature_reference__np.ndarray, _temperature_reference__np.ndarray]:
        if isinstance(modes, bool) or not isinstance(modes, (int, _temperature_reference__np.integer)) or modes < 1:
            raise ValueError('modes must be a positive integer')
        zero_bounds = _temperature_reference__jn_zeros(0, int(modes))
        roots = _temperature_reference__np.empty(int(modes), dtype=_temperature_reference__np.float64)

        def residual(value):
            return value * _temperature_reference__j1(value) - _temperature_reference__BIOT * _temperature_reference__j0(value)
        left = 0.0
        for index, right in enumerate(zero_bounds):
            if residual(left) * residual(right) >= 0.0:
                raise RuntimeError(f'Robin root {index + 1} was not bracketed')
            roots[index] = _temperature_reference__brentq(residual, left, float(right), xtol=1e-13, rtol=4.0 * _temperature_reference__np.finfo(_temperature_reference__np.float64).eps)
            left = float(right)
        b0 = _temperature_reference__j0(roots)
        b1 = _temperature_reference__j1(roots)
        weights = 2.0 * b1 / (roots * (b0 * b0 + b1 * b1))
        decay_rates = _temperature_reference__ALPHA_M2_S * (roots / _temperature_reference__RADIUS_M) ** 2
        for array in (roots, weights, decay_rates):
            array.setflags(write=False)
        return (roots, weights, decay_rates)

    def _temperature_reference__reference_temperature(times_s, radii_m, environment_knots_s, temperature_knots_K, modes) -> _temperature_reference__np.ndarray:
        """Return P1 temperature in K with shape (len(times_s), len(radii_m)).

        The initial temperature is INITIAL_T_K everywhere. Input environmental
        temperatures are interpolated linearly between their knots; extrapolation
        is disallowed. Knots must start at t=0 and cover all requested times.
        Requested times must be nondecreasing. Physical radii include 0 and R.
        """
        times = _temperature_reference___vector(times_s, 'times_s')
        radii = _temperature_reference___vector(radii_m, 'radii_m')
        knots = _temperature_reference___vector(environment_knots_s, 'environment_knots_s')
        temperatures = _temperature_reference___vector(temperature_knots_K, 'temperature_knots_K')
        if knots.size < 2 or knots.size != temperatures.size:
            raise ValueError('environmental knot arrays must match and contain at least two knots')
        if knots[0] != 0.0 or _temperature_reference__np.any(_temperature_reference__np.diff(knots) <= 0.0):
            raise ValueError('environmental knot times must start at zero and increase strictly')
        if times[0] < 0.0 or _temperature_reference__np.any(_temperature_reference__np.diff(times) < 0.0) or times[-1] > knots[-1]:
            raise ValueError('requested times must be ordered and lie within the environmental data')
        if _temperature_reference__np.any(radii < 0.0) or _temperature_reference__np.any(radii > _temperature_reference__RADIUS_M):
            raise ValueError('requested radii must lie in [0, 0.02] m')
        roots, weights, rates = _temperature_reference___modal_basis(modes)
        radial_basis = _temperature_reference__j0(roots[:, None] * radii[None, :] / _temperature_reference__RADIUS_M)
        coefficients = weights * (_temperature_reference__INITIAL_T_K - temperatures[0])
        output = _temperature_reference__np.full((times.size, radii.size), _temperature_reference__np.nan, dtype=_temperature_reference__np.float64)
        for index in range(knots.size - 1):
            start, end = knots[index:index + 2]
            interval = end - start
            slope = (temperatures[index + 1] - temperatures[index]) / interval
            final_interval = times[-1] <= end
            mask = (times >= start) & (times <= end if final_interval else times < end)
            if mask.any():
                offsets = times[mask] - start
                exponents = -offsets[:, None] * rates[None, :]
                modal_values = _temperature_reference__np.exp(exponents) * coefficients[None, :] - (weights * slope / rates)[None, :] * -_temperature_reference__np.expm1(exponents)
                environmental = temperatures[index] + slope * offsets
                output[mask, :] = environmental[:, None] + modal_values @ radial_basis
            if final_interval:
                break
            exponents = -rates * interval
            coefficients = _temperature_reference__np.exp(exponents) * coefficients - weights * slope * -_temperature_reference__np.expm1(exponents) / rates
        output[times == 0.0, :] = _temperature_reference__INITIAL_T_K
        if not _temperature_reference__np.isfinite(output).all():
            raise RuntimeError('reference produced an unfilled or nonfinite temperature')
        return output

    def _temperature_reference___sha256(path: _temperature_reference__Path) -> str:
        return _temperature_reference__hashlib.sha256(path.read_bytes()).hexdigest()

    def _temperature_reference__read_temperature_data(input_dir: _temperature_reference__Path):
        """Read and validate original Attachment 1 without sharing solver code."""
        path = input_dir / '附件1.xlsx'
        workbook = _temperature_reference__openpyxl.load_workbook(path, data_only=True, read_only=True)
        try:
            sheet = workbook.worksheets[0]
            header = list(next(sheet.iter_rows(min_row=1, max_row=1, values_only=True)))
            times, temperatures = ([], [])
            for row_number, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), 2):
                if len(row) < 2 or row[0] is None or row[1] is None:
                    raise ValueError(f'Attachment 1 row {row_number} is incomplete')
                if isinstance(row[0], bool) or isinstance(row[1], bool):
                    raise ValueError(f'Attachment 1 row {row_number} is not numeric')
                times.append(float(row[0]))
                temperatures.append(float(row[1]) + 273.15)
            sheet_name = sheet.title
        finally:
            workbook.close()
        knots = _temperature_reference___vector(times, 'attachment times')
        temp_K = _temperature_reference___vector(temperatures, 'attachment temperatures')
        if knots[0] != 0.0 or _temperature_reference__np.any(_temperature_reference__np.diff(knots) <= 0.0):
            raise ValueError('Attachment 1 times must start at 0 and strictly increase')
        if knots[-1] < 1800.0:
            raise ValueError('Attachment 1 does not cover the full P1 interval')
        if abs(temp_K[0] - _temperature_reference__INITIAL_T_K) > 1e-12:
            raise ValueError('Attachment 1 initial temperature is inconsistent with P1')
        return (knots, temp_K, {'filename': path.name, 'sha256': _temperature_reference___sha256(path), 'worksheet': sheet_name, 'headers': header, 'rows': int(knots.size), 'time_start_s': float(knots[0]), 'time_end_s': float(knots[-1]), 'temperature_input_unit': 'degree_Celsius', 'temperature_internal_unit': 'K'})

    def _temperature_reference___difference_summary(left, right, times, radii):
        differences = _temperature_reference__np.abs(left - right)
        flat_index = int(_temperature_reference__np.argmax(differences))
        time_index, radial_index = _temperature_reference__np.unravel_index(flat_index, differences.shape)
        return {'max_abs_K': float(differences[time_index, radial_index]), 'rms_K': float(_temperature_reference__np.sqrt(_temperature_reference__np.mean(differences * differences))), 'max_at_t_s': float(times[time_index]), 'max_at_r_m': float(radii[radial_index]), 'center_max_abs_K': float(differences[:, 0].max()), 'surface_max_abs_K': float(differences[:, -1].max())}

    def _temperature_reference___self_checks(modes: int):
        times = _temperature_reference__np.array([0.0, 1.0, 60.0, 1800.0, 1000000.0])
        radii = _temperature_reference__np.linspace(0.0, _temperature_reference__RADIUS_M, 21)
        knots = _temperature_reference__np.array([0.0, 1000000.0])
        constant = _temperature_reference__reference_temperature(times, radii, knots, [_temperature_reference__INITIAL_T_K] * 2, modes)
        constant_error = float(_temperature_reference__np.abs(constant - _temperature_reference__INITIAL_T_K).max())
        ambient = _temperature_reference__INITIAL_T_K + 20.0
        step = _temperature_reference__reference_temperature(times, radii, knots, [ambient] * 2, modes)
        initial_error = float(_temperature_reference__np.abs(step[0] - _temperature_reference__INITIAL_T_K).max())
        late_error = float(_temperature_reference__np.abs(step[-1] - ambient).max())
        envelope_violation = float(max(_temperature_reference__INITIAL_T_K - step.min(), step.max() - ambient, 0.0))
        affine_times = _temperature_reference__np.arange(0.0, 1801.0)
        affine_unsplit = _temperature_reference__reference_temperature(affine_times, radii, [0.0, 1800.0], [_temperature_reference__INITIAL_T_K, _temperature_reference__INITIAL_T_K + 18.0], modes)
        split_knots = _temperature_reference__np.array([0.0, 33.0, 127.0, 641.0, 1300.0, 1800.0])
        affine_split = _temperature_reference__reference_temperature(affine_times, radii, split_knots, _temperature_reference__INITIAL_T_K + 0.01 * split_knots, modes)
        affine_error = float(_temperature_reference__np.abs(affine_unsplit - affine_split).max())
        checks = {'constant_initial_environment_max_abs_K': constant_error, 'constant_step_initial_max_abs_K': initial_error, 'constant_step_at_1e6_s_max_abs_from_ambient_K': late_error, 'constant_step_envelope_violation_K': envelope_violation, 'affine_partition_invariance_max_abs_K': affine_error, 'tolerance_K': 1e-10}
        checks['passed'] = all((checks[key] <= checks['tolerance_K'] for key in checks if key != 'tolerance_K'))
        return checks

    def _temperature_reference__main(argv=None) -> int:
        parser = _temperature_reference__argparse.ArgumentParser(description=_temperature_reference____doc__)
        parser.add_argument('--input-dir', required=True, type=_temperature_reference__Path)
        parser.add_argument('--output-dir', required=True, type=_temperature_reference__Path)
        parser.add_argument('--truncation-target-K', dest='truncation_target_K', type=float, default=1e-08)
        parser.add_argument('--max-modes', type=int, default=4096)
        args = parser.parse_args(argv)
        if not _temperature_reference__np.isfinite(args.truncation_target_K) or args.truncation_target_K <= 0:
            parser.error('--truncation-target-K must be positive and finite')
        if args.max_modes < 256 or args.max_modes & args.max_modes - 1:
            parser.error('--max-modes must be a power of two at least 256')
        started = _temperature_reference__perf_counter()
        knots, temperatures, input_details = _temperature_reference__read_temperature_data(args.input_dir)
        times = _temperature_reference__np.arange(1801, dtype=_temperature_reference__np.float64)
        radii = _temperature_reference__np.linspace(0.0, _temperature_reference__RADIUS_M, 21)
        records = []
        previous = None
        output = None
        modes = 64
        truncation_converged = False
        while modes <= args.max_modes:
            solve_started = _temperature_reference__perf_counter()
            output = _temperature_reference__reference_temperature(times, radii, knots, temperatures, modes)
            record = {'modes': modes, 'elapsed_s': _temperature_reference__perf_counter() - solve_started}
            if previous is not None:
                record['versus_previous_modes'] = modes // 2
                record['difference'] = _temperature_reference___difference_summary(output, previous, times, radii)
            records.append(record)
            if modes >= 256 and record['difference']['max_abs_K'] <= args.truncation_target_K:
                truncation_converged = True
                break
            if modes == args.max_modes:
                break
            previous = output
            modes *= 2
        roots, weights, _ = _temperature_reference___modal_basis(modes)
        root_residuals = roots * _temperature_reference__j1(roots) - _temperature_reference__BIOT * _temperature_reference__j0(roots)
        root_scale = _temperature_reference__np.maximum(1.0, _temperature_reference__np.abs(roots * _temperature_reference__j1(roots)) + _temperature_reference__np.abs(_temperature_reference__BIOT * _temperature_reference__j0(roots)))
        root_normalized = _temperature_reference__np.abs(root_residuals) / root_scale
        zero_bounds = _temperature_reference__jn_zeros(0, modes)
        left_bounds = _temperature_reference__np.r_[0.0, zero_bounds[:-1]]
        root_checks = {'max_absolute_residual': float(_temperature_reference__np.abs(root_residuals).max()), 'max_scaled_residual': float(root_normalized.max()), 'scaled_residual_tolerance': 1e-09, 'worst_root_index_one_based': int(_temperature_reference__np.argmax(root_normalized)) + 1, 'strictly_increasing': bool(_temperature_reference__np.all(_temperature_reference__np.diff(roots) > 0.0)), 'all_in_expected_open_brackets': bool(_temperature_reference__np.all((roots > left_bounds) & (roots < zero_bounds))), 'first_five_roots': roots[:5].tolist(), 'last_root': float(roots[-1]), 'all_coefficients_finite': bool(_temperature_reference__np.isfinite(weights).all())}
        root_checks['passed'] = root_checks['max_scaled_residual'] <= root_checks['scaled_residual_tolerance'] and root_checks['strictly_increasing'] and root_checks['all_in_expected_open_brackets'] and root_checks['all_coefficients_finite']
        self_checks = _temperature_reference___self_checks(modes)
        used_temperature = temperatures[knots <= 1800.0]
        environmental_min = float(min(_temperature_reference__INITIAL_T_K, used_temperature.min()))
        environmental_max = float(max(_temperature_reference__INITIAL_T_K, used_temperature.max()))
        actual_envelope_violation = float(max(environmental_min - output.min(), output.max() - environmental_max, 0.0))
        initial_error = float(_temperature_reference__np.abs(output[0] - _temperature_reference__INITIAL_T_K).max())
        metrics = {'artifact_kind': 'independent_P1_temperature_reference', 'generated_at_utc': _temperature_reference__datetime.now(_temperature_reference__timezone.utc).isoformat(), 'method': 'continuous_cylindrical_Robin_Bessel_modes_with_exact_affine_time_recurrence', 'independence': 'Reads the original xlsx directly; imports no FVM, MOL, or shared environment code.', 'parameters': {'R_m': _temperature_reference__RADIUS_M, 'rho_kg_m3': _temperature_reference__RHO_KG_M3, 'cp_J_kg_K': _temperature_reference__CP_J_KG_K, 'k_W_m_K': _temperature_reference__K_W_M_K, 'h_W_m2_K': _temperature_reference__H_W_M2_K, 'T0_K': _temperature_reference__INITIAL_T_K, 'alpha_m2_s': _temperature_reference__ALPHA_M2_S, 'Bi_T': _temperature_reference__BIOT}, 'input': input_details, 'sampling': {'shape': list(output.shape), 'time_start_s': 0, 'time_end_s': 1800, 'time_step_s': 1, 'radius_start_m': 0.0, 'radius_end_m': _temperature_reference__RADIUS_M, 'radius_step_m': 0.001, 'includes_initial_condition': True, 'formal_time_rows_exclude_t0': True}, 'modes_selected': modes, 'mode_convergence': records, 'truncation_target_K': args.truncation_target_K, 'truncation_target_met': truncation_converged, 'truncation_statement': 'The last successive mode-doubling difference is an empirical truncation check, not a proved uniform error bound; physical model assumptions are not validated here.', 'root_checks': root_checks, 'self_checks': self_checks, 'actual_environment_checks': {'all_finite': bool(_temperature_reference__np.isfinite(output).all()), 'initial_max_abs_K': initial_error, 'global_input_envelope_violation_K': actual_envelope_violation, 'min_T_K': float(output.min()), 'max_T_K': float(output.max())}, 'versions': {'python': _temperature_reference__sys.version, 'numpy': _temperature_reference__np.__version__, 'scipy': _temperature_reference__scipy.__version__, 'openpyxl': _temperature_reference__openpyxl.__version__, 'platform': _temperature_reference__platform.platform()}, 'source_identity': _temperature_reference__source_identity()}
        metrics['reference_checks_passed'] = bool(truncation_converged and root_checks['passed'] and self_checks['passed'] and (initial_error <= 1e-10) and (actual_envelope_violation <= 1e-08))
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / 'temperature_reference.npz'
        _temperature_reference__np.savez_compressed(npz_path, t_s=times, r_m=radii, T_K=output)
        with _temperature_reference__np.load(npz_path, allow_pickle=False) as reread:
            roundtrip_ok = set(reread.files) == {'t_s', 'r_m', 'T_K'} and _temperature_reference__np.array_equal(reread['t_s'], times) and _temperature_reference__np.array_equal(reread['r_m'], radii) and _temperature_reference__np.array_equal(reread['T_K'], output)
        if not roundtrip_ok:
            raise RuntimeError('Saved reference arrays did not survive an exact round trip')
        metrics['output'] = {'npz_sha256': _temperature_reference___sha256(npz_path), 'npz_bytes': npz_path.stat().st_size, 'exact_npz_roundtrip_passed': roundtrip_ok}
        metrics['elapsed_total_s'] = _temperature_reference__perf_counter() - started
        (output_dir / 'reference_metrics.json').write_text(_temperature_reference__json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(_temperature_reference__json.dumps({'modes': modes, 'last_mode_difference_K': records[-1]['difference']['max_abs_K'], 'root_scaled_residual': root_checks['max_scaled_residual'], 'reference_checks_passed': metrics['reference_checks_passed'], 'elapsed_s': metrics['elapsed_total_s'], 'output_directory_name': output_dir.name}, ensure_ascii=False))
        return 0 if metrics['reference_checks_passed'] else 2

    # ===== solvers/solve_q1.py: readable implementation =====
    _solve_q1____doc__ = 'Reproducible P1 calculation with nodal radial finite volumes and Radau/BDF.'
    import argparse as _solve_q1__argparse
    from datetime import datetime as _solve_q1__datetime, timezone as _solve_q1__timezone
    import json as _solve_q1__json
    from pathlib import Path as _solve_q1__Path
    import platform as _solve_q1__platform
    import time as _solve_q1__time
    _solve_q1__source_identity = source_identity
    import numpy as _solve_q1__np
    from numpy.polynomial.legendre import leggauss as _solve_q1__leggauss
    import scipy as _solve_q1__scipy
    from scipy.integrate import solve_ivp as _solve_q1__solve_ivp
    _solve_q1__NodeRadialGrid = _sections['radial_core'].NodeRadialGrid
    _solve_q1__Environment = _sections['radial_core'].Environment
    _solve_q1__diffusion_rhs = _sections['radial_core'].diffusion_rhs
    _solve_q1__face_arithmetic = _sections['radial_core'].face_arithmetic
    _solve_q1__linear_diffusion_matrix = _sections['radial_core'].linear_diffusion_matrix
    _solve_q1__load_contract = _sections['radial_core'].load_contract
    _solve_q1__sha256 = _sections['radial_core'].sha256
    _solve_q1__tridiagonal_sparsity = _sections['radial_core'].tridiagonal_sparsity

    def _solve_q1__solve_q1(input_dir, contract_path, output_dir, intervals=320, method='Radau', rtol=2e-10, atol_T=1e-11, atol_C=2e-12, quadrature_orders=(48, 96), moisture_rtol=None, solve_ivp_impl=None):
        integrate = _solve_q1__solve_ivp if solve_ivp_impl is None else solve_ivp_impl
        started = _solve_q1__time.perf_counter()
        contract = _solve_q1__load_contract(contract_path)
        if intervals % 20:
            raise ValueError('Use a multiple of 20 intervals so every requested radius is a node')
        out = _solve_q1__Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = contract['initial_state']
        p = contract['property_sets']['P1']
        R, T0, C0 = (cfg['radius_m'], cfg['temperature_K'], cfg['moisture_C'])
        rho_cp, k = (p['rho_offset'] * p['cp_offset'], p['k_offset'])
        h = contract['fixed_domain_model']['h_W_m2_K']
        hm = contract['fixed_domain_model']['h_m_m_s']
        env = _solve_q1__Environment(input_dir, contract)
        grid = _solve_q1__NodeRadialGrid(R, intervals)
        sample_indices = _solve_q1__np.arange(0, intervals + 1, intervals // 20)
        t_out = _solve_q1__np.arange(0.0, 1801.0)
        T_output = _solve_q1__np.full((len(t_out), 21), T0)
        C_output = _solve_q1__np.full((len(t_out), 21), C0)
        mean_T = _solve_q1__np.full(len(t_out), T0)
        mean_C = _solve_q1__np.full(len(t_out), C0)
        theta = _solve_q1__np.zeros(intervals + 1)
        moisture = _solve_q1__np.full(intervals + 1, C0)
        heat_matrix = _solve_q1__linear_diffusion_matrix(grid, k, h, rho_cp)
        boundary_gain = R * h / (grid.weights_m2[-1] * rho_cp)
        sparse_pattern = _solve_q1__tridiagonal_sparsity(intervals + 1)
        quad_rules = [_solve_q1__leggauss(order) for order in quadrature_orders]
        balance_integrals = _solve_q1__np.zeros((len(quad_rules), 2))
        balance_rows = []
        segment_records = []
        snapshots = {}
        profile_times = set(contract['outputs']['paper_tables']['tables_1_2']['time_s'])
        extrema = {'minimum_C_on_grid': C0, 'maximum_C_on_grid': C0, 'minimum_T_K_on_grid': T0, 'maximum_T_K_on_grid': T0}

        def D(C):
            if _solve_q1__np.any(C <= 0) or not _solve_q1__np.all(_solve_q1__np.isfinite(C)):
                raise ValueError('P1 trial state not positive/finite; investigate rather than clip')
            return p['D_prefactor'] * _solve_q1__np.exp(-p['D_moisture_exponent'] / C)

        def heat_rhs(t, y):
            result = heat_matrix @ y
            result[-1] += boundary_gain * (env(t)[0] - T0)
            return result

        def moisture_rhs(t, y):
            flux = hm * (y[-1] - env(t)[1])
            return _solve_q1__diffusion_rhs(grid, y, _solve_q1__face_arithmetic(D(y)), flux)
        max_instant_balance = _solve_q1__np.zeros(2)
        moisture_first_step = min(0.01, 0.1 * (R / intervals) ** 2 / float(D(_solve_q1__np.array([C0]))[0]))
        for left, right in zip(env.breakpoints(0, 1800)[:-1], env.breakpoints(0, 1800)[1:]):
            segment_started = _solve_q1__time.perf_counter()
            heat = integrate(heat_rhs, (left, right), theta, method=method, jac=heat_matrix, rtol=rtol, atol=atol_T, dense_output=True)
            wet = integrate(moisture_rhs, (left, right), moisture, method=method, jac_sparsity=sparse_pattern, rtol=rtol if moisture_rtol is None else moisture_rtol, atol=atol_C, dense_output=True, first_step=moisture_first_step if left == 0 else None)
            if not heat.success or not wet.success:
                raise RuntimeError(f'Segment {left}-{right} failed: {heat.message}; {wet.message}')
            theta, moisture = (heat.y[:, -1], wet.y[:, -1])
            indices = _solve_q1__np.flatnonzero((t_out > left) & (t_out <= right))
            time_values = t_out[indices]
            temperatures = heat.sol(time_values) + T0
            concentrations = wet.sol(time_values)
            T_output[indices] = temperatures[sample_indices].T
            C_output[indices] = concentrations[sample_indices].T
            mean_T[indices] = grid.weights_m2 @ temperatures / (R * R / 2)
            mean_C[indices] = grid.weights_m2 @ concentrations / (R * R / 2)
            for key, value in (('minimum_C_on_grid', _solve_q1__np.min(concentrations)), ('minimum_T_K_on_grid', _solve_q1__np.min(temperatures))):
                extrema[key] = min(extrema[key], float(value))
            for key, value in (('maximum_C_on_grid', _solve_q1__np.max(concentrations)), ('maximum_T_K_on_grid', _solve_q1__np.max(temperatures))):
                extrema[key] = max(extrema[key], float(value))
            for ts in time_values:
                if ts in profile_times:
                    j = int(ts - left - 1)
                    snapshots[str(int(ts))] = (temperatures[:, j].copy(), concentrations[:, j].copy())
            for q, (nodes, weights) in enumerate(quad_rules):
                times = (left + right) / 2 + (right - left) / 2 * nodes
                eT, eC = env(times)
                T_surface = heat.sol(times)[-1] + T0
                C_surface = wet.sol(times)[-1]
                dmeanT = -2 * h / (rho_cp * R) * (T_surface - eT)
                dmeanC = -2 * hm / R * (C_surface - eC)
                balance_integrals[q] += (right - left) / 2 * _solve_q1__np.array([weights @ dmeanT, weights @ dmeanC])
            true_change = _solve_q1__np.array([grid.weights_m2 @ theta / (R * R / 2), grid.weights_m2 @ moisture / (R * R / 2) - C0])
            residuals = true_change[None, :] - balance_integrals
            balance_rows.append({'t_s': float(right), 'mean_T_K': float(true_change[0] + T0), 'mean_C': float(true_change[1] + C0), 'residuals_by_quadrature': residuals.tolist()})
            eT, eC = env(right)
            instant = _solve_q1__np.array([grid.weights_m2 @ heat_rhs(right, theta) + R * h / rho_cp * (theta[-1] + T0 - eT), grid.weights_m2 @ moisture_rhs(right, moisture) + R * hm * (moisture[-1] - eC)])
            max_instant_balance = _solve_q1__np.maximum(max_instant_balance, _solve_q1__np.abs(instant))
            segment_records.append({'start_s': float(left), 'end_s': float(right), 'T_nfev': heat.nfev, 'C_nfev': wet.nfev, 'T_steps': len(heat.t) - 1, 'C_steps': len(wet.t) - 1, 'T_nlu': heat.nlu, 'C_nlu': wet.nlu, 'seconds': _solve_q1__time.perf_counter() - segment_started})
        assert _solve_q1__np.all(_solve_q1__np.isfinite(T_output)) and _solve_q1__np.all(_solve_q1__np.isfinite(C_output))
        _solve_q1__np.savez_compressed(out / 'solution.npz', t_s=t_out, r_m=grid.r_m[sample_indices], T_K=T_output, C=C_output, mean_T_K=mean_T, mean_C=mean_C)
        _solve_q1__np.savez_compressed(out / 'grid_profiles.npz', r_m=grid.r_m, t_s=_solve_q1__np.array(sorted(profile_times)), T_K=_solve_q1__np.array([snapshots[str(t)][0] for t in sorted(profile_times)]), C=_solve_q1__np.array([snapshots[str(t)][1] for t in sorted(profile_times)]))
        metadata = {'created_at_utc': _solve_q1__datetime.now(_solve_q1__timezone.utc).isoformat(), 'problem': 'q1', 'property_set': 'P1', 'model_version': contract['model_version'], 'stage_status': 'candidate_pending_aggregate_verification', 'intervals': intervals, 'grid_nodes': intervals + 1, 'state_interpretation': 'nodal_point_approximations_with_half_distance_FV_weights', 'thermal_integration_state': 'theta_K=T_K-T0_K', 'method': method, 'rtol': rtol, 'moisture_rtol': rtol if moisture_rtol is None else moisture_rtol, 'atol_theta_K': atol_T, 'atol_C': atol_C, 'moisture_first_step_s': moisture_first_step, 'face_rule': 'arithmetic_mean_of_nodal_coefficients', 'break_at_every_environment_knot': True, 'uses_environment_extrapolation': False, 'internal_initial_row': True, 'formal_first_time_s': 1, 'sample_schema': {'t_s': '(nt,)', 'r_m': '(nr,)', 'T_K': '(nt,nr)', 'C': '(nt,nr)'}, 'source_identity': _solve_q1__source_identity(), 'input_sha256': {env.path.name: env.source_sha256, 'model_contract': _solve_q1__sha256(contract_path)}, 'software': {'python': _solve_q1__platform.python_version(), 'numpy': _solve_q1__np.__version__, 'scipy': _solve_q1__scipy.__version__}, 'quadrature_orders': list(quadrature_orders), 'balance_history': balance_rows, 'max_instant_weighted_balance_abs': max_instant_balance.tolist(), 'final_independent_quadrature_difference': _solve_q1__np.abs(balance_integrals[-1] - balance_integrals[-2]).tolist(), 'extrema_at_all_integer_times_all_grid_nodes': extrema, 'segments': segment_records, 'elapsed_seconds': _solve_q1__time.perf_counter() - started}
        (out / 'metadata.json').write_text(_solve_q1__json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        print(_solve_q1__json.dumps({'output_directory_name': out.name, 'N': intervals, 'method': method, 'elapsed_s': metadata['elapsed_seconds'], 'final_T_C': (T_output[-1, [0, -1]] - 273.15).tolist(), 'final_C': C_output[-1, [0, -1]].tolist(), 'balance': balance_rows[-1]['residuals_by_quadrature'][-1]}, ensure_ascii=False), flush=True)
        return metadata

    # ===== solvers/solve_all.py: readable implementation =====
    _solve_all____doc__ = 'Recompute all four questions, convergence evidence and dynamic dry endpoints.\n\nNo report time is hard coded. Strictness refers to the continuous piecewise-\nlinear reconstruction of the computed spatial field; mesh differences are\nengineering error estimates, not certified bounds on the continuum PDE.'
    from pathlib import Path as _solve_all__Path
    import argparse as _solve_all__argparse
    import csv as _solve_all__csv
    import hashlib as _solve_all__hashlib
    import json as _solve_all__json
    import math as _solve_all__math
    import sys as _solve_all__sys
    import time as _solve_all__time
    from decimal import Decimal as _solve_all__Decimal
    _solve_all__source_identity = source_identity
    import numpy as _solve_all__np
    from numpy.polynomial.legendre import leggauss as _solve_all__leggauss
    from scipy.integrate import simpson as _solve_all__simpson
    _solve_all__m = _sections['moving_model']
    _solve_all__solve_q1 = _sections['solve_q1'].solve_q1
    _solve_all__reference_temperature = _sections['temperature_reference'].reference_temperature
    _solve_all__RelativeTemperatureModel = _sections['relative_temperature'].RelativeTemperatureModel
    _solve_all__wrap = _sections['relative_temperature'].wrap

    def _solve_all__sha(p):
        return _solve_all__hashlib.sha256(_solve_all__Path(p).read_bytes()).hexdigest()

    def _solve_all__write(p, d):
        p = _solve_all__Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_solve_all__json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    def _solve_all__save_csv(p, rows):
        p = _solve_all__Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8-sig', newline='') as f:
            w = _solve_all__csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def _solve_all__times(step, end, start=None):
        return _solve_all__m.upstream.requested_time_grid(step if start is None else start, step, end)

    def _solve_all__fields(traj, t, r):
        T, C = traj.evaluate_fields(t)
        return (_solve_all__m.upstream.sample_radii(T, traj.model.grid.r_m, r), _solve_all__m.upstream.sample_radii(C, traj.model.grid.r_m, r))

    def _solve_all__settings(kind, tight=False):
        ratio = 0.1 if tight else 1.0
        return _solve_all__m.IntegrationSettings('BDF', 1e-10 * ratio, (1e-10 if kind == 'p23' else 1e-09) * ratio, 1e-12 * ratio, 75.0 if kind == 'p23' else 600.0, True)

    def _solve_all__run_case(args, kind, N, tight=False):
        model = _solve_all__m.build_model(args.inputs, args.contract, N=N, properties='P23' if kind == 'p23' else 'P4', geometry='fixed' if kind == 'p23' else 'measured')
        model = _solve_all__RelativeTemperatureModel.from_model(model)
        s = _solve_all__settings(kind, tight)
        started = _solve_all__time.perf_counter()
        tr = _solve_all__wrap(_solve_all__m.solve_moving_to_event(model, s, horizon_s=864000.0, post_event_s=2.0))
        print(_solve_all__json.dumps({'case': kind, 'N': N, 'tight': tight, 'root_h': tr.event_time_s / 3600, 'seconds': _solve_all__time.perf_counter() - started}), flush=True)
        return tr

    def _solve_all__compact_sample(tr, end):
        early = _solve_all__np.arange(1.0, 10801.0)
        late = _solve_all__np.unique(_solve_all__np.r_[_solve_all__np.arange(10860.0, end, 60.0), end])
        t = _solve_all__np.r_[early[early < end], late]
        r = _solve_all__np.arange(21) * 0.001
        T = []
        C = []
        for start in range(0, len(t), 128):
            ts = t[start:start + 128]
            data = _solve_all__m.sample_physical(tr, ts, r[:-1])
            T.append(_solve_all__np.c_[data['T_K_sample'], data['T_K_surface']])
            C.append(_solve_all__np.c_[data['C_sample'], data['C_surface']])
        return (t, _solve_all__np.concatenate(T), _solve_all__np.concatenate(C))

    def _solve_all__balance(tr, end):
        model = tr.model
        weight = 2 * model.grid.volume_weights_m2 / model.grid.radius_m ** 2
        integrations = []
        for order in (8, 16):
            nodes, weights = _solve_all__leggauss(order)
            sum_c = 0.0
            sum_heat = 0.0
            abs_heat = 0.0
            sum_heat_store = 0.0
            for seg in tr.segments:
                if seg.t_start_s >= end:
                    break
                dense_solution = seg.result.sol
                for left, right, poly in zip(dense_solution.ts[:-1], dense_solution.ts[1:], dense_solution.interpolants):
                    right = min(float(right), end)
                    if right <= left:
                        continue
                    tq = (left + right) / 2 + (right - left) / 2 * nodes
                    y = poly(tq)
                    theta = y[:model.nr].T
                    T = theta + model.temperature_reference_K
                    C = y[model.nr:].T
                    R = _solve_all__np.asarray(model.geometry.radius(tq))
                    env = model.ambient.values(tq, right_of_breakpoint=seg.right_of_breakpoint)
                    flux_c = 2 * model.h_m_m_s / R * (C[:, -1] - env.moisture_eq)
                    J = (R / model.grid.radius_m) ** 2
                    flux_h = J * 2 * model.h_W_m2_K / R * (theta[:, -1] - (env.temperature_K - model.temperature_reference_K))
                    products = _solve_all__np.ones(len(tq))
                    derivatives = _solve_all__np.zeros(len(tq))
                    dy = _solve_all__np.zeros_like(y)
                    for degree in range(poly.order):
                        xx = (tq - poly.t_shift[degree]) / poly.denom[degree]
                        derivatives = derivatives * xx + products / poly.denom[degree]
                        products = products * xx
                        dy += poly.D[degree + 1, :, None] * derivatives[None, :]
                    rho, cp, _, _ = model.properties.evaluate(C, T)
                    thermal_store = J * (rho * cp * dy[:model.nr].T @ weight)
                    width = (right - left) / 2
                    sum_c += width * (weights @ flux_c)
                    sum_heat += width * (weights @ flux_h)
                    abs_heat += width * (weights @ _solve_all__np.abs(flux_h))
                    sum_heat_store += width * (weights @ thermal_store)
            mean0 = float(weight @ tr.evaluate_fields(0.0)[1])
            mean1 = float(weight @ tr.evaluate_fields(end)[1])
            integrations.append({'quadrature_order': order, 'mean_C_change': mean1 - mean0, 'integrated_moisture_outflux': sum_c, 'moisture_balance_abs': abs(mean1 - mean0 + sum_c), 'thermal_rate_integral_residual_abs': abs(sum_heat_store + sum_heat), 'thermal_rate_integral_residual_relative': abs(sum_heat_store + sum_heat) / max(abs_heat, 1e-30)})
        ts = _solve_all__np.unique(_solve_all__np.r_[0.0, 1.0, 60.0, 1800.0, 14400.0, _solve_all__np.arange(15000.0, end, 600.0), end])
        audits = [model.audit(float(t), tr.evaluate_state(float(t)), right_of_breakpoint=t > 14400) for t in ts]
        _, C = tr.evaluate_fields(ts)
        simpson_means = 2 * _solve_all__simpson(C * model.xi, x=model.xi, axis=1)
        weighted = C @ weight
        return {'quadrature': integrations, 'quadrature_flux_difference': abs(integrations[0]['integrated_moisture_outflux'] - integrations[1]['integrated_moisture_outflux']), 'maximum_semidiscrete_moisture_residual': max((abs(a['moisture_mean_rate_residual']) for a in audits)), 'maximum_semidiscrete_thermal_residual': max((abs(a['thermal_current_volume_rate_residual']) for a in audits)), 'independent_Simpson_mean_difference': float(_solve_all__np.max(abs(simpson_means - weighted))), 'maximum_positive_radial_C_difference': float(_solve_all__np.max(_solve_all__np.diff(C, axis=1))), 'maximum_location_xi': max((a['maximum_C_xi'] for a in audits)), 'thermal_scope': 'Integral of J*b(C)*T_t from independently differentiated BDF interpolant versus current-boundary flux, per initial volume; not latent heat or derivative of b(C)*T*J.'}

    def _solve_all__solve_long(args, kind):
        out = args.out / kind
        out.mkdir(parents=True, exist_ok=False)
        Ns = [320, 640, 1280] if kind == 'p23' else [640, 1280, 2560]
        trajectories = [_solve_all__run_case(args, kind, N) for N in Ns]
        fine = trajectories[-1]
        tight = _solve_all__run_case(args, kind, Ns[-1], True)
        root = fine.event_time_s
        space = abs(root - trajectories[-2].event_time_s)
        temporal = abs(root - tight.event_time_s)
        margin = space + temporal
        tick = 0.36
        index = _solve_all__math.ceil((root + margin) / tick)
        report = float(_solve_all__Decimal(index) * _solve_all__Decimal('0.36'))
        if report <= root:
            report = float(_solve_all__Decimal(index + 1) * _solve_all__Decimal('0.36'))
        if report > fine.final_time_s:
            s = _solve_all__settings(kind)
            seg = _solve_all__m.upstream._solve_segment(fine.model, (fine.final_time_s, report), fine.final_state, s, right_of_breakpoint=True, detect_event=False)
            fine.segments.append(seg)
        maximum = float(_solve_all__np.max(fine.evaluate_fields(report)[1]))
        assert maximum < 0.15
        check_end = min((tr.event_time_s for tr in trajectories + [tight]))
        samples = [_solve_all__compact_sample(tr, check_end) for tr in trajectories + [tight]]
        rows = []
        for j, N in enumerate(Ns):
            row = {'case': kind, 'N': N, 'root_s': trajectories[j].event_time_s, 'root_h': trajectories[j].event_time_s / 3600, 'root_difference_to_next_s': 0.0 if j == 2 else abs(trajectories[j].event_time_s - trajectories[j + 1].event_time_s), 'sample_T_difference_to_next_K': 0.0 if j == 2 else float(_solve_all__np.nanmax(abs(samples[j][1] - samples[j + 1][1]))), 'sample_C_difference_to_next': 0.0 if j == 2 else float(_solve_all__np.nanmax(abs(samples[j][2] - samples[j + 1][2])))}
            rows.append(row)
        convergence = {'meshes': rows, 'time_tolerance_root_difference_s': temporal, 'tight_equality_root_s': tight.event_time_s, 'tight_settings': _solve_all__settings(kind, True).__dict__, 'time_tolerance_sample_T_difference_K': float(_solve_all__np.nanmax(abs(samples[2][1] - samples[3][1]))), 'time_tolerance_sample_C_difference': float(_solve_all__np.nanmax(abs(samples[2][2] - samples[3][2]))), 'comparison_end_s': check_end, 'sampling': 'every second to 3h, then every 60s; 20 physical radii plus true surface'}
        endpoint = {'equality_root_s': root, 'equality_root_h': root / 3600, 'spatial_root_difference_s': space, 'time_tolerance_root_difference_s': temporal, 'engineering_margin_s': margin, 'reported_time_s': report, 'reported_time_h': report / 3600, 'reported_max_C': maximum, 'strict_reconstructed_field_dry': True, 'continuous_space_maximum': 'exact maximum of the piecewise-linear reconstructed radial field occurs at a node', 'continuous_PDE_error_certified': False, 'interpretation': 'Equality crossing is the infimum of later strictly dry times; report adds empirical mesh/time allowance and rounds to dry side.'}
        _solve_all__write(out / 'convergence.json', convergence)
        _solve_all__write(out / 'endpoint.json', endpoint)
        _solve_all__save_csv(out / 'mesh_convergence.csv', rows)
        _solve_all__write(out / 'balance.json', _solve_all__balance(fine, report))
        internal_times = _solve_all__np.r_[0.0, _solve_all__times(60.0, report)]
        T, C = fine.evaluate_fields(internal_times)
        R = _solve_all__np.asarray(fine.model.geometry.radius(internal_times))
        w = 2 * fine.model.grid.volume_weights_m2 / fine.model.grid.radius_m ** 2
        _solve_all__np.savez_compressed(out / 'internal.npz', t_s=internal_times, xi=fine.model.xi, R_m=R, T_K=T, C=C, mean_C=C @ w)
        paper_times = _solve_all__np.r_[_solve_all__np.arange(21600.0, report, 21600.0), report]
        if kind == 'p23':
            data = out / 'data'
            data.mkdir()
            rr = _solve_all__np.arange(21) * 0.001
            t3 = _solve_all__times(60.0, report)
            T3, C3 = _solve_all__fields(fine, t3, rr)
            _solve_all__np.savez_compressed(data / 'q3_fixed_grid_60s.npz', t_s=t3, r_m=rr, T_K=T3, C=C3)
            t2 = _solve_all__times(1.0, report)
            chunks = []
            chunkdir = data / 'q2_fixed_grid_1s_chunks'
            chunkdir.mkdir()
            for j, start in enumerate(range(0, len(t2), 10800)):
                ts = t2[start:start + 10800]
                aa = []
                bb = []
                for begin in range(0, len(ts), 128):
                    a, b = _solve_all__fields(fine, ts[begin:begin + 128], rr)
                    aa.append(a)
                    bb.append(b)
                p = chunkdir / f'q2_1s_{j:03d}.npz'
                _solve_all__np.savez_compressed(p, t_s=ts, r_m=rr, T_K=_solve_all__np.concatenate(aa), C=_solve_all__np.concatenate(bb))
                chunks.append({'file': p.relative_to(out).as_posix(), 'rows': len(ts), 'sha256': _solve_all__sha(p)})
            _solve_all__write(data / 'q2_fixed_grid_1s_index.json', {'chunks': chunks, 'row_count': len(t2), 'final_time_s': report})
            tt = _solve_all__np.arange(1800.0, 10801.0, 1800.0)
            a, b = _solve_all__fields(fine, tt, _solve_all__np.arange(5) * 0.005)
            _solve_all__np.savez_compressed(out / 'paper_first3h.npz', t_s=tt, T_K=a, C=b)
            a, b = _solve_all__fields(fine, paper_times, _solve_all__np.arange(5) * 0.005)
            _solve_all__np.savez_compressed(out / 'paper_full.npz', t_s=paper_times, T_K=a, C=b)
        else:
            result = _solve_all__m.sample_physical(fine, _solve_all__times(60.0, report), _solve_all__np.arange(20) * 0.001)
            _solve_all__np.savez_compressed(out / 'result4_candidate_data.npz', **result)
            _solve_all__np.savez_compressed(out / 'paper_full.npz', **_solve_all__m.sample_physical(fine, paper_times, _solve_all__np.arange(4) * 0.005))
            sample_t = _solve_all__times(60.0, report)
            linear = _solve_all__m.sample_physical(fine, sample_t, _solve_all__np.arange(20) * 0.001)
            smooth = _solve_all__m.sample_physical(fine, sample_t, _solve_all__np.arange(20) * 0.001, method='pchip')
            _solve_all__write(out / 'interpolation_check.json', {'linear_vs_pchip_C_max': float(_solve_all__np.nanmax(abs(linear['C_sample'] - smooth['C_sample']))), 'linear_vs_pchip_T_max_K': float(_solve_all__np.nanmax(abs(linear['T_K_sample'] - smooth['T_K_sample']))), 'formal_method': 'linear', 'only_actual_interior_positions_compared': True})
        _solve_all__write(out / 'metadata.json', {'question': kind, 'N': Ns[-1], 'endpoint': endpoint, 'settings': _solve_all__settings(kind).__dict__, 'thermal_state': 'theta=T-Tref; Tref is the extrapolated ambient temperature; diffusivity uses absolute K', 'source_identity': _solve_all__source_identity()})
        return endpoint

    def _solve_all__solve_first(args):
        out = args.out / 'q1'
        out.mkdir(parents=True, exist_ok=False)
        Ns = [2560, 5120, 10240]
        records = []
        for N in Ns:
            _solve_all__solve_q1(args.inputs, args.contract, out / f'N{N}', intervals=N)
        _solve_all__solve_q1(args.inputs, args.contract, out / 'tight', intervals=10240, moisture_rtol=2e-12, atol_C=2e-14)
        arrays = [_solve_all__np.load(out / f'N{N}/solution.npz') for N in Ns]
        last = arrays[-1]
        tight = _solve_all__np.load(out / 'tight/solution.npz')
        for j, N in enumerate(Ns):
            records.append({'N': N, 'max_T_difference_to_next_K': 0.0 if j == 2 else float(_solve_all__np.max(abs(arrays[j]['T_K'] - arrays[j + 1]['T_K']))), 'max_C_difference_to_next': 0.0 if j == 2 else float(_solve_all__np.max(abs(arrays[j]['C'] - arrays[j + 1]['C'])))})
        ambient = _solve_all__m.AmbientProfile.from_xlsx(args.inputs / '附件1.xlsx')
        reference = _solve_all__reference_temperature(last['t_s'], last['r_m'], ambient.t_s, ambient.temperature_C + 273.15, 4096)
        coarser = _solve_all__reference_temperature(last['t_s'], last['r_m'], ambient.t_s, ambient.temperature_C + 273.15, 2048)
        checks = {'meshes': records, 'temperature_reference_max_difference_K': float(_solve_all__np.max(abs(last['T_K'] - reference))), 'reference_mode_2048_to_4096_difference_K': float(_solve_all__np.max(abs(reference - coarser))), 'temperature_validation': 'Independent converged Bessel reference; tight run changes moisture tolerances only.', 'time_tolerance_C_difference': float(_solve_all__np.max(abs(last['C'] - tight['C'])))}
        _solve_all__write(out / 'convergence.json', checks)
        _solve_all__save_csv(out / 'mesh_convergence.csv', records)
        return checks

    def _solve_all__main():
        ap = _solve_all__argparse.ArgumentParser(description=_solve_all____doc__)
        ap.add_argument('--inputs', required=True, type=_solve_all__Path)
        ap.add_argument('--out', required=True, type=_solve_all__Path)
        ap.add_argument('--contract', type=_solve_all__Path, required=True)
        ap.add_argument('--question', choices=['q1', 'p23', 'q4', 'all'], default='all')
        a = ap.parse_args()
        a.out.mkdir(parents=True, exist_ok=True)
        initial = {p.name: _solve_all__sha(p) for p in a.inputs.glob('附件*.xlsx')}
        results = {}
        if a.question in ('q1', 'all'):
            results['q1'] = _solve_all__solve_first(a)
        for kind in ('p23', 'q4'):
            if a.question in (kind, 'all'):
                results[kind] = _solve_all__solve_long(a, kind)
        assert initial == {p.name: _solve_all__sha(p) for p in a.inputs.glob('附件*.xlsx')}
        _solve_all__write(a.out / ('run_' + a.question + '.json'), {'input_hashes': initial, 'results': results, 'original_inputs_unchanged': True})

    # ===== validation/absolute_temperature_check.py: readable implementation =====
    _absolute_temperature_check____doc__ = 'Streaming BDF base used by the endpoint workflows.\n\nThe original integration, event detection and observation order are retained.\nStandalone historical analysis and its filesystem layout are not part of the\nfour-question delivery.'
    import ctypes as _absolute_temperature_check__ctypes
    import hashlib as _absolute_temperature_check__hashlib
    import json as _absolute_temperature_check__json
    import os as _absolute_temperature_check__os
    from time import perf_counter as _absolute_temperature_check__perf_counter
    import numpy as _absolute_temperature_check__np
    from scipy.integrate import BDF as _absolute_temperature_check__BDF
    from scipy.integrate._ivp.ivp import solve_event_equation as _absolute_temperature_check__solve_event_equation
    _absolute_temperature_check__moving = _sections['moving_model']
    _absolute_temperature_check__core = _absolute_temperature_check__moving.upstream

    def _absolute_temperature_check__writej(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_absolute_temperature_check__json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def _absolute_temperature_check__memory_status():
        if _absolute_temperature_check__os.name != 'nt':
            return {}

        class Counters(_absolute_temperature_check__ctypes.Structure):
            _fields_ = [('cb', _absolute_temperature_check__ctypes.c_ulong), ('PageFaultCount', _absolute_temperature_check__ctypes.c_ulong)] + [(n, _absolute_temperature_check__ctypes.c_size_t) for n in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]

        class Memory(_absolute_temperature_check__ctypes.Structure):
            _fields_ = [('length', _absolute_temperature_check__ctypes.c_ulong), ('load', _absolute_temperature_check__ctypes.c_ulong)] + [(n, _absolute_temperature_check__ctypes.c_ulonglong) for n in ['total_phys', 'available_phys', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'available_extended_virtual']]
        kernel = _absolute_temperature_check__ctypes.WinDLL('kernel32')
        psapi = _absolute_temperature_check__ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = _absolute_temperature_check__ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [_absolute_temperature_check__ctypes.c_void_p, _absolute_temperature_check__ctypes.c_void_p, _absolute_temperature_check__ctypes.c_ulong]
        c = Counters()
        c.cb = _absolute_temperature_check__ctypes.sizeof(c)
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), _absolute_temperature_check__ctypes.byref(c), c.cb)
        m = Memory()
        m.length = _absolute_temperature_check__ctypes.sizeof(m)
        kernel.GlobalMemoryStatusEx(_absolute_temperature_check__ctypes.byref(m))
        return {'working_set_bytes': int(c.WorkingSetSize), 'private_bytes': int(c.PrivateUsage), 'available_physical_bytes': int(m.available_phys), 'memory_load_percent': int(m.load)}

    def _absolute_temperature_check__settings():
        return _absolute_temperature_check__core.IntegrationSettings('BDF', 1e-11, 1e-10, 1e-13, 600.0, True)

    class _absolute_temperature_check__StreamingBDF:

        def __init__(self, model, recorder, progress_seconds=50.0, capture_step_times=False):
            self.model = model
            self.recorder = recorder
            self.s = _absolute_temperature_check__settings()
            self.step_count = 0
            self.nfev = 0
            self.njev = 0
            self.nlu = 0
            self.segments = []
            self.step_hash = _absolute_temperature_check__hashlib.sha256()
            self.step_times = [] if capture_step_times else None
            self.started = _absolute_temperature_check__perf_counter()
            self.last_progress = self.started
            self.progress_seconds = progress_seconds
            self.event_time = None
            self.event_state = None
            self.event_steps = None

        def progress(self, solver=None, force=False, stage='running'):
            now = _absolute_temperature_check__perf_counter()
            if not force and now - self.last_progress < self.progress_seconds:
                return
            info = {'status': stage, 'physical_time_s': float(solver.t) if solver else self.event_time or 0, 'physical_time_h': float(solver.t / 3600) if solver else (self.event_time or 0) / 3600, 'accepted_steps': self.step_count, 'step_size_s': float(solver.step_size or 0) if solver else None, 'nfev': self.nfev + (solver.nfev if solver else 0), 'njev': self.njev + (solver.njev if solver else 0), 'nlu': self.nlu + (solver.nlu if solver else 0), 'current_max_C': float(solver.y[self.model.nr:].max()) if solver else None, 'wall_seconds': now - self.started, **_absolute_temperature_check__memory_status()}
            _absolute_temperature_check__writej(self.recorder.directory / 'progress.json', info)
            with (self.recorder.directory / 'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(_absolute_temperature_check__json.dumps(info, ensure_ascii=False) + '\n')
            self.recorder.flush()
            print(_absolute_temperature_check__json.dumps(info), flush=True)
            self.last_progress = now

        def span(self, a, b, y, targets, detect_event=True):
            model = self.model
            s = self.s
            right = bool(a >= model.ambient.breakpoint_s and _absolute_temperature_check__np.isfinite(model.ambient.breakpoint_s))
            first = None
            if s.limit_initial_step and a == 0:
                T, C = model.unpack(y)
                D = model.properties.evaluate(C, T)[3]
                maximum = float(D.max())
                if maximum > 0:
                    first = min(0.01, 0.1 * model.grid.minimum_spacing_m ** 2 / maximum, b - a)
            kwargs = {} if first is None else {'first_step': first}
            solver = _absolute_temperature_check__BDF(lambda t, u: model.rhs(t, u, right_of_breakpoint=right), float(a), y, float(b), rtol=s.rtol, atol=s.atol_vector(model.nr), max_step=s.max_step_s, jac_sparsity=model.jacobian_sparsity(), **kwargs)
            event = _absolute_temperature_check__core._event_function(model)
            oldg = event(a, y)
            p = 0
            while p < len(targets) and targets[p] <= a + 1e-10:
                p += 1
            step_start = self.step_count
            terminated = False
            while solver.status == 'running':
                left = float(solver.t)
                message = solver.step()
                if solver.status == 'failed':
                    raise RuntimeError(f'BDF failed at {left}: {message}')
                self.step_count += 1
                right_t = float(solver.t)
                dense = solver.dense_output()
                newg = event(right_t, solver.y)
                if detect_event and oldg >= 0 and (newg <= 0):
                    right_t = float(_absolute_temperature_check__solve_event_equation(event, dense, left, right_t))
                    y = _absolute_temperature_check__np.asarray(dense(right_t))
                    self.event_time = right_t
                    self.event_state = y.copy()
                    self.event_steps = self.step_count
                    terminated = True
                else:
                    y = _absolute_temperature_check__np.asarray(solver.y)
                self.step_hash.update(_absolute_temperature_check__np.float64(right_t).tobytes())
                if self.step_times is not None:
                    self.step_times.append(right_t)
                while p < len(targets) and targets[p] <= right_t + 1e-10:
                    t = float(targets[p])
                    self.recorder.add(t, _absolute_temperature_check__np.asarray(dense(t)), None if detect_event else 'post_event_tick')
                    p += 1
                if terminated:
                    self.recorder.add(right_t, y, 'equality_root')
                self.progress(solver)
                del dense
                oldg = newg
                if terminated:
                    break
            self.nfev += solver.nfev
            self.njev += solver.njev
            self.nlu += solver.nlu
            self.segments.append({'start_s': float(a), 'requested_end_s': float(b), 'actual_end_s': right_t, 'first_step_s': first, 'accepted_steps': self.step_count - step_start, 'nfev': solver.nfev, 'njev': solver.njev, 'nlu': solver.nlu, 'event': terminated})
            del solver
            return (y, terminated)

        def integrate(self, horizon=864000.0, post_event_s=2.0):
            model = self.model
            y = model.initial_state(301.15, 2.55)
            self.recorder.add(0.0, y, 'initial')
            targets = _absolute_temperature_check__np.unique(_absolute_temperature_check__np.r_[_absolute_temperature_check__np.arange(1.0, 10801.0), _absolute_temperature_check__np.arange(10860.0, horizon + 1, 60.0)])
            points = model.ambient.breakpoints(0, horizon)
            self.progress(force=True, stage='starting')
            for a, b in zip(points[:-1], points[1:]):
                selected = targets[(targets > a) & (targets <= b)]
                y, event = self.span(float(a), float(b), y, selected, True)
                if event:
                    break
            if self.event_time is None:
                raise RuntimeError('Drying event not reached within original 240h horizon')
            root = self.event_time
            end = root + post_event_s
            ticks = _absolute_temperature_check__np.arange(_absolute_temperature_check__np.ceil(root / 0.36), _absolute_temperature_check__np.floor(end / 0.36) + 1) * 0.36
            extra = _absolute_temperature_check__np.unique(_absolute_temperature_check__np.r_[targets[(targets > root) & (targets <= end)], ticks, end])
            points = model.ambient.breakpoints(root, end)
            for a, b in zip(points[:-1], points[1:]):
                selected = extra[(extra > a) & (extra <= b)]
                y, _ = self.span(float(a), float(b), y, selected, False)
            self.recorder.checkpoints[end] = y.copy()
            self.recorder.checkpoint_labels[end] = 'post_event_final'
            self.recorder.flush(final=True)
            self.progress(force=True, stage='complete')
            return {'event_root_s': root, 'event_root_h': root / 3600, 'post_event_final_s': end, 'event_max_C': float(self.event_state[model.nr:].max()), 'accepted_steps_to_event': self.event_steps, 'accepted_steps_including_post': self.step_count, 'accepted_step_time_sha256': self.step_hash.hexdigest(), 'nfev': self.nfev, 'njev': self.njev, 'nlu': self.nlu, 'segments': self.segments, 'wall_seconds': _absolute_temperature_check__perf_counter() - self.started, 'memory_at_completion': _absolute_temperature_check__memory_status()}

    # ===== validation/q1_full_tolerance.py: readable implementation =====
    _q1_full_tolerance____doc__ = 'Complete P1 with both heat and moisture tolerances tightened by 100.\n\nCalls the statically merged solve_q1 implementation at N=10240. No timeout, reduced\nquadrature, different initial step, clipping or solver substitution is used.\nThis independent addition does not replace the production or C-only check.'
    import os as _q1_full_tolerance__os
    for _q1_full_tolerance___key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        _q1_full_tolerance__os.environ[_q1_full_tolerance___key] = '1'
    import sys as _q1_full_tolerance__sys
    _q1_full_tolerance__sys.dont_write_bytecode = True
    from pathlib import Path as _q1_full_tolerance__Path
    from datetime import datetime as _q1_full_tolerance__datetime, timezone as _q1_full_tolerance__timezone
    from time import perf_counter as _q1_full_tolerance__perf_counter
    from decimal import Decimal as _q1_full_tolerance__Decimal, ROUND_HALF_UP as _q1_full_tolerance__ROUND_HALF_UP
    import argparse as _q1_full_tolerance__argparse
    import csv as _q1_full_tolerance__csv
    import hashlib as _q1_full_tolerance__hashlib
    import json as _q1_full_tolerance__json
    import threading as _q1_full_tolerance__threading
    import inspect as _q1_full_tolerance__inspect
    _q1_full_tolerance__source_identity = source_identity
    from scipy.integrate import Radau as _q1_full_tolerance__Radau, solve_ivp as _q1_full_tolerance__scipy_solve_ivp
    from scipy.integrate._ivp.common import OdeSolution as _q1_full_tolerance__OdeSolution
    import numpy as _q1_full_tolerance__np
    _q1_full_tolerance__solve_q1 = _sections['solve_q1'].solve_q1
    _q1_full_tolerance__reference_temperature = _sections['temperature_reference'].reference_temperature
    _q1_full_tolerance__read_temperature_data = _sections['temperature_reference'].read_temperature_data
    _q1_full_tolerance__SETTINGS = {'method': 'Radau', 'rtol': 2e-12, 'atol_T': 1e-13, 'atol_C': 2e-14, 'quadrature_orders': [48, 96], 'moisture_rtol': None}

    def _q1_full_tolerance__sha(path):
        return _q1_full_tolerance__hashlib.sha256(_q1_full_tolerance__Path(path).read_bytes()).hexdigest()

    def _q1_full_tolerance__readj(path):
        return _q1_full_tolerance__json.loads(_q1_full_tolerance__Path(path).read_text(encoding='utf-8'))

    def _q1_full_tolerance__writej(path, data):
        path = _q1_full_tolerance__Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_q1_full_tolerance__json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def _q1_full_tolerance__protected_sources(inputs, config):
        """Bind actual merged files, data and the imported SciPy implementations."""
        identity = _q1_full_tolerance__source_identity()
        if identity.get('schema') != 'readable-five-file-v1':
            raise RuntimeError('This comparison requires the readable five-file source identity')
        return dict(identity, input_sha256={'model_config': _q1_full_tolerance__sha(config), 'attachment1': _q1_full_tolerance__sha(_q1_full_tolerance__Path(inputs) / '附件1.xlsx'), 'attachment2': _q1_full_tolerance__sha(_q1_full_tolerance__Path(inputs) / '附件2.xlsx')}, scipy_source_sha256={name: _q1_full_tolerance__sha(_q1_full_tolerance__inspect.getsourcefile(obj)) for name, obj in (('scipy_Radau', _q1_full_tolerance__Radau), ('scipy_solve_ivp', _q1_full_tolerance__scipy_solve_ivp), ('scipy_OdeSolution', _q1_full_tolerance__OdeSolution))})

    def _q1_full_tolerance__observe_progress(main_thread_id):
        """Read scalar progress without wrapping or modifying any solver function."""
        result = {}
        frame = _q1_full_tolerance__sys._current_frames().get(main_thread_id)
        try:
            while frame is not None:
                if frame.f_code.co_name == 'solve_ivp':
                    solver = frame.f_locals.get('solver')
                    if solver is not None:
                        result.update(physical_time_s=float(solver.t), nfev=int(solver.nfev), njev=int(solver.njev), nlu=int(solver.nlu))
                    fun = frame.f_locals.get('fun')
                    if fun is not None:
                        result['active_equation'] = getattr(fun, '__name__', 'solver_callable')
                if frame.f_code.co_name == 'solve_q1':
                    for key in ['left', 'right']:
                        if key in frame.f_locals:
                            result['environment_segment_' + key + '_s'] = float(frame.f_locals[key])
                frame = frame.f_back
        finally:
            del frame
        return result

    def _q1_full_tolerance__progress_worker(stop, main_thread_id, out, started):
        while not stop.wait(50.0):
            progress = {'status': 'running', 'elapsed_wall_s': _q1_full_tolerance__perf_counter() - started, 'observation_only_no_integrator_mutation': True, **_q1_full_tolerance__observe_progress(main_thread_id)}
            _q1_full_tolerance__writej(out / 'progress.json', progress)
            print(_q1_full_tolerance__json.dumps({'q1_full_tolerance_progress': progress}), flush=True)

    def _q1_full_tolerance__difference(a, b, times, radii):
        delta = _q1_full_tolerance__np.abs(a - b)
        i, j = _q1_full_tolerance__np.unravel_index(_q1_full_tolerance__np.argmax(delta), delta.shape)
        return {'maximum_absolute_difference': float(delta[i, j]), 'time_s': float(times[i]), 'radius_m': float(radii[j]), 'rms': float(_q1_full_tolerance__np.sqrt(_q1_full_tolerance__np.mean(delta ** 2)))}

    def _q1_full_tolerance__rounding_mismatches(a, b, celsius=False):
        if celsius:
            a = a - 273.15
            b = b - 273.15
        quantum = _q1_full_tolerance__Decimal('.0001')
        return sum((_q1_full_tolerance__Decimal(str(float(x))).quantize(quantum, rounding=_q1_full_tolerance__ROUND_HALF_UP) != _q1_full_tolerance__Decimal(str(float(y))).quantize(quantum, rounding=_q1_full_tolerance__ROUND_HALF_UP) for x, y in zip(a.ravel(), b.ravel())))

    def _q1_full_tolerance__compare(inputs, config, runs, full, out, source_hashes):
        inputs, config, runs, full, out = map(_q1_full_tolerance__Path, (inputs, config, runs, full, out))
        if source_hashes.get('schema') != 'readable-five-file-v1' or _q1_full_tolerance__protected_sources(inputs, config) != source_hashes:
            raise RuntimeError('Comparison requires matching new merged-source evidence')
        if out.resolve() == full.resolve() or out.resolve().is_relative_to(full.resolve()):
            raise ValueError('Comparison reports must be separate from the read-only full solver output')
        out.mkdir(parents=True, exist_ok=True)
        core_meta = _q1_full_tolerance__readj(full / 'metadata.json')
        primary = runs / 'q1/N10240'
        reference_meta = _q1_full_tolerance__readj(primary / 'metadata.json')
        merged_identity = {key: source_hashes[key] for key in ('schema', 'files')}
        if core_meta.get('source_identity') != merged_identity or reference_meta.get('source_identity') != merged_identity:
            raise RuntimeError('Full and baseline arrays must come from this merged implementation; historical outputs cannot be relabelled')
        paths = [primary / 'solution.npz', primary / 'grid_profiles.npz', primary / 'metadata.json', runs / 'q1/convergence.json']
        c_only = runs / 'q1/tight/solution.npz'
        if c_only.exists():
            paths.append(c_only)
        reference_hashes = {p.relative_to(runs).as_posix(): _q1_full_tolerance__sha(p) for p in paths}
        expected = {'intervals': 10240, 'method': 'Radau', 'rtol': 2e-12, 'moisture_rtol': 2e-12, 'atol_theta_K': 1e-13, 'atol_C': 2e-14, 'quadrature_orders': [48, 96], 'break_at_every_environment_knot': True}
        if any((core_meta[k] != v for k, v in expected.items())):
            raise RuntimeError('Executed settings differ from the requested full check')
        with _q1_full_tolerance__np.load(full / 'solution.npz') as a, _q1_full_tolerance__np.load(primary / 'solution.npz') as b:
            if not _q1_full_tolerance__np.array_equal(a['t_s'], b['t_s']) or not _q1_full_tolerance__np.array_equal(a['r_m'], b['r_m']):
                raise RuntimeError('Full and production output coordinates differ')
            t = a['t_s'].copy()
            r = a['r_m'].copy()
            T = a['T_K'].copy()
            C = a['C'].copy()
            main_T = b['T_K'].copy()
            main_C = b['C'].copy()
            if not _q1_full_tolerance__np.array_equal(t, _q1_full_tolerance__np.arange(1801.0)) or T.shape != (1801, 21):
                raise RuntimeError('Full first-1800-second output missing')
            delta_T = _q1_full_tolerance__difference(T, main_T, t, r)
            delta_C = _q1_full_tolerance__difference(C, main_C, t, r)
            mean_differences = {key: float(_q1_full_tolerance__np.max(abs(a[key] - b[key]))) for key in ['mean_T_K', 'mean_C']}
        with _q1_full_tolerance__np.load(full / 'grid_profiles.npz') as a, _q1_full_tolerance__np.load(primary / 'grid_profiles.npz') as b:
            if not _q1_full_tolerance__np.array_equal(a['t_s'], b['t_s']) or not _q1_full_tolerance__np.array_equal(a['r_m'], b['r_m']):
                raise RuntimeError('Full-nodal checkpoint coordinates differ')
            full_profiles = {'time_rows': len(a['t_s']), 'spatial_nodes': len(a['r_m']), 'temperature': _q1_full_tolerance__difference(a['T_K'], b['T_K'], a['t_s'], a['r_m']), 'moisture': _q1_full_tolerance__difference(a['C'], b['C'], a['t_s'], a['r_m'])}
        knots, temp, attachment = _q1_full_tolerance__read_temperature_data(inputs)
        coarse_reference = _q1_full_tolerance__reference_temperature(t, r, knots, temp, 2048)
        fine_reference = _q1_full_tolerance__reference_temperature(t, r, knots, temp, 4096)
        bessel = {'modes': [2048, 4096], 'mode_truncation_difference_K': float(_q1_full_tolerance__np.max(abs(coarse_reference - fine_reference))), 'tight_max_abs_temperature_difference_K': float(_q1_full_tolerance__np.max(abs(T - fine_reference))), 'production_max_abs_temperature_difference_K': float(_q1_full_tolerance__np.max(abs(main_T - fine_reference))), 'tight_difference_location': _q1_full_tolerance__difference(T, fine_reference, t, r), 'method': 'Existing independent continuous Robin Bessel eigenbasis, exact modal advancement under piecewise-affine forcing; same implemented 2048/4096-mode convergence reference rerun on all 1801 output times.', 'attachment': attachment, 'existing_production_convergence': _q1_full_tolerance__readj(runs / 'q1/convergence.json')}
        expected_knots = _q1_full_tolerance__np.unique(_q1_full_tolerance__np.r_[0.0, knots[(knots > 0) & (knots < 1800)], 1800.0])
        segments = core_meta['segments']
        complete_breakpoints = bool(_q1_full_tolerance__np.array_equal([v['start_s'] for v in segments], expected_knots[:-1]) and _q1_full_tolerance__np.array_equal([v['end_s'] for v in segments], expected_knots[1:]))
        residuals = _q1_full_tolerance__np.asarray([v['residuals_by_quadrature'] for v in core_meta['balance_history']])
        balance = {'quadrature_orders': core_meta['quadrature_orders'], 'complete_environment_breakpoints': complete_breakpoints, 'segment_count': len(segments), 'expected_segment_count': len(expected_knots) - 1, 'max_cumulative_temperature_residual_K': float(_q1_full_tolerance__np.max(abs(residuals[:, :, 0]))), 'max_cumulative_moisture_residual': float(_q1_full_tolerance__np.max(abs(residuals[:, :, 1]))), 'max_instant_weighted_balance_abs': core_meta['max_instant_weighted_balance_abs'], 'final_independent_quadrature_difference': core_meta['final_independent_quadrature_difference'], 'scope': 'All original segments and both original 48/96-point flux quadratures are retained; thermal balance uses the P1 heat capacity, without adding latent heat or changing physical closure.'}
        c_only_result = {'available': c_only.exists()}
        if c_only.exists():
            with _q1_full_tolerance__np.load(c_only) as d:
                if not _q1_full_tolerance__np.array_equal(t, d['t_s']) or not _q1_full_tolerance__np.array_equal(r, d['r_m']):
                    raise RuntimeError('C-only check coordinates differ')
                c_only_result.update(max_abs_moisture_difference=float(_q1_full_tolerance__np.max(abs(C - d['C']))), moisture_values_bitwise_identical=bool(_q1_full_tolerance__np.array_equal(C, d['C'])))
        comparison = {'max_abs_temperature_difference_K': delta_T['maximum_absolute_difference'], 'max_abs_moisture_difference': delta_C['maximum_absolute_difference'], 'shared_time_rows': len(t), 'spatial_nodes': len(r), 'formal_time_rows_excluding_initial': len(t) - 1, 'temperature_difference_location': delta_T, 'moisture_difference_location': delta_C, 'mean_differences': mean_differences, 'four_decimal_temperature_differences': _q1_full_tolerance__rounding_mismatches(T[1:], main_T[1:], True), 'four_decimal_moisture_differences': _q1_full_tolerance__rounding_mismatches(C[1:], main_C[1:])}
        extrema = core_meta['extrema_at_all_integer_times_all_grid_nodes']
        checks = {'full_requested_solver_settings': True, 'all_environment_segments': complete_breakpoints, 'finite_outputs': bool(_q1_full_tolerance__np.isfinite(T).all() and _q1_full_tolerance__np.isfinite(C).all()), 'nonnegative_full_nodal_sampled_moisture': extrema['minimum_C_on_grid'] >= 0, 'temperature_difference_below_1e_7_K': delta_T['maximum_absolute_difference'] < 1e-07, 'moisture_difference_below_1e_7': delta_C['maximum_absolute_difference'] < 1e-07, 'independent_Bessel_difference_below_1e_7_K': bessel['tight_max_abs_temperature_difference_K'] < 1e-07, 'Bessel_2048_4096_difference_below_1e_9_K': bessel['mode_truncation_difference_K'] < 1e-09, 'cumulative_heat_balance_below_1e_7_K': balance['max_cumulative_temperature_residual_K'] < 1e-07, 'cumulative_moisture_balance_below_1e_7': balance['max_cumulative_moisture_residual'] < 1e-07}
        if _q1_full_tolerance__protected_sources(inputs, config) != source_hashes:
            raise RuntimeError('Protected source or input changed during the full run')
        for p in paths:
            if _q1_full_tolerance__sha(p) != reference_hashes[p.relative_to(runs).as_posix()]:
                raise RuntimeError('Production reference changed during comparison')
        report = {'completed': True, 'passed': all(checks.values()), 'N': 10240, 'settings': _q1_full_tolerance__SETTINGS, 'reference_settings': {k: reference_meta[k] for k in ['intervals', 'method', 'rtol', 'moisture_rtol', 'atol_theta_K', 'atol_C', 'quadrature_orders']}, 'comparison': comparison, 'full_grid_profiles_comparison': full_profiles, 'bessel_reference': bessel, 'balance': balance, 'existing_C_only_comparison': c_only_result, 'checks': checks, 'extrema': extrema, 'source_identity': source_hashes, 'reference_sha256': reference_hashes, 'output_sha256': {p.name: _q1_full_tolerance__sha(p) for p in [full / 'solution.npz', full / 'grid_profiles.npz', full / 'metadata.json']}, 'solver_elapsed_seconds': core_meta['elapsed_seconds'], 'total_heat_steps': sum((s['T_steps'] for s in segments)), 'total_moisture_steps': sum((s['C_steps'] for s in segments)), 'total_heat_nfev': sum((s['T_nfev'] for s in segments)), 'total_moisture_nfev': sum((s['C_nfev'] for s in segments)), 'statically_merged_solve_q1': True, 'all_original_integration_and_balance_checks_retained': True, 'inputs_and_sources_unchanged': True, 'timeout_or_cost_stop_used': False, 'replaces_production_or_prior_verification': False, 'interpretation': 'Additional complete same-grid tolerance validation of both decoupled P1 fields; the existing C-only run and independent Bessel check remain. Numerical thresholds are declared engineering checks, not a certified continuous-PDE error bound or a guarantee of equal four-decimal rounding.', 'created_at_utc': _q1_full_tolerance__datetime.now(_q1_full_tolerance__timezone.utc).isoformat()}
        _q1_full_tolerance__writej(out / 'summary.json', report)
        rows = [{'t_s': float(x), 'max_abs_temperature_difference_K': float(_q1_full_tolerance__np.max(abs(T[i] - main_T[i]))), 'max_abs_moisture_difference': float(_q1_full_tolerance__np.max(abs(C[i] - main_C[i]))), 'max_abs_temperature_Bessel_difference_K': float(_q1_full_tolerance__np.max(abs(T[i] - fine_reference[i])))} for i, x in enumerate(t)]
        table = out / 'q1_full_tolerance.csv'
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open('w', encoding='utf-8-sig', newline='') as f:
            writer = _q1_full_tolerance__csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        text = f"# 问题1温度与水分同时收紧100倍的完整复核\n\n实际调用静态合并后的solve_q1，N=10240，Radau rtol=2e-12，温升绝对容差1e-13 K，水分绝对容差2e-14。全部{len(segments)}个环境段和48/96阶双通量积分平衡均保留。运行{core_meta['elapsed_seconds']:.2f}秒。\n\n全1801个每秒时刻、21个半径对主算例最大温差{delta_T['maximum_absolute_difference']:.12g} K，水分差{delta_C['maximum_absolute_difference']:.12g}。对独立4096项Bessel参考最大温差{bessel['tight_max_abs_temperature_difference_K']:.12g} K；2048到4096项差{bessel['mode_truncation_difference_K']:.12g} K。\n\n全部门限通过={report['passed']}。没有因耗时停止、削减平衡检查、替换积分器、放宽容差或修改模型。该新增证据不替换任何已有主算例和验证，也不作为连续PDE的严格误差证书。\n"
        (out / 'q1_full_tolerance.md').write_text(text, encoding='utf-8')
        print(_q1_full_tolerance__json.dumps({'q1_full_tolerance_completed': True, 'passed': report['passed'], 'comparison': comparison, 'bessel_max_abs_K': bessel['tight_max_abs_temperature_difference_K']}), flush=True)
        if not report['passed']:
            raise RuntimeError('The full P1 tolerance check failed; inspect saved evidence')
        return report

    def _q1_full_tolerance__main():
        parser = _q1_full_tolerance__argparse.ArgumentParser(description=_q1_full_tolerance____doc__)
        for name in ('inputs', 'config', 'runs', 'full', 'out'):
            parser.add_argument('--' + name, type=_q1_full_tolerance__Path, required=True)
        args = parser.parse_args()
        _q1_full_tolerance__compare(args.inputs, args.config, args.runs, args.full, args.out, _q1_full_tolerance__protected_sources(args.inputs, args.config))

    # ===== extensions/q1/q1_disk_history_frozen.py: readable implementation =====
    _q1_disk_history_frozen____doc__ = 'Preserve Q1 Radau arithmetic while placing accepted histories on disk.\n\nOnly the storage around unmodified SciPy Radau and RadauDenseOutput changes.\nThe merged solve_q1 function retains all tolerances, every environment\nbreakpoint, all outputs and both 48/96-point balance integrations. Source\nidentity belongs to the readable five-file implementation, not the old file.'
    import os as _q1_disk_history_frozen__os
    import sys as _q1_disk_history_frozen__sys
    for _q1_disk_history_frozen__key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _q1_disk_history_frozen__os.environ[_q1_disk_history_frozen__key] = '1'
    _q1_disk_history_frozen__sys.dont_write_bytecode = True
    from pathlib import Path as _q1_disk_history_frozen__Path
    from types import SimpleNamespace as _q1_disk_history_frozen__SimpleNamespace
    from datetime import datetime as _q1_disk_history_frozen__datetime, timezone as _q1_disk_history_frozen__timezone
    import argparse as _q1_disk_history_frozen__argparse
    import hashlib as _q1_disk_history_frozen__hashlib
    import json as _q1_disk_history_frozen__json
    import time as _q1_disk_history_frozen__time
    import traceback as _q1_disk_history_frozen__traceback
    import ctypes as _q1_disk_history_frozen__ctypes
    import shutil as _q1_disk_history_frozen__shutil
    import numpy as _q1_disk_history_frozen__np
    import scipy as _q1_disk_history_frozen__scipy
    from scipy.integrate import Radau as _q1_disk_history_frozen__Radau, solve_ivp as _q1_disk_history_frozen__scipy_solve_ivp
    from scipy.integrate._ivp.radau import RadauDenseOutput as _q1_disk_history_frozen__RadauDenseOutput
    from scipy.integrate._ivp.common import OdeSolution as _q1_disk_history_frozen__OdeSolution
    _q1_disk_history_frozen__SETTINGS = {'intervals': 10240, 'method': 'Radau', 'rtol': 2e-12, 'atol_T': 1e-13, 'atol_C': 2e-14, 'quadrature_orders': (48, 96), 'moisture_rtol': None}

    def _q1_disk_history_frozen__sha(path):
        return _q1_disk_history_frozen__hashlib.sha256(_q1_disk_history_frozen__Path(path).read_bytes()).hexdigest()

    def _q1_disk_history_frozen__write(path, value):
        path = _q1_disk_history_frozen__Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_q1_disk_history_frozen__json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def _q1_disk_history_frozen__array_sha(a):
        return _q1_disk_history_frozen__hashlib.sha256(_q1_disk_history_frozen__np.asarray(a, dtype='<f8').tobytes(order='C')).hexdigest()

    def _q1_disk_history_frozen__wait_for_resources(out, storage):
        """Queue the unchanged large check; resource waiting never changes precision."""
        if _q1_disk_history_frozen__os.name != 'nt':
            raise RuntimeError('This recovery resource gate currently requires Windows')

        class MemoryStatus(_q1_disk_history_frozen__ctypes.Structure):
            _fields_ = [('length', _q1_disk_history_frozen__ctypes.c_ulong), ('load', _q1_disk_history_frozen__ctypes.c_ulong), ('total_physical', _q1_disk_history_frozen__ctypes.c_ulonglong), ('available_physical', _q1_disk_history_frozen__ctypes.c_ulonglong), ('total_page_file', _q1_disk_history_frozen__ctypes.c_ulonglong), ('available_page_file', _q1_disk_history_frozen__ctypes.c_ulonglong), ('total_virtual', _q1_disk_history_frozen__ctypes.c_ulonglong), ('available_virtual', _q1_disk_history_frozen__ctypes.c_ulonglong), ('available_extended_virtual', _q1_disk_history_frozen__ctypes.c_ulonglong)]
        while True:
            memory = MemoryStatus()
            memory.length = _q1_disk_history_frozen__ctypes.sizeof(memory)
            if not _q1_disk_history_frozen__ctypes.windll.kernel32.GlobalMemoryStatusEx(_q1_disk_history_frozen__ctypes.byref(memory)):
                raise _q1_disk_history_frozen__ctypes.WinError()
            disk_free = _q1_disk_history_frozen__shutil.disk_usage(storage).free
            ready = memory.available_page_file >= 8 * 1024 ** 3 and memory.available_physical >= 4 * 1024 ** 3 and (disk_free >= 16 * 1024 ** 3)
            record = {'status': 'resources_ready' if ready else 'waiting_for_resources', 'available_commit_bytes': memory.available_page_file, 'available_physical_bytes': memory.available_physical, 'scratch_disk_free_bytes': disk_free, 'required_commit_GiB': 8, 'required_physical_GiB': 4, 'required_scratch_GiB': 16, 'precision_and_mesh_unchanged': True}
            _q1_disk_history_frozen__write(out / 'queue_status.json', record)
            print(_q1_disk_history_frozen__json.dumps(record), flush=True)
            if ready:
                return record
            _q1_disk_history_frozen__time.sleep(45.0)

    class _q1_disk_history_frozen__DiskStates:
        """The original (nodes, accepted-times) y interface without retaining it."""

        def __init__(self, path, count, n):
            self.path = path
            self.shape = (n, count)
            self.count = count
            self.n = n

        def __getitem__(self, key):
            mapped = _q1_disk_history_frozen__np.memmap(self.path, dtype='<f8', mode='r', shape=(self.count, self.n))
            try:
                return _q1_disk_history_frozen__np.array(mapped.T[key], copy=True)
            finally:
                mapped._mmap.close()

    class _q1_disk_history_frozen__DiskInterpolant:

        def __init__(self, path, index, n, left, right):
            self.path = path
            self.index = index
            self.n = n
            self.t_old = left
            self.t = right

        def __call__(self, t):
            with self.path.open('rb') as stream:
                stream.seek(self.index * 4 * self.n * 8)
                record = _q1_disk_history_frozen__np.fromfile(stream, dtype='<f8', count=4 * self.n)
            if len(record) != 4 * self.n:
                raise IOError('Incomplete dense-history record')
            original = _q1_disk_history_frozen__RadauDenseOutput(self.t_old, self.t, record[:self.n], record[self.n:].reshape(self.n, 3))
            return original(t)

    class _q1_disk_history_frozen__DiskRadau:

        def __init__(self, scratch, progress):
            self.scratch = _q1_disk_history_frozen__Path(scratch).resolve()
            self.scratch.mkdir(parents=True, exist_ok=False)
            self.progress = _q1_disk_history_frozen__Path(progress)
            self.files = []
            self.records = []
            self.started = _q1_disk_history_frozen__time.perf_counter()
            self.next_progress = 0.0
            self.maximum_live_bytes = 0

        def cleanup(self):
            for path in self.files:
                if path.resolve().parent != self.scratch:
                    raise RuntimeError('Scratch path escaped its owner')
                path.unlink()
            self.files.clear()

        def __call__(self, fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options):
            if method != 'Radau' or t_eval is not None or events is not None or (args is not None) or (not dense_output):
                raise ValueError('This storage adapter supports exactly the Q1 Radau/dense/no-events call pattern')
            if fun.__name__ == 'heat_rhs':
                self.cleanup()
            left, right = map(float, t_span)
            n = len(y0)
            index = len(self.records)
            states_path = self.scratch / f'{index:03d}_states.bin'
            dense_path = self.scratch / f'{index:03d}_dense.bin'
            self.files.extend([states_path, dense_path])
            state_hash = _q1_disk_history_frozen__hashlib.sha256()
            dense_hash = _q1_disk_history_frozen__hashlib.sha256()
            solver = _q1_disk_history_frozen__Radau(fun, left, y0, right, vectorized=vectorized, **options)
            times = [left]
            interpolants = []
            call_started = _q1_disk_history_frozen__time.perf_counter()
            with states_path.open('xb') as state_stream, dense_path.open('xb') as dense_stream:
                raw = _q1_disk_history_frozen__np.asarray(y0, dtype='<f8').tobytes()
                state_stream.write(raw)
                state_hash.update(raw)
                while solver.status == 'running':
                    message = solver.step()
                    if solver.status == 'failed':
                        raise RuntimeError(message)
                    dense = solver.dense_output()
                    if dense.Q.shape != (n, 3):
                        raise RuntimeError('Unexpected Radau interpolation schema')
                    raw = _q1_disk_history_frozen__np.asarray(solver.y, dtype='<f8').tobytes()
                    state_stream.write(raw)
                    state_hash.update(raw)
                    for a in (dense.y_old, dense.Q):
                        raw = _q1_disk_history_frozen__np.asarray(a, dtype='<f8').tobytes(order='C')
                        dense_stream.write(raw)
                        dense_hash.update(raw)
                    interpolants.append(_q1_disk_history_frozen__DiskInterpolant(dense_path, len(interpolants), n, float(solver.t_old), float(solver.t)))
                    times.append(float(solver.t))
                    elapsed = _q1_disk_history_frozen__time.perf_counter() - self.started
                    if elapsed >= self.next_progress:
                        record = {'status': 'running', 'equation': fun.__name__, 't_s': float(solver.t), 'segment_s': [left, right], 'segment_steps': len(times) - 1, 'step_size_s': float(solver.step_size), 'nfev': solver.nfev, 'nlu': solver.nlu, 'elapsed_s': elapsed, 'unchanged_integrator': 'scipy.integrate.Radau', 'history_storage_only': 'disk'}
                        _q1_disk_history_frozen__write(self.progress, record)
                        print(_q1_disk_history_frozen__json.dumps(record), flush=True)
                        self.next_progress = elapsed + 45.0
            times = _q1_disk_history_frozen__np.asarray(times)
            live_bytes = sum((p.stat().st_size for p in self.files))
            self.maximum_live_bytes = max(self.maximum_live_bytes, live_bytes)
            self.records.append({'equation': fun.__name__, 'left_s': left, 'right_s': right, 'nodes': n, 'steps': len(times) - 1, 'nfev': solver.nfev, 'njev': solver.njev, 'nlu': solver.nlu, 'accepted_times_sha256': _q1_disk_history_frozen__array_sha(times), 'accepted_states_time_major_sha256': state_hash.hexdigest(), 'dense_y_old_then_Q_sha256': dense_hash.hexdigest(), 'elapsed_s': _q1_disk_history_frozen__time.perf_counter() - call_started, 'states_bytes': states_path.stat().st_size, 'dense_bytes': dense_path.stat().st_size})
            return _q1_disk_history_frozen__SimpleNamespace(t=times, y=_q1_disk_history_frozen__DiskStates(states_path, len(times), n), sol=_q1_disk_history_frozen__OdeSolution(times, interpolants, alt_segment=False), success=True, status=0, message='The solver successfully reached the end of the integration interval.', nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu)

    def _q1_disk_history_frozen__reference_capture(records):

        def wrapped(fun, t_span, y0, **options):
            result = _q1_disk_history_frozen__scipy_solve_ivp(fun, t_span, y0, **options)
            dense_hash = _q1_disk_history_frozen__hashlib.sha256()
            for dense in result.sol.interpolants:
                for a in (dense.y_old, dense.Q):
                    dense_hash.update(_q1_disk_history_frozen__np.asarray(a, dtype='<f8').tobytes(order='C'))
            records.append({'equation': fun.__name__, 'left_s': float(t_span[0]), 'right_s': float(t_span[1]), 'nodes': len(y0), 'steps': len(result.t) - 1, 'nfev': result.nfev, 'njev': result.njev, 'nlu': result.nlu, 'accepted_times_sha256': _q1_disk_history_frozen__array_sha(result.t), 'accepted_states_time_major_sha256': _q1_disk_history_frozen__array_sha(result.y.T), 'dense_y_old_then_Q_sha256': dense_hash.hexdigest()})
            return result
        return wrapped

    # ===== extensions/q1/portable_q1_disk.py: readable implementation =====
    _portable_q1_disk____doc__ = 'Portable storage-only Q1 recovery; the full N10240 precision is fixed.\n\nValidate all 30 actual input intervals at N160 before the full run. This entry\nuses the statically merged DiskRadau adapter around SciPy Radau. New\nequivalence evidence binds the actual five delivery files.'
    import os as _portable_q1_disk__os
    import sys as _portable_q1_disk__sys
    for _portable_q1_disk__key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _portable_q1_disk__os.environ[_portable_q1_disk__key] = '1'
    _portable_q1_disk__sys.dont_write_bytecode = True
    from pathlib import Path as _portable_q1_disk__Path
    import argparse as _portable_q1_disk__argparse
    import inspect as _portable_q1_disk__inspect
    import json as _portable_q1_disk__json
    import traceback as _portable_q1_disk__traceback
    import numpy as _portable_q1_disk__np
    import scipy as _portable_q1_disk__scipy
    _portable_q1_disk__disk = _sections['q1_disk_history_frozen']
    _portable_q1_disk__module = _sections['solve_q1']
    _portable_q1_disk__protected_sources = _sections['q1_full_tolerance'].protected_sources

    def _portable_q1_disk__identities(args):
        return _portable_q1_disk__protected_sources(args.inputs, args.config)

    def _portable_q1_disk__invoke(module, args, out, N, adapter):
        settings = dict(_portable_q1_disk__disk.SETTINGS)
        settings['intervals'] = N
        return module.solve_q1(args.inputs, args.config, out, solve_ivp_impl=adapter, **settings)

    def _portable_q1_disk__validate(args, module, before):
        out = args.scratch / 'equivalence_N160'
        out.mkdir(parents=True, exist_ok=False)
        records = []
        reference = _portable_q1_disk__invoke(module, args, out / 'reference', 160, _portable_q1_disk__disk.reference_capture(records))
        adapter = _portable_q1_disk__disk.DiskRadau(out / 'scratch', out / 'progress.json')
        rebuilt = _portable_q1_disk__invoke(module, args, out / 'disk', 160, adapter)
        adapter.cleanup()
        calls = [{key: a[key] == b[key] for key in a} for a, b in zip(records, adapter.records)]
        fields = {}
        for name in ('solution.npz', 'grid_profiles.npz'):
            with _portable_q1_disk__np.load(out / 'reference' / name) as a, _portable_q1_disk__np.load(out / 'disk' / name) as b:
                fields[name] = {'keys_equal': set(a.files) == set(b.files), 'arrays': {key: bool(_portable_q1_disk__np.array_equal(a[key], b[key])) for key in a.files}}
        metadata = {key: reference[key] == rebuilt[key] for key in ('balance_history', 'max_instant_weighted_balance_abs', 'final_independent_quadrature_difference', 'extrema_at_all_integer_times_all_grid_nodes', 'moisture_first_step_s', 'quadrature_orders')}
        unchanged = before == _portable_q1_disk__identities(args)
        passed = len(records) == len(adapter.records) == 60 and all((all(q.values()) for q in calls)) and all((q['keys_equal'] and all(q['arrays'].values()) for q in fields.values())) and all(metadata.values()) and unchanged
        report = {'completed': True, 'passed': passed, 'N': 160, 'production_settings': _portable_q1_disk__disk.SETTINGS, 'reference_calls': records, 'disk_calls': adapter.records, 'call_checks': calls, 'output_arrays_bitwise_equal': fields, 'metadata_bitwise_equal': metadata, 'all_30_environment_segments_and_both_fields': True, 'source_identity': before, 'sources_unchanged': unchanged, 'scipy_version': _portable_q1_disk__scipy.__version__, 'scope': 'All accepted times/states/dense coefficients and all output/balance arrays; storage-only equivalence, no discretization-error claim.'}
        _portable_q1_disk__disk.write(out / 'equivalence.json', report)
        if not passed:
            raise RuntimeError('The complete N160 bitwise equivalence check failed')

    def _portable_q1_disk__run(args, module, before):
        evidence = args.scratch / 'equivalence_N160/equivalence.json'
        check = _portable_q1_disk__json.loads(evidence.read_text(encoding='utf-8'))
        if check.get('source_identity', {}).get('schema') != 'readable-five-file-v1' or not check.get('passed') or check.get('source_identity') != before:
            raise RuntimeError('Run the new N160 validation with these exact merged-source/input hashes; old frozen evidence cannot be reused')
        out = args.scratch / 'full'
        out.mkdir(parents=True, exist_ok=False)
        resource = _portable_q1_disk__disk.wait_for_resources(out, args.scratch)
        status = {'status': 'running', 'completed': False, 'passed': False, 'N': 10240, 'settings': _portable_q1_disk__disk.SETTINGS, 'source_identity': before, 'resource_gate_at_start': resource, 'equivalence_sha256': _portable_q1_disk__disk.sha(evidence), 'storage_only': True}
        _portable_q1_disk__disk.write(out / 'run_status.json', status)
        adapter = _portable_q1_disk__disk.DiskRadau(out / 'scratch', out / 'progress.json')
        try:
            _portable_q1_disk__invoke(module, args, out, 10240, adapter)
            adapter.cleanup()
            unchanged = _portable_q1_disk__identities(args) == before
            if not unchanged:
                raise RuntimeError('Source/input identity changed during the full solve')
            _portable_q1_disk__disk.write(out / 'storage_audit.json', {'source_identity': before, 'calls': adapter.records, 'maximum_live_history_disk_bytes': adapter.maximum_live_bytes, 'temporary_histories_removed': True})
            status.update(status='solver_completed', completed=True, sources_unchanged=True, solver_completed=True, comparison_completed=False, passed=None, output_sha256={name: _portable_q1_disk__disk.sha(out / name) for name in ('solution.npz', 'grid_profiles.npz', 'metadata.json')}, storage_audit_sha256=_portable_q1_disk__disk.sha(out / 'storage_audit.json'), scope='Statically merged solve_q1 with all sampling and 48/96 balance integration completed. Independent production/Bessel comparison is a separate required stage; this status does not assert its pass.')
            _portable_q1_disk__disk.write(out / 'run_status.json', status)
        except Exception as exc:
            status.update(status='failed', completed=False, passed=False, exception_type=type(exc).__name__, exception=str(exc))
            _portable_q1_disk__disk.write(out / 'run_status.json', status)
            (out / 'failure_trace.txt').write_text(_portable_q1_disk__traceback.format_exc(), encoding='utf-8')
            raise

    def _portable_q1_disk__main():
        p = _portable_q1_disk__argparse.ArgumentParser(description=_portable_q1_disk____doc__)
        for name in ('base-code', 'inputs', 'scratch'):
            p.add_argument('--' + name, type=_portable_q1_disk__Path, required=True)
        p.add_argument('--config', type=_portable_q1_disk__Path, required=True)
        p.add_argument('--mode', choices=('validate', 'run'), required=True)
        args = p.parse_args()
        args.base_code = args.base_code.resolve()
        args.inputs = args.inputs.resolve()
        args.scratch = args.scratch.resolve()
        args.scratch.mkdir(parents=True, exist_ok=True)
        args.config = args.config.resolve()
        before = _portable_q1_disk__identities(args)
        if args.mode == 'validate':
            _portable_q1_disk__validate(args, _portable_q1_disk__module, before)
        else:
            _portable_q1_disk__run(args, _portable_q1_disk__module, before)

    # ===== extensions/q1/portable_q1_compare.py: readable implementation =====
    _portable_q1_compare____doc__ = 'Compare a completed Q1 run with the readable five-file implementation.\n\nInputs and solver outputs remain read-only. Reports use a separate directory;\nno old module layout is created and no historical source identity is asserted.'
    import os as _portable_q1_compare__os
    import sys as _portable_q1_compare__sys
    for _portable_q1_compare__key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _portable_q1_compare__os.environ[_portable_q1_compare__key] = '1'
    _portable_q1_compare__sys.dont_write_bytecode = True
    from pathlib import Path as _portable_q1_compare__Path
    import argparse as _portable_q1_compare__argparse
    import hashlib as _portable_q1_compare__hashlib
    import json as _portable_q1_compare__json
    import traceback as _portable_q1_compare__traceback
    _portable_q1_compare__comparison = _sections['q1_full_tolerance']

    def _portable_q1_compare__sha(path):
        return _portable_q1_compare__hashlib.sha256(_portable_q1_compare__Path(path).read_bytes()).hexdigest()

    def _portable_q1_compare__read(path):
        return _portable_q1_compare__json.loads(_portable_q1_compare__Path(path).read_text(encoding='utf-8'))

    def _portable_q1_compare__write(path, obj):
        _portable_q1_compare__Path(path).write_text(_portable_q1_compare__json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def _portable_q1_compare__all_true(obj):
        if isinstance(obj, dict):
            return bool(obj) and all((_portable_q1_compare__all_true(v) for v in obj.values()))
        if isinstance(obj, list):
            return bool(obj) and all((_portable_q1_compare__all_true(v) for v in obj))
        return obj is True

    def _portable_q1_compare__main():
        parser = _portable_q1_compare__argparse.ArgumentParser(description=_portable_q1_compare____doc__)
        for name in ('base-code', 'inputs', 'runs', 'full', 'equivalence', 'scratch'):
            parser.add_argument('--' + name, type=_portable_q1_compare__Path, required=True)
        parser.add_argument('--config', type=_portable_q1_compare__Path, required=True)
        args = parser.parse_args()
        for name in ('base_code', 'inputs', 'runs', 'full', 'equivalence', 'scratch', 'config'):
            setattr(args, name, getattr(args, name).resolve())
        for protected in (args.full, args.runs, args.inputs, args.config.parent):
            if args.scratch == protected or args.scratch.is_relative_to(protected) or protected.is_relative_to(args.scratch):
                raise ValueError('Comparison output must be separate from read-only solver outputs and input data')
        args.scratch.mkdir(parents=True, exist_ok=False)
        status = {'solver_completed': False, 'comparison_completed': False, 'completed': False, 'passed': None, 'scope': 'New five-file source identity; original input and solver directories are read-only.'}
        _portable_q1_compare__write(args.scratch / 'run_status.json', status)
        try:
            identity = _portable_q1_compare__comparison.protected_sources(args.inputs, args.config)
            executed = _portable_q1_compare__read(args.full / 'run_status.json')
            proof = _portable_q1_compare__read(args.equivalence)
            audit = _portable_q1_compare__read(args.full / 'storage_audit.json')
            for label, record in (('full execution', executed), ('N160 equivalence', proof), ('storage audit', audit)):
                actual = record.get('source_identity')
                if not isinstance(actual, dict) or actual.get('schema') != 'readable-five-file-v1':
                    raise RuntimeError(label + ' uses historical or unsupported source evidence; rerun the new validation and solver')
                if actual != identity:
                    raise RuntimeError(label + ' does not bind the current five source files, inputs and SciPy implementation')
            if executed.get('status') not in ('solver_completed', 'complete') or not executed.get('completed'):
                raise RuntimeError('A completed full solver execution is required')
            if not proof.get('passed') or proof.get('N') != 160 or (not proof.get('sources_unchanged')):
                raise RuntimeError('A passing N160 equivalence check for this implementation is required')
            if not proof.get('all_30_environment_segments_and_both_fields'):
                raise RuntimeError('Equivalence must cover all 30 environment segments and both fields')
            checks = proof.get('call_checks', [])
            if len(checks) != 60 or not _portable_q1_compare__all_true(checks) or (not _portable_q1_compare__all_true(proof.get('output_arrays_bitwise_equal', {}))) or (not _portable_q1_compare__all_true(proof.get('metadata_bitwise_equal', {}))):
                raise RuntimeError('All 60 call comparisons and all output/metadata equivalence checks must pass')
            if len(audit.get('calls', [])) != 60:
                raise RuntimeError('Full storage audit must contain all 60 field calls')
            outputs = executed.get('output_sha256', {})
            for name in ('solution.npz', 'grid_profiles.npz', 'metadata.json'):
                if outputs.get(name) != _portable_q1_compare__sha(args.full / name):
                    raise RuntimeError('Executed output hash mismatch: ' + name)
            if executed.get('storage_audit_sha256') != _portable_q1_compare__sha(args.full / 'storage_audit.json'):
                raise RuntimeError('The full execution does not bind this storage audit')
            if executed.get('equivalence_sha256') != _portable_q1_compare__sha(args.equivalence):
                raise RuntimeError('The full execution does not bind this N160 equivalence proof')
            files = {'equivalence': args.equivalence, 'full/run_status.json': args.full / 'run_status.json', 'full/storage_audit.json': args.full / 'storage_audit.json', 'inputs/attachment1': args.inputs / '附件1.xlsx', 'inputs/attachment2': args.inputs / '附件2.xlsx', 'inputs/model_config': args.config}
            for name in ('solution.npz', 'grid_profiles.npz', 'metadata.json'):
                files['full/' + name] = args.full / name
                files['production/q1/N10240/' + name] = args.runs / 'q1/N10240' / name
            files['production/q1/convergence.json'] = args.runs / 'q1/convergence.json'
            optional = args.runs / 'q1/tight/solution.npz'
            if optional.exists():
                files['production/q1/tight/solution.npz'] = optional
            tracked = {name: _portable_q1_compare__sha(path) for name, path in files.items()}
            status.update(solver_completed=True, solver_source_bound=True, comparison_started=True, source_identity=identity)
            _portable_q1_compare__write(args.scratch / 'run_status.json', status)
            report = _portable_q1_compare__comparison.compare(args.inputs, args.config, args.runs, args.full, args.scratch, identity)
            if _portable_q1_compare__comparison.protected_sources(args.inputs, args.config) != identity:
                raise RuntimeError('Source or input identity changed during comparison')
            if any((_portable_q1_compare__sha(path) != tracked[name] for name, path in files.items())):
                raise RuntimeError('An original input, reference, or solver evidence file changed during comparison')
            report.update(solver_completed=True, comparison_completed=True, statically_merged_comparator=True, source_identity=identity, all_original_files_unchanged=True, storage_adapter_used=True, equivalence_evidence_sha256=tracked['equivalence'], storage_audit_sha256=tracked['full/storage_audit.json'], original_solver_status_sha256=tracked['full/run_status.json'], original_output_receipt_hashes_verified=True, historical_frozen_source_identity_claimed=False, source_layout='Inputs and solver results are read in place; only these reports are written.')
            _portable_q1_compare__write(args.scratch / 'summary.json', report)
            _portable_q1_compare__write(args.scratch / 'input_manifest.json', {'sha256_by_role': tracked, 'originals_unchanged': True, 'source_identity': identity, 'data_copied': False})
            status.update(comparison_completed=True, completed=True, passed=report['passed'], summary_sha256=_portable_q1_compare__sha(args.scratch / 'summary.json'))
            _portable_q1_compare__write(args.scratch / 'run_status.json', status)
            print(_portable_q1_compare__json.dumps({'portable_comparison_completed': True, 'passed': report['passed'], 'comparison': report['comparison']}), flush=True)
        except Exception as exc:
            status.update(completed=False, passed=False, exception_type=type(exc).__name__, exception=str(exc))
            _portable_q1_compare__write(args.scratch / 'run_status.json', status)
            (args.scratch / 'failure_trace.txt').write_text(_portable_q1_compare__traceback.format_exc(), encoding='utf-8')
            raise

    # ===== extensions/endpoint/endpoint_study.py: readable implementation =====
    _endpoint_study____doc__ = 'Shared endpoint output helpers; unrelated exploratory studies are excluded.'
    from pathlib import Path as _endpoint_study__Path
    import hashlib as _endpoint_study__hashlib
    import json as _endpoint_study__json

    def _endpoint_study__sha(p):
        return _endpoint_study__hashlib.sha256(_endpoint_study__Path(p).read_bytes()).hexdigest()

    def _endpoint_study__write(p, obj):
        _endpoint_study__Path(p).parent.mkdir(parents=True, exist_ok=True)
        _endpoint_study__Path(p).write_text(_endpoint_study__json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    # ===== extensions/endpoint/global_refinement.py: readable implementation =====
    _global_refinement____doc__ = 'Recorder used by matched-tolerance endpoint refinement.'
    from pathlib import Path as _global_refinement__Path
    import hashlib as _global_refinement__hashlib
    import json as _global_refinement__json
    import numpy as _global_refinement__np

    def _global_refinement__sha(p):
        return _global_refinement__hashlib.sha256(_global_refinement__Path(p).read_bytes()).hexdigest()

    def _global_refinement__write(p, o):
        _global_refinement__Path(p).write_text(_global_refinement__json.dumps(o, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    class _global_refinement__Recorder:

        def __init__(self, model, directory):
            self.model = model
            self.directory = directory
            directory.mkdir(parents=True, exist_ok=False)
            self.times = []
            self.T = []
            self.C = []
            self.R = []
            self.checkpoints = {}
            self.checkpoint_labels = {}
            self.xi = _global_refinement__np.linspace(0, 1, 21)

        def add(self, t, y, full_label=None):
            if self.times and abs(self.times[-1] - t) < 1e-09:
                return
            T, C = self.model.unpack(y)
            self.times.append(float(t))
            self.R.append(float(self.model.geometry.radius(t)))
            self.T.append(_global_refinement__np.interp(self.xi, self.model.xi, T))
            self.C.append(_global_refinement__np.interp(self.xi, self.model.xi, C))
            if full_label or abs(t / 21600 - round(t / 21600)) < 1e-12:
                self.checkpoints[float(t)] = y.copy()
                self.checkpoint_labels[float(t)] = full_label or 'six_hour'

        def flush(self, final=False):
            _global_refinement__np.savez_compressed(self.directory / 'samples.npz', t_s=_global_refinement__np.array(self.times), xi=self.xi, T_K=_global_refinement__np.array(self.T), C=_global_refinement__np.array(self.C), R_m=_global_refinement__np.array(self.R))
            if final:
                t = _global_refinement__np.array(sorted(self.checkpoints))
                y = _global_refinement__np.array([self.checkpoints[x] for x in t])
                n = self.model.nr
                _global_refinement__np.savez_compressed(self.directory / 'checkpoints.npz', t_s=t, xi=self.model.xi, T_K=y[:, :n] + self.model.temperature_reference_K, C=y[:, n:], labels=_global_refinement__np.array([self.checkpoint_labels[x] for x in t]))

    # ===== extensions/endpoint/global_refinement_local_clock.py: readable implementation =====
    _global_refinement_local_clock____doc__ = 'Global refinement with exact-equivalent per-input-segment time origins.\n\nA fine surface heat transient after t=14400 needs steps below the binary64\nspacing at that absolute time. Each BDF segment therefore starts at tau=0 and\npasses t=a+tau to the same physical RHS. No mesh, tolerance or physical term is\nrelaxed. This file is separate from, and does not alter, the original solvers.'
    import os as _global_refinement_local_clock__os
    import sys as _global_refinement_local_clock__sys
    _global_refinement_local_clock__sys.dont_write_bytecode = True
    for _global_refinement_local_clock__k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _global_refinement_local_clock__os.environ[_global_refinement_local_clock__k] = '1'
    from pathlib import Path as _global_refinement_local_clock__Path
    from types import SimpleNamespace as _global_refinement_local_clock__SimpleNamespace
    import argparse as _global_refinement_local_clock__argparse
    import json as _global_refinement_local_clock__json
    import numpy as _global_refinement_local_clock__np
    from scipy.integrate import BDF as _global_refinement_local_clock__BDF
    from scipy.integrate._ivp.ivp import solve_event_equation as _global_refinement_local_clock__solve_event_equation
    _global_refinement_local_clock__Recorder = _sections['global_refinement'].Recorder
    _global_refinement_local_clock__sha = _sections['global_refinement'].sha
    _global_refinement_local_clock__write = _sections['global_refinement'].write
    _global_refinement_local_clock__source_identity = source_identity

    def _global_refinement_local_clock__local_class(stream):

        class LocalClock(stream.StreamingBDF):

            def span(self, a, b, y, targets, detect_event=True):
                model = self.model
                s = self.s
                right = bool(a >= model.ambient.breakpoint_s and _global_refinement_local_clock__np.isfinite(model.ambient.breakpoint_s))
                first = None
                if s.limit_initial_step and a == 0:
                    T, C = model.unpack(y)
                    diff = model.properties.evaluate(C, T)[3]
                    dd = float(diff.max())
                    if dd > 0:
                        first = min(0.01, 0.1 * model.grid.minimum_spacing_m ** 2 / dd, b - a)
                kwargs = {} if first is None else {'first_step': first}
                solver = _global_refinement_local_clock__BDF(lambda tau, u: model.rhs(a + tau, u, right_of_breakpoint=right), 0.0, y, float(b - a), rtol=s.rtol, atol=s.atol_vector(model.nr), max_step=s.max_step_s, jac_sparsity=model.jacobian_sparsity(), **kwargs)
                event = stream.core._event_function(model)
                oldg = event(a, y)
                p = 0
                while p < len(targets) and targets[p] <= a + 1e-10:
                    p += 1
                step_start = self.step_count
                terminated = False
                min_local_step = float('inf')
                lost_absolute_clock_steps = 0
                while solver.status == 'running':
                    tau_left = float(solver.t)
                    message = solver.step()
                    if solver.status == 'failed':
                        raise RuntimeError(f'Local-clock BDF failed at a={a},tau={tau_left}: {message}')
                    self.step_count += 1
                    tau_right = float(solver.t)
                    min_local_step = min(min_local_step, tau_right - tau_left)
                    left = a + tau_left
                    right_t = a + tau_right
                    lost_absolute_clock_steps += int(left == right_t)
                    local_dense = solver.dense_output()
                    dense = lambda t: local_dense(_global_refinement_local_clock__np.asarray(t) - a)
                    newg = event(right_t, solver.y)
                    if detect_event and oldg >= 0 and (newg <= 0):
                        tau_root = float(_global_refinement_local_clock__solve_event_equation(lambda tau, u: event(a + tau, u), local_dense, tau_left, tau_right))
                        right_t = a + tau_root
                        y = _global_refinement_local_clock__np.asarray(local_dense(tau_root))
                        self.event_time = right_t
                        self.event_state = y.copy()
                        self.event_steps = self.step_count
                        terminated = True
                    else:
                        y = _global_refinement_local_clock__np.asarray(solver.y)
                    self.step_hash.update(_global_refinement_local_clock__np.float64(right_t).tobytes())
                    if self.step_times is not None:
                        self.step_times.append(right_t)
                    while p < len(targets) and targets[p] <= right_t + 1e-10:
                        t = float(targets[p])
                        self.recorder.add(t, _global_refinement_local_clock__np.asarray(dense(t)), None if detect_event else 'post_event_tick')
                        p += 1
                    if terminated:
                        self.recorder.add(right_t, y, 'equality_root')
                    proxy = _global_refinement_local_clock__SimpleNamespace(t=right_t, y=solver.y, step_size=solver.step_size, nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu)
                    self.progress(proxy)
                    del dense, local_dense
                    oldg = newg
                    if terminated:
                        break
                self.nfev += solver.nfev
                self.njev += solver.njev
                self.nlu += solver.nlu
                self.segments.append({'start_s': float(a), 'requested_end_s': float(b), 'actual_end_s': right_t, 'first_step_s': first, 'accepted_steps': self.step_count - step_start, 'nfev': solver.nfev, 'njev': solver.njev, 'nlu': solver.nlu, 'event': terminated, 'local_time_origin': True, 'minimum_local_step_s': min_local_step, 'steps_unresolvable_by_absolute_clock': lost_absolute_clock_steps})
                del solver
                return (y, terminated)
        return LocalClock

    def _global_refinement_local_clock__main():
        p = _global_refinement_local_clock__argparse.ArgumentParser(description=_global_refinement_local_clock____doc__)
        for k in ('inputs', 'contract', 'out'):
            p.add_argument('--' + k, type=_global_refinement_local_clock__Path, required=True)
        for k in ('solvers', 'validation'):
            p.add_argument('--' + k, type=_global_refinement_local_clock__Path, help=_global_refinement_local_clock__argparse.SUPPRESS)
        p.add_argument('--case', choices=['p23', 'q4', 'both'], default='both')
        a = p.parse_args()
        m = _sections['moving_model']
        RelativeTemperatureModel = _sections['relative_temperature'].RelativeTemperatureModel
        stream = _sections['absolute_temperature_check']
        LocalClock = _global_refinement_local_clock__local_class(stream)
        for case in ['p23', 'q4'] if a.case == 'both' else [a.case]:
            cases = [(1280, 1e-11), (2560, 1e-11), (2560, 1e-12)] if case == 'p23' else [(2560, 1e-11), (5120, 1e-11), (5120, 1e-12)]
            for N, rtol in cases:
                directory = a.out / f'{case}_N{N}_rtol{rtol:g}'
                if (directory / 'record.json').exists():
                    continue
                model = RelativeTemperatureModel.from_model(m.build_model(a.inputs, a.contract, N=N, properties='P23' if case == 'p23' else 'P4', geometry='fixed' if case == 'p23' else 'measured'))
                rec = _global_refinement_local_clock__Recorder(model, directory)
                solver = LocalClock(model, rec, progress_seconds=45)
                fac = rtol / 1e-10
                solver.s = m.IntegrationSettings('BDF', rtol, (1e-10 if case == 'p23' else 1e-09) * fac, 1e-12 * fac, 75.0 if case == 'p23' else 600.0, True)
                result = solver.integrate()
                result.update({'case': case, 'N': N, 'settings': solver.s.__dict__, 'continuous_PDE_certified': False, 'coordinate': 'normalized material radius xi', 'time_coordinate': 'tau=t-segment_start, same physical RHS and input knots', 'source_identity': _global_refinement_local_clock__source_identity(), 'input_sha256': {'model_config': _global_refinement_local_clock__sha(a.contract), 'attachment1': _global_refinement_local_clock__sha(a.inputs / '附件1.xlsx'), 'attachment2': _global_refinement_local_clock__sha(a.inputs / '附件2.xlsx')}})
                _global_refinement_local_clock__write(directory / 'record.json', result)
                print(_global_refinement_local_clock__json.dumps({'completed_case': case, 'N': N, 'rtol': rtol, 'root_s': result['event_root_s']}), flush=True)

    # ===== extensions/endpoint/finest_candidate.py: readable implementation =====
    _finest_candidate____doc__ = 'Full requested outputs from the already tested finest settings.\n\nNo accepted source is edited. The prior local-clock BDF span is inherited\nunchanged. New storage retains full-second fixed-radius P23 output, minute\nprofiles, physical moving-radius columns, and exact old/new report queries.'
    import os as _finest_candidate__os
    import sys as _finest_candidate__sys
    _finest_candidate__sys.dont_write_bytecode = True
    for _finest_candidate__k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _finest_candidate__os.environ[_finest_candidate__k] = '1'
    from pathlib import Path as _finest_candidate__Path
    from decimal import Decimal as _finest_candidate__Decimal, ROUND_CEILING as _finest_candidate__ROUND_CEILING
    from time import perf_counter as _finest_candidate__perf_counter
    import argparse as _finest_candidate__argparse
    import json as _finest_candidate__json
    import gc as _finest_candidate__gc
    import numpy as _finest_candidate__np
    from scipy.interpolate import PchipInterpolator as _finest_candidate__PchipInterpolator
    _finest_candidate__sha = _sections['global_refinement'].sha
    _finest_candidate__write = _sections['global_refinement'].write
    _finest_candidate__clock = _sections['global_refinement_local_clock']
    _finest_candidate__source_identity = source_identity

    def _finest_candidate__ticks(step, end):
        x = _finest_candidate__np.arange(step, end + 1e-08, step)
        return _finest_candidate__np.r_[x, end] if not len(x) or abs(x[-1] - end) > 1e-08 else x

    class _finest_candidate__CompleteRecorder:

        def __init__(self, model, directory, case, report, old_report):
            self.model = model
            self.directory = directory
            directory.mkdir(parents=True, exist_ok=False)
            self.case = case
            self.report = report
            self.old_report = old_report
            self.times = []
            self.T = []
            self.C = []
            self.R = []
            self.checkpoints = {}
            self.checkpoint_labels = {}
            self.internal = {}
            self.r = _finest_candidate__np.arange(21) * 0.001
            self.all_physical = []
            self.all_physical_T = []
            self.interpolation_T = 0.0
            self.interpolation_C = 0.0
            self.full_query = {}
            self.index = {}
            self.audit_rows = []

        def add(self, t, y, full_label=None):
            t = float(t)
            if t in self.index:
                return
            model = self.model
            T, C = model.unpack(y)
            R = float(model.geometry.radius(t))
            self.index[t] = len(self.times)
            self.times.append(t)
            self.R.append(R)
            self.T.append(_finest_candidate__np.interp(_finest_candidate__np.linspace(0, 1, 21), model.xi, T))
            self.C.append(_finest_candidate__np.interp(_finest_candidate__np.linspace(0, 1, 21), model.xi, C))
            valid = self.r < R
            ct = _finest_candidate__np.full(22, _finest_candidate__np.nan)
            cc = _finest_candidate__np.full(22, _finest_candidate__np.nan)
            ct[:-1][valid] = _finest_candidate__np.interp(self.r[valid] / R, model.xi, T)
            cc[:-1][valid] = _finest_candidate__np.interp(self.r[valid] / R, model.xi, C)
            ct[-1] = T[-1]
            cc[-1] = C[-1]
            self.all_physical_T.append(ct)
            self.all_physical.append(cc)
            full = full_label or abs(t / 21600 - round(t / 21600)) < 1e-12 or t in [self.report, self.old_report]
            if full:
                self.checkpoints[t] = y.copy()
                self.checkpoint_labels[t] = full_label or 'six_hour_or_exact_report'
            if t == 0 or abs(t / 60 - round(t / 60)) < 1e-12 or t in [self.report, self.old_report]:
                self.internal[t] = y.copy()
                if t <= self.report:
                    self.audit_rows.append(model.audit(t, y, right_of_breakpoint=t > model.ambient.breakpoint_s))
                    smoothT = _finest_candidate__PchipInterpolator(model.xi, T)(self.r[valid] / R)
                    smoothC = _finest_candidate__PchipInterpolator(model.xi, C)(self.r[valid] / R)
                    self.interpolation_T = max(self.interpolation_T, float(_finest_candidate__np.max(abs(smoothT - ct[:-1][valid]))))
                    self.interpolation_C = max(self.interpolation_C, float(_finest_candidate__np.max(abs(smoothC - cc[:-1][valid]))))
            if t in [self.report, self.old_report]:
                self.full_query[t] = y.copy()

        def flush(self, final=False):
            if not final:
                return
            m = self.model
            n = m.nr
            Tref = m.temperature_reference_K
            tt = _finest_candidate__np.array(self.times)
            TT = _finest_candidate__np.array(self.T)
            CC = _finest_candidate__np.array(self.C)
            RR = _finest_candidate__np.array(self.R)
            _finest_candidate__np.savez_compressed(self.directory / 'samples.npz', t_s=tt, xi=_finest_candidate__np.linspace(0, 1, 21), T_K=TT, C=CC, R_m=RR)
            t = _finest_candidate__np.array(sorted(self.checkpoints))
            y = _finest_candidate__np.array([self.checkpoints[x] for x in t])
            _finest_candidate__np.savez_compressed(self.directory / 'checkpoints.npz', t_s=t, xi=m.xi, T_K=y[:, :n] + Tref, C=y[:, n:], labels=_finest_candidate__np.array([self.checkpoint_labels[x] for x in t]))
            ti = _finest_candidate__np.r_[0.0, _finest_candidate__ticks(60.0, self.report)]
            iy = _finest_candidate__np.array([self.internal[float(t)] for t in ti])
            w = 2 * m.grid.volume_weights_m2 / m.grid.radius_m ** 2
            _finest_candidate__np.savez_compressed(self.directory / 'internal.npz', t_s=ti, xi=m.xi, R_m=m.geometry.radius(ti), T_K=iy[:, :n] + Tref, C=iy[:, n:], mean_C=iy[:, n:] @ w)

            def sample(ts, rr, moving=False):
                ys = _finest_candidate__np.array([self.internal[float(t)] if float(t) in self.internal else self.checkpoints[float(t)] for t in ts])
                T = ys[:, :n] + Tref
                C = ys[:, n:]
                radii = _finest_candidate__np.asarray(m.geometry.radius(ts))
                valid = rr[None, :] < radii[:, None]
                ta = _finest_candidate__np.full(valid.shape, _finest_candidate__np.nan)
                ca = _finest_candidate__np.full(valid.shape, _finest_candidate__np.nan)
                for j, R in enumerate(radii):
                    mask = valid[j] if moving else _finest_candidate__np.ones(len(rr), dtype=bool)
                    ta[j, mask] = _finest_candidate__np.interp(rr[mask] / R, m.xi, T[j])
                    ca[j, mask] = _finest_candidate__np.interp(rr[mask] / R, m.xi, C[j])
                return {'t_s': ts, 'r_sample_m': rr, 'R_m': radii, 'valid_interior': valid, 'T_K_sample': ta, 'C_sample': ca, 'T_K_surface': T[:, -1], 'C_surface': C[:, -1]}
            paper_t = _finest_candidate__np.r_[_finest_candidate__np.arange(21600.0, self.report, 21600.0), self.report]
            if self.case == 'p23':
                data = self.directory / 'data'
                data.mkdir()
                chunkdir = data / 'q2_fixed_grid_1s_chunks'
                chunkdir.mkdir()
                ts = _finest_candidate__ticks(1.0, self.report)
                lookup = _finest_candidate__np.array([self.index[float(t)] for t in ts])
                chunks = []
                physical_T = _finest_candidate__np.array(self.all_physical_T)
                physical_C = _finest_candidate__np.array(self.all_physical)
                t2 = _finest_candidate__np.column_stack([physical_T[lookup, :20], physical_T[lookup, -1]])
                c2 = _finest_candidate__np.column_stack([physical_C[lookup, :20], physical_C[lookup, -1]])
                for j, start in enumerate(range(0, len(ts), 10800)):
                    p = chunkdir / f'q2_1s_{j:03d}.npz'
                    _finest_candidate__np.savez_compressed(p, t_s=ts[start:start + 10800], r_m=self.r, T_K=t2[start:start + 10800], C=c2[start:start + 10800])
                    chunks.append({'file': p.relative_to(self.directory).as_posix(), 'rows': len(ts[start:start + 10800]), 'sha256': _finest_candidate__sha(p)})
                _finest_candidate__write(data / 'q2_fixed_grid_1s_index.json', {'chunks': chunks, 'row_count': len(ts), 'final_time_s': self.report})
                ts = _finest_candidate__ticks(60.0, self.report)
                sm = sample(ts, self.r)
                _finest_candidate__np.savez_compressed(data / 'q3_fixed_grid_60s.npz', t_s=ts, r_m=self.r, T_K=sm['T_K_sample'], C=sm['C_sample'])
                p1 = sample(_finest_candidate__np.arange(1800.0, 10801.0, 1800.0), _finest_candidate__np.arange(5) * 0.005)
                p2 = sample(paper_t, _finest_candidate__np.arange(5) * 0.005)
                for name, s in [('paper_first3h.npz', p1), ('paper_full.npz', p2)]:
                    _finest_candidate__np.savez_compressed(self.directory / name, t_s=s['t_s'], T_K=s['T_K_sample'], C=s['C_sample'])
            else:
                ts = _finest_candidate__ticks(60.0, self.report)
                _finest_candidate__np.savez_compressed(self.directory / 'result4_candidate_data.npz', **sample(ts, _finest_candidate__np.arange(20) * 0.001, True))
                _finest_candidate__np.savez_compressed(self.directory / 'physical_21_fixed_plus_surface.npz', **sample(ts, self.r, True))
                _finest_candidate__np.savez_compressed(self.directory / 'paper_full.npz', **sample(paper_t, _finest_candidate__np.arange(4) * 0.005, True))
                _finest_candidate__write(self.directory / 'interpolation_check.json', {'linear_vs_pchip_C_max': self.interpolation_C, 'linear_vs_pchip_T_max_K': self.interpolation_T, 'formal_method': 'linear', 'only_actual_interior_positions_compared': True})
            full_t = _finest_candidate__np.array([self.report, self.old_report])
            full_y = _finest_candidate__np.array([self.full_query[float(t)] for t in full_t])
            _finest_candidate__np.savez_compressed(self.directory / 'exact_report_fields.npz', t_s=full_t, xi=m.xi, T_K=full_y[:, :n] + Tref, C=full_y[:, n:], R_m=m.geometry.radius(full_t))

    def _finest_candidate__complete_class(stream):
        Parent = _finest_candidate__clock.local_class(stream)

        class Complete(Parent):

            def integrate_complete(self, old_report, margin, horizon=864000.0):
                model = self.model
                y = model.initial_state(301.15, 2.55)
                self.recorder.add(0.0, y, 'initial')
                step = 1.0 if self.recorder.case == 'p23' else 60.0
                targets = _finest_candidate__np.unique(_finest_candidate__np.r_[_finest_candidate__np.arange(1.0, 10801.0), _finest_candidate__np.arange(step, horizon + 1, step)])
                points = model.ambient.breakpoints(0, horizon)
                self.progress(force=True, stage='starting')
                for a, b in zip(points[:-1], points[1:]):
                    selected = targets[(targets > a) & (targets <= b)]
                    y, event = self.span(float(a), float(b), y, selected, True)
                    if event:
                        break
                if self.event_time is None:
                    raise RuntimeError('No threshold crossing before horizon')
                root = self.event_time
                minimum = _finest_candidate__Decimal(str(root)) + _finest_candidate__Decimal(str(margin))
                index = (minimum / _finest_candidate__Decimal('.36')).to_integral_value(rounding=_finest_candidate__ROUND_CEILING)
                report = float(index * _finest_candidate__Decimal('.36'))
                if report <= root:
                    report = float((index + 1) * _finest_candidate__Decimal('.36'))
                assert report == self.recorder.report
                end = root + 2.0
                assert old_report <= end and report <= end
                old_ticks = _finest_candidate__np.arange(_finest_candidate__np.ceil(root / 0.36), _finest_candidate__np.floor(end / 0.36) + 1) * 0.36
                extra = _finest_candidate__np.unique(_finest_candidate__np.r_[targets[(targets > root) & (targets <= end)], old_ticks, report, old_report, end])
                points = model.ambient.breakpoints(root, end)
                for a, b in zip(points[:-1], points[1:]):
                    y, _ = self.span(float(a), float(b), y, extra[(extra > a) & (extra <= b)], False)
                self.recorder.checkpoints[end] = y.copy()
                self.recorder.checkpoint_labels[end] = 'post_event_final'
                self.recorder.flush(final=True)
                self.progress(force=True, stage='complete')
                return {'event_root_s': root, 'event_root_h': root / 3600, 'post_event_final_s': end, 'event_max_C': float(self.event_state[model.nr:].max()), 'accepted_steps_to_event': self.event_steps, 'accepted_steps_including_post': self.step_count, 'accepted_step_time_sha256': self.step_hash.hexdigest(), 'nfev': self.nfev, 'njev': self.njev, 'nlu': self.nlu, 'segments': self.segments, 'wall_seconds': _finest_candidate__perf_counter() - self.started, 'memory_at_completion': stream.memory_status()}
        return Complete

    def _finest_candidate__main():
        p = _finest_candidate__argparse.ArgumentParser(description=_finest_candidate____doc__)
        for k in ('inputs', 'contract', 'base-code', 'refinements', 'production', 'out'):
            p.add_argument('--' + k, type=_finest_candidate__Path, required=True)
        p.add_argument('--case', choices=['p23', 'q4'], required=True)
        a = p.parse_args()
        m = _sections['moving_model']
        RelativeTemperatureModel = _sections['relative_temperature'].RelativeTemperatureModel
        stream = _sections['absolute_temperature_check']
        case = a.case
        N = 2560 if case == 'p23' else 5120
        ds = [a.refinements / f'{case}_N{N // 2}_rtol1e-11', a.refinements / f'{case}_N{N}_rtol1e-11', a.refinements / f'{case}_N{N}_rtol1e-12']
        refs = [_finest_candidate__json.loads((d / 'record.json').read_text(encoding='utf-8')) for d in ds]
        old = _finest_candidate__json.loads((a.production / case / 'endpoint.json').read_text(encoding='utf-8'))
        space = abs(refs[1]['event_root_s'] - refs[0]['event_root_s'])
        temporal = abs(refs[2]['event_root_s'] - refs[1]['event_root_s'])
        margin = space + temporal
        root = refs[-1]['event_root_s']
        tickindex = ((_finest_candidate__Decimal(str(root)) + _finest_candidate__Decimal(str(margin))) / _finest_candidate__Decimal('.36')).to_integral_value(rounding=_finest_candidate__ROUND_CEILING)
        report = float(tickindex * _finest_candidate__Decimal('.36'))
        model = RelativeTemperatureModel.from_model(m.build_model(a.inputs, a.contract, N=N, properties='P23' if case == 'p23' else 'P4', geometry='fixed' if case == 'p23' else 'measured'))
        dest = a.out / case
        rec = _finest_candidate__CompleteRecorder(model, dest, case, report, float(old['reported_time_s']))
        solver = _finest_candidate__complete_class(stream)(model, rec, progress_seconds=45)
        solver.s = m.IntegrationSettings(**refs[-1]['settings'])
        result = solver.integrate_complete(float(old['reported_time_s']), margin)
        same_steps = result['accepted_step_time_sha256'] == refs[-1]['accepted_step_time_sha256']
        same_root = result['event_root_s'] == root
        assert same_steps and same_root and (result['nfev'] == refs[-1]['nfev'])
        with _finest_candidate__np.load(ds[-1] / 'samples.npz') as earlier, _finest_candidate__np.load(dest / 'samples.npz') as current:
            t, ie, ic = _finest_candidate__np.intersect1d(earlier['t_s'], current['t_s'], return_indices=True)
            dt = float(_finest_candidate__np.max(abs(earlier['T_K'][ie] - current['T_K'][ic])))
            dc = float(_finest_candidate__np.max(abs(earlier['C'][ie] - current['C'][ic])))
        assert dt == 0 and dc == 0
        C = model.unpack(rec.full_query[report])[1]
        C_old = model.unpack(rec.full_query[float(old['reported_time_s'])])[1]
        endpoint = {'equality_root_s': root, 'equality_root_h': root / 3600, 'spatial_root_difference_s': space, 'time_tolerance_root_difference_s': temporal, 'engineering_margin_s': margin, 'reported_time_s': report, 'reported_time_h': report / 3600, 'reported_max_C': float(C.max()), 'strict_reconstructed_field_dry': bool(C.max() < 0.15), 'continuous_space_maximum': 'exact maximum of the piecewise-linear radial reconstruction occurs at a node', 'continuous_PDE_error_certified': False, 'interpretation': 'Finest tight-run candidate; complete adjacent matched-tolerance spatial difference plus finest-grid time difference gives empirical allowance, followed by the next dry-side 0.0001h tick.'}
        assert endpoint['strict_reconstructed_field_dry']
        convergence = {'reference_runs': [{'directory': d.name, 'record_sha256': _finest_candidate__sha(d / 'record.json'), **r} for d, r in zip(ds, refs)], 'time_tolerance_root_difference_s': temporal, 'spatial_root_difference_s': space, 'candidate_uses_tighter_run': True, 'replayed_accepted_step_times_bitwise_identical': same_steps, 'replayed_root_bitwise_identical': same_root, 'shared_earlier_sample_rows': len(t), 'shared_T_difference_K': dt, 'shared_C_difference': dc}
        source = _finest_candidate__source_identity()
        _finest_candidate__write(dest / 'endpoint.json', endpoint)
        _finest_candidate__write(dest / 'convergence.json', convergence)
        _finest_candidate__write(dest / 'record.json', result)
        _finest_candidate__write(dest / 'old_report_comparison.json', {'time_s': old['reported_time_s'], 'exact_time_query': True, 'unrounded_max_C': float(C_old.max()), 'strictly_dry': bool(C_old.max() < 0.15), 'candidate_minus_original_report_s': report - old['reported_time_s'], 'original_endpoint_sha256': _finest_candidate__sha(a.production / case / 'endpoint.json')})
        _finest_candidate__write(dest / 'metadata.json', {'question': case, 'N': N, 'settings': solver.s.__dict__, 'endpoint': endpoint, 'source_identity': source, 'input_sha256': {'model_config': _finest_candidate__sha(a.contract), 'attachment1': _finest_candidate__sha(a.inputs / '附件1.xlsx'), 'attachment2': _finest_candidate__sha(a.inputs / '附件2.xlsx')}, 'generator': 'Merged finest_candidate with the original local-clock BDF span and observation order.', 'new_time_integrated_balance_not_yet_computed': True, 'thermal_state': 'theta=T-Tref; Arrhenius uses absolute kelvin'})
        print(_finest_candidate__json.dumps({'case': case, 'completed': True, 'endpoint': endpoint, 'replay_bitwise_equal': same_root and same_steps}), flush=True)

    # ===== extensions/endpoint/finest_balance_audit.py: readable implementation =====
    _finest_balance_audit____doc__ = 'Independent streaming 8/16-Gauss audit of the finest candidate trajectory.\n\nThe original BDF step is called unchanged. A weak-reference observer reads its\ncurrent dense polynomial, integrates the physical fluxes, and differentiates\nthe polynomial coefficients independently of the RHS. No step, tolerance,\nJacobian, interpolation coefficient or solver state is changed.'
    import os as _finest_balance_audit__os
    import sys as _finest_balance_audit__sys
    _finest_balance_audit__sys.dont_write_bytecode = True
    for _finest_balance_audit__k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _finest_balance_audit__os.environ[_finest_balance_audit__k] = '1'
    from pathlib import Path as _finest_balance_audit__Path
    import argparse as _finest_balance_audit__argparse
    import json as _finest_balance_audit__json
    import weakref as _finest_balance_audit__weakref
    import numpy as _finest_balance_audit__np
    from numpy.polynomial.legendre import leggauss as _finest_balance_audit__leggauss
    _finest_balance_audit__candidate = _sections['finest_candidate']
    _finest_balance_audit__clock = _sections['global_refinement_local_clock']
    _finest_balance_audit__sha = _sections['endpoint_study'].sha
    _finest_balance_audit__write = _sections['endpoint_study'].write
    _finest_balance_audit__source_file = source_file
    _finest_balance_audit__source_identity = source_identity

    class _finest_balance_audit__Observer:

        def __init__(self, root, report):
            self.root = root
            self.report = report
            self.context = None
            self.rows = {}
            for order in [8, 16]:
                x, w = _finest_balance_audit__leggauss(order)
                self.rows[order] = {'nodes': x, 'weights': w, 'moisture_flux': 0.0, 'heat_flux': 0.0, 'heat_abs_flux': 0.0, 'heat_store': 0.0, 'intervals': 0}

        def observe(self, solver, left, context):
            model = context['model']
            a = context['start']
            cut = self.root if context['detect'] else self.report
            right = min(float(solver.t), float(cut - a))
            if right <= left:
                return
            poly = solver.dense_output()
            n = model.nr
            R0 = model.grid.radius_m
            weight = 2 * model.grid.volume_weights_m2 / R0 ** 2
            for order, row in self.rows.items():
                width = (right - left) / 2
                tau = (left + right) / 2 + width * row['nodes']
                t = a + tau
                y = poly(tau)
                theta = y[:n].T
                C = y[n:].T
                T = theta + model.temperature_reference_K
                R = _finest_balance_audit__np.asarray(model.geometry.radius(t))
                env = model.ambient.values(t, right_of_breakpoint=context['right_side'])
                multiplier = _finest_balance_audit__np.asarray(model.moisture_boundary_multiplier(t, R, R0))
                flux_c = 2 * model.h_m_m_s * multiplier / R * (C[:, -1] - env.moisture_eq)
                J = (R / R0) ** 2
                flux_h = J * 2 * model.h_W_m2_K / R * (theta[:, -1] - (env.temperature_K - model.temperature_reference_K))
                products = _finest_balance_audit__np.ones(order)
                derivatives = _finest_balance_audit__np.zeros(order)
                dy = _finest_balance_audit__np.zeros_like(y)
                for degree in range(poly.order):
                    xx = (tau - poly.t_shift[degree]) / poly.denom[degree]
                    derivatives = derivatives * xx + products / poly.denom[degree]
                    products = products * xx
                    dy += poly.D[degree + 1, :, None] * derivatives[None, :]
                rho, cp, _, _ = model.properties.evaluate(C, T)
                thermal_store = J * (rho * cp * dy[:n].T @ weight)
                row['moisture_flux'] += width * (row['weights'] @ flux_c)
                row['heat_flux'] += width * (row['weights'] @ flux_h)
                row['heat_abs_flux'] += width * (row['weights'] @ _finest_balance_audit__np.abs(flux_h))
                row['heat_store'] += width * (row['weights'] @ thermal_store)
                row['intervals'] += 1
            del poly

    def _finest_balance_audit__main():
        p = _finest_balance_audit__argparse.ArgumentParser(description=_finest_balance_audit____doc__)
        for k in ('inputs', 'contract', 'base-code', 'refinements', 'production', 'candidate', 'out'):
            p.add_argument('--' + k, type=_finest_balance_audit__Path, required=True)
        p.add_argument('--case', choices=['p23', 'q4'], required=True)
        a = p.parse_args()
        source = a.candidate / a.case
        ep = _finest_balance_audit__json.loads((source / 'endpoint.json').read_text(encoding='utf-8'))
        old = _finest_balance_audit__json.loads((source / 'record.json').read_text(encoding='utf-8'))
        observer = _finest_balance_audit__Observer(ep['equality_root_s'], ep['reported_time_s'])
        original_factory = _finest_balance_audit__clock.BDF
        original_class = _finest_balance_audit__candidate.complete_class

        def factory(*args, **kwargs):
            solver = original_factory(*args, **kwargs)
            context = dict(observer.context)
            ref = _finest_balance_audit__weakref.ref(solver)
            unbound = type(solver).step

            def step():
                obj = ref()
                left = float(obj.t)
                message = unbound(obj)
                if obj.status != 'failed':
                    observer.observe(obj, left, context)
                return message
            solver.step = step
            return solver

        def instrumented_class(stream):
            Parent = original_class(stream)

            class Instrumented(Parent):

                def span(self, start, end, y, targets, detect_event=True):
                    observer.context = {'model': self.model, 'start': float(start), 'detect': bool(detect_event), 'right_side': bool(start >= self.model.ambient.breakpoint_s and _finest_balance_audit__np.isfinite(self.model.ambient.breakpoint_s))}
                    return super().span(start, end, y, targets, detect_event)
            return Instrumented
        _finest_balance_audit__clock.BDF = factory
        _finest_balance_audit__candidate.complete_class = instrumented_class
        _finest_balance_audit__sys.argv = [str(_finest_balance_audit__source_file()), '--inputs', str(a.inputs), '--contract', str(a.contract), '--base-code', str(a.base_code), '--refinements', str(a.refinements), '--production', str(a.production), '--out', str(a.out), '--case', a.case]
        _finest_balance_audit__candidate.main()
        dest = a.out / a.case
        new = _finest_balance_audit__json.loads((dest / 'record.json').read_text(encoding='utf-8'))
        same_steps = old['accepted_step_time_sha256'] == new['accepted_step_time_sha256']
        assert same_steps and old['event_root_s'] == new['event_root_s'] and (old['nfev'] == new['nfev'])
        with _finest_balance_audit__np.load(source / 'samples.npz') as x, _finest_balance_audit__np.load(dest / 'samples.npz') as y:
            same_times = _finest_balance_audit__np.array_equal(x['t_s'], y['t_s'])
            same_T = _finest_balance_audit__np.array_equal(x['T_K'], y['T_K'])
            same_C = _finest_balance_audit__np.array_equal(x['C'], y['C'])
        assert same_times and same_T and same_C
        with _finest_balance_audit__np.load(dest / 'internal.npz') as z:
            mean0 = float(z['mean_C'][0])
            mean1 = float(z['mean_C'][-1])
        rows = []
        for order, r in observer.rows.items():
            rows.append({'quadrature_order': order, 'mean_C_change': mean1 - mean0, 'integrated_moisture_outflux': r['moisture_flux'], 'moisture_balance_abs': abs(mean1 - mean0 + r['moisture_flux']), 'integrated_J_thermal_accumulation': r['heat_store'], 'integrated_J_thermal_boundary_flux': r['heat_flux'], 'thermal_rate_integral_residual_abs': abs(r['heat_store'] + r['heat_flux']), 'thermal_rate_integral_residual_relative': abs(r['heat_store'] + r['heat_flux']) / max(r['heat_abs_flux'], 1e-30), 'integrated_accepted_step_intervals': r['intervals']})
        result = {'completed': True, 'passed': bool(rows[-1]['moisture_balance_abs'] < 5e-08 and rows[-1]['thermal_rate_integral_residual_relative'] < 1e-07), 'case': a.case, 'quadrature': rows, 'quadrature_flux_difference': abs(rows[0]['integrated_moisture_outflux'] - rows[1]['integrated_moisture_outflux']), 'thermal_scope': 'Integral J*b(C)*T_t from an independent derivative of the actual local-time BDF polynomial, plus current boundary flux per initial volume. Not a derivative of J*b*T and not latent heat.', 'observer_does_not_change_solver': {'accepted_step_hash_identical': same_steps, 'equality_root_bitwise_identical': old['event_root_s'] == new['event_root_s'], 'all_output_times_bitwise_identical': same_times, 'all_compact_T_bitwise_identical': same_T, 'all_compact_C_bitwise_identical': same_C}, 'interval_truncation': 'Pre-event observer truncates at the candidate equality root; post-event observer truncates at its exact reported tick. Exact replay of event root is required and verified.', 'source_identity': _finest_balance_audit__source_identity(), 'reference_sha256': {p.name: _finest_balance_audit__sha(p) for p in [source / 'record.json', source / 'endpoint.json']}, 'input_sha256': {'model_config': _finest_balance_audit__sha(a.contract), 'attachment1': _finest_balance_audit__sha(a.inputs / '附件1.xlsx'), 'attachment2': _finest_balance_audit__sha(a.inputs / '附件2.xlsx')}, 'continuous_PDE_error_certified': False}
        _finest_balance_audit__write(dest / 'balance.json', result)
        _finest_balance_audit__write(a.out / f'{a.case}_balance_audit.json', result)
        metadata = _finest_balance_audit__json.loads((dest / 'metadata.json').read_text(encoding='utf-8'))
        metadata['new_time_integrated_balance_not_yet_computed'] = False
        metadata['streaming_balance_passed'] = result['passed']
        metadata['source_identity'] = _finest_balance_audit__source_identity()
        metadata['generator'] = 'Merged finest_candidate with the balance observer; original BDF span, settings and observation order retained'
        _finest_balance_audit__write(dest / 'metadata.json', metadata)
        if not result['passed']:
            raise RuntimeError('Actual finest balance checks did not pass the declared numerical gates')
        print(_finest_balance_audit__json.dumps({'case': a.case, 'balance_passed': result['passed'], 'quadrature': rows}), flush=True)

    # ===== extensions/endpoint/validation_replay.py: readable implementation =====
    _validation_replay____doc__ = 'Storage-complete replay for physical-point and whole-profile comparisons.\n\nThree frozen settings per case are repeated. Every accepted step and root must\nmatch the previously completed reference. First 3h physical samples are every\nsecond; full radial profiles cover every minute throughout the trajectory.'
    import os as _validation_replay__os
    import sys as _validation_replay__sys
    _validation_replay__sys.dont_write_bytecode = True
    for _validation_replay__k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _validation_replay__os.environ[_validation_replay__k] = '1'
    from pathlib import Path as _validation_replay__Path
    import argparse as _validation_replay__argparse
    import json as _validation_replay__json
    import gc as _validation_replay__gc
    import numpy as _validation_replay__np
    _validation_replay__local_class = _sections['global_refinement_local_clock'].local_class
    _validation_replay__sha = _sections['endpoint_study'].sha
    _validation_replay__write = _sections['endpoint_study'].write
    _validation_replay__source_identity = source_identity

    class _validation_replay__Recorder:

        def __init__(self, model, directory):
            self.model = model
            self.directory = directory
            directory.mkdir(parents=True, exist_ok=False)
            self.times = []
            self.T = []
            self.C = []
            self.R = []
            self.checkpoints = {}
            self.checkpoint_labels = {}
            self.physical_T = []
            self.physical_C = []
            self.valid = []
            self.r = _validation_replay__np.arange(21) * 0.001
            self.internal = {}
            self.xi = _validation_replay__np.linspace(0, 1, 21)

        def add(self, t, y, full_label=None):
            if self.times and abs(self.times[-1] - t) < 1e-09:
                return
            T, C = self.model.unpack(y)
            R = float(self.model.geometry.radius(t))
            self.times.append(float(t))
            self.R.append(R)
            self.T.append(_validation_replay__np.interp(self.xi, self.model.xi, T))
            self.C.append(_validation_replay__np.interp(self.xi, self.model.xi, C))
            valid = self.r < R
            ta = _validation_replay__np.full(22, _validation_replay__np.nan)
            ca = _validation_replay__np.full(22, _validation_replay__np.nan)
            ta[:-1][valid] = _validation_replay__np.interp(self.r[valid] / R, self.model.xi, T)
            ca[:-1][valid] = _validation_replay__np.interp(self.r[valid] / R, self.model.xi, C)
            ta[-1] = T[-1]
            ca[-1] = C[-1]
            self.physical_T.append(ta)
            self.physical_C.append(ca)
            self.valid.append(valid)
            if abs(t / 60 - round(t / 60)) < 1e-12 or full_label:
                self.internal[float(t)] = y.copy()
            if full_label or abs(t / 21600 - round(t / 21600)) < 1e-12:
                self.checkpoints[float(t)] = y.copy()
                self.checkpoint_labels[float(t)] = full_label or 'six_hour'

        def flush(self, final=False):
            if not final:
                return
            m = self.model
            n = m.nr
            times = _validation_replay__np.array(self.times)
            T = _validation_replay__np.array(self.T)
            C = _validation_replay__np.array(self.C)
            _validation_replay__np.savez_compressed(self.directory / 'samples.npz', t_s=times, xi=self.xi, T_K=T, C=C, R_m=_validation_replay__np.array(self.R))
            PT = _validation_replay__np.array(self.physical_T)
            PC = _validation_replay__np.array(self.physical_C)
            _validation_replay__np.savez_compressed(self.directory / 'physical_samples.npz', t_s=times, r_sample_m=self.r, R_m=_validation_replay__np.array(self.R), valid_interior=_validation_replay__np.array(self.valid), T_K_sample=PT[:, :-1], C_sample=PC[:, :-1], T_K_surface=PT[:, -1], C_surface=PC[:, -1])
            t = _validation_replay__np.array(sorted(self.internal))
            y = _validation_replay__np.array([self.internal[x] for x in t])
            _validation_replay__np.savez_compressed(self.directory / 'radial_profiles.npz', t_s=t, xi=m.xi, R_m=m.geometry.radius(t), T_K=y[:, :n] + m.temperature_reference_K, C=y[:, n:])
            t = _validation_replay__np.array(sorted(self.checkpoints))
            y = _validation_replay__np.array([self.checkpoints[x] for x in t])
            _validation_replay__np.savez_compressed(self.directory / 'checkpoints.npz', t_s=t, xi=m.xi, T_K=y[:, :n] + m.temperature_reference_K, C=y[:, n:], labels=_validation_replay__np.array([self.checkpoint_labels[x] for x in t]))

    def _validation_replay__main():
        p = _validation_replay__argparse.ArgumentParser(description=_validation_replay____doc__)
        for k in ('inputs', 'contract', 'base-code', 'references', 'out'):
            p.add_argument('--' + k, type=_validation_replay__Path, required=True)
        p.add_argument('--case', choices=['p23', 'q4'], required=True)
        a = p.parse_args()
        m = _sections['moving_model']
        RelativeTemperatureModel = _sections['relative_temperature'].RelativeTemperatureModel
        stream = _sections['absolute_temperature_check']
        Local = _validation_replay__local_class(stream)
        case = a.case
        N = 1280 if case == 'p23' else 2560
        rows = []
        for count, rtol in [(N, 1e-11), (2 * N, 1e-11), (2 * N, 1e-12)]:
            name = f'{case}_N{count}_rtol{rtol:g}'
            refdir = a.references / name
            ref = _validation_replay__json.loads((refdir / 'record.json').read_text(encoding='utf-8'))
            model = RelativeTemperatureModel.from_model(m.build_model(a.inputs, a.contract, N=count, properties='P23' if case == 'p23' else 'P4', geometry='fixed' if case == 'p23' else 'measured'))
            rec = _validation_replay__Recorder(model, a.out / name)
            solver = Local(model, rec, progress_seconds=45)
            solver.s = m.IntegrationSettings(**ref['settings'])
            result = solver.integrate()
            same = result['event_root_s'] == ref['event_root_s'] and result['accepted_step_time_sha256'] == ref['accepted_step_time_sha256'] and (result['nfev'] == ref['nfev'])
            with _validation_replay__np.load(refdir / 'samples.npz') as old, _validation_replay__np.load(rec.directory / 'samples.npz') as new:
                exact = _validation_replay__np.array_equal(old['t_s'], new['t_s']) and _validation_replay__np.array_equal(old['T_K'], new['T_K']) and _validation_replay__np.array_equal(old['C'], new['C'])
            assert same and exact
            result.update({'case': case, 'N': count, 'settings': solver.s.__dict__, 'reference_root_step_samples_bitwise_identical': bool(same and exact), 'reference_record_sha256': _validation_replay__sha(refdir / 'record.json'), 'reference_samples_sha256': _validation_replay__sha(refdir / 'samples.npz'), 'scope': 'Physical radii 0..2cm strictly inside current radius plus actual surface, every second 0..10800s and every minute thereafter. Full radial fields at every integer minute and event/report diagnostic points.', 'source_identity': _validation_replay__source_identity(), 'input_sha256': {'model_config': _validation_replay__sha(a.contract), 'attachment1': _validation_replay__sha(a.inputs / '附件1.xlsx'), 'attachment2': _validation_replay__sha(a.inputs / '附件2.xlsx')}})
            _validation_replay__write(rec.directory / 'record.json', result)
            rows.append({'directory': name, 'root_s': result['event_root_s'], 'replay_passed': True})
            print(_validation_replay__json.dumps(rows[-1]), flush=True)
            del rec, solver, model
            _validation_replay__gc.collect()
        _validation_replay__write(a.out / f'{case}_replays.json', {'completed': True, 'passed': True, 'rows': rows})

    # ===== extensions/endpoint/compare_validation_replays.py: readable implementation =====
    _compare_validation_replays____doc__ = 'Field convergence on actually replayed physical points and radial profiles.'
    import os as _compare_validation_replays__os
    import sys as _compare_validation_replays__sys
    _compare_validation_replays__sys.dont_write_bytecode = True
    for _compare_validation_replays__k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _compare_validation_replays__os.environ[_compare_validation_replays__k] = '1'
    from pathlib import Path as _compare_validation_replays__Path
    import argparse as _compare_validation_replays__argparse
    import json as _compare_validation_replays__json
    import gc as _compare_validation_replays__gc
    import numpy as _compare_validation_replays__np
    _compare_validation_replays__sha = _sections['endpoint_study'].sha
    _compare_validation_replays__write = _sections['endpoint_study'].write
    _compare_validation_replays__source_identity = source_identity

    def _compare_validation_replays__read(p):
        with _compare_validation_replays__np.load(p) as z:
            return {k: z[k] for k in z.files}

    def _compare_validation_replays__physical(a, b, end):
        aa = _compare_validation_replays__read(a / 'physical_samples.npz')
        bb = _compare_validation_replays__read(b / 'physical_samples.npz')
        tt, ia, ib = _compare_validation_replays__np.intersect1d(aa['t_s'], bb['t_s'], return_indices=True)
        mask = tt <= end
        tt = tt[mask]
        ia = ia[mask]
        ib = ib[mask]
        assert _compare_validation_replays__np.array_equal(aa['r_sample_m'], bb['r_sample_m']) and _compare_validation_replays__np.array_equal(aa['R_m'][ia], bb['R_m'][ib])
        assert _compare_validation_replays__np.array_equal(aa['valid_interior'][ia], bb['valid_interior'][ib])
        r = aa['r_sample_m']
        out = {'common_time_rows': len(tt), 'first_time_s': float(tt[0]), 'last_time_s': float(tt[-1]), 'first3h_every_second_included': bool(_compare_validation_replays__np.all(_compare_validation_replays__np.isin(_compare_validation_replays__np.arange(10801.0), tt))), 'all_integer_minutes_to_comparison_end_included': bool(_compare_validation_replays__np.all(_compare_validation_replays__np.isin(_compare_validation_replays__np.arange(0, end, 60.0), tt))), 'fixed_physical_radii_m': r.tolist(), 'true_surface_included': True, 'domain_masks_identical': True, 'time_interpolation_used': False}
        for f in ['T_K', 'C']:
            x = _compare_validation_replays__np.column_stack([aa[f + '_sample'][ia], aa[f + '_surface'][ia]])
            y = _compare_validation_replays__np.column_stack([bb[f + '_sample'][ib], bb[f + '_surface'][ib]])
            assert _compare_validation_replays__np.array_equal(_compare_validation_replays__np.isnan(x), _compare_validation_replays__np.isnan(y))
            d = _compare_validation_replays__np.abs(x - y)
            j, k = _compare_validation_replays__np.unravel_index(_compare_validation_replays__np.nanargmax(d), d.shape)
            out[f] = {'max_abs_difference': float(d[j, k]), 'time_s': float(tt[j]), 'position': 'surface' if k == len(r) else float(r[k]), 'position_unit': 'm for fixed points, instantaneous R(t) for surface', 'finite_compared_values': int(_compare_validation_replays__np.count_nonzero(_compare_validation_replays__np.isfinite(d)))}
        return out

    def _compare_validation_replays__profiles(a, b, end):
        aa = _compare_validation_replays__read(a / 'radial_profiles.npz')
        bb = _compare_validation_replays__read(b / 'radial_profiles.npz')
        tt, ia, ib = _compare_validation_replays__np.intersect1d(aa['t_s'], bb['t_s'], return_indices=True)
        mask = (tt <= end) & (_compare_validation_replays__np.abs(tt / 60 - _compare_validation_replays__np.rint(tt / 60)) < 1e-12)
        tt = tt[mask]
        ia = ia[mask]
        ib = ib[mask]
        assert _compare_validation_replays__np.array_equal(aa['R_m'][ia], bb['R_m'][ib])
        assert _compare_validation_replays__np.array_equal(tt, _compare_validation_replays__np.arange(0, end, 60.0))
        union = _compare_validation_replays__np.union1d(aa['xi'], bb['xi'])
        fine = bb['xi']
        out = {'common_time_rows': len(tt), 'first_time_s': float(tt[0]), 'last_time_s': float(tt[-1]), 'every_integer_minute_included': True, 'coarse_nodes': len(aa['xi']), 'fine_nodes': len(fine), 'union_nodes': len(union), 'time_interpolation_used': False, 'reconstruction': 'Piecewise linear in xi; common geometry maps each xi to the same physical r=R(t)*xi. Union-node maximum is the exact spatial maximum of the difference of the two declared piecewise-linear reconstructions at each saved minute.'}
        for f in ['T_K', 'C']:
            best = -1.0
            fine_best = -1.0
            location = None
            fine_location = None
            for j, (i, k) in enumerate(zip(ia, ib)):
                xa = _compare_validation_replays__np.interp(union, aa['xi'], aa[f][i])
                xb = _compare_validation_replays__np.interp(union, bb['xi'], bb[f][k])
                d = abs(xb - xa)
                index = int(d.argmax())
                if d[index] > best:
                    best = float(d[index])
                    location = {'time_s': float(tt[j]), 'xi': float(union[index]), 'r_m': float(union[index] * aa['R_m'][i])}
                df = abs(bb[f][k] - _compare_validation_replays__np.interp(fine, aa['xi'], aa[f][i]))
                index = int(df.argmax())
                if df[index] > fine_best:
                    fine_best = float(df[index])
                    fine_location = {'time_s': float(tt[j]), 'xi': float(fine[index]), 'r_m': float(fine[index] * aa['R_m'][i])}
            out[f] = {'max_abs_difference_on_union': best, 'union_max_location': location, 'max_abs_difference_on_all_fine_nodes': fine_best, 'fine_max_location': fine_location, 'fine_node_compared_values': len(tt) * len(fine), 'union_node_compared_values': len(tt) * len(union)}
        return out

    def _compare_validation_replays__main():
        p = _compare_validation_replays__argparse.ArgumentParser(description=_compare_validation_replays____doc__)
        p.add_argument('--replays', type=_compare_validation_replays__Path, required=True)
        p.add_argument('--out', type=_compare_validation_replays__Path, required=True)
        p.add_argument('--case', choices=['p23', 'q4', 'both'], default='both')
        a = p.parse_args()
        a.out.mkdir(parents=True, exist_ok=True)
        summary = []
        for case in ['p23', 'q4'] if a.case == 'both' else [a.case]:
            n = 1280 if case == 'p23' else 2560
            ds = [a.replays / f'{case}_N{N}_rtol{rtol:g}' for N, rtol in [(n, 1e-11), (2 * n, 1e-11), (2 * n, 1e-12)]]
            rr = [_compare_validation_replays__json.loads((d / 'record.json').read_text(encoding='utf-8')) for d in ds]
            assert all((r['reference_root_step_samples_bitwise_identical'] for r in rr))
            pairs = []
            for (ia, ib), kind in [((0, 1), 'matched_tolerance_spatial'), ((1, 2), 'same_fine_grid_time_tolerance')]:
                end = min(rr[ia]['event_root_s'], rr[ib]['event_root_s'])
                ph = _compare_validation_replays__physical(ds[ia], ds[ib], end)
                pr = _compare_validation_replays__profiles(ds[ia], ds[ib], end)
                pairs.append({'kind': kind, 'comparison_end_s': end, 'physical_points': ph, 'full_radial_minute_profiles': pr, 'combined_sample_T_difference_K': max(ph['T_K']['max_abs_difference'], pr['T_K']['max_abs_difference_on_union']), 'combined_sample_C_difference': max(ph['C']['max_abs_difference'], pr['C']['max_abs_difference_on_union'])})
                _compare_validation_replays__gc.collect()
            space, time = pairs
            conv = {'meshes': [{'case': case, 'N': n, 'root_s': rr[0]['event_root_s'], 'root_h': rr[0]['event_root_s'] / 3600, 'root_difference_to_next_s': abs(rr[1]['event_root_s'] - rr[0]['event_root_s']), 'sample_T_difference_to_next_K': space['combined_sample_T_difference_K'], 'sample_C_difference_to_next': space['combined_sample_C_difference']}, {'case': case, 'N': 2 * n, 'root_s': rr[1]['event_root_s'], 'root_h': rr[1]['event_root_s'] / 3600, 'root_difference_to_next_s': None, 'sample_T_difference_to_next_K': None, 'sample_C_difference_to_next': None, 'no_further_grid_compared': True}], 'time_tolerance_root_difference_s': abs(rr[2]['event_root_s'] - rr[1]['event_root_s']), 'tight_equality_root_s': rr[2]['event_root_s'], 'tight_settings': rr[2]['settings'], 'time_tolerance_sample_T_difference_K': time['combined_sample_T_difference_K'], 'time_tolerance_sample_C_difference': time['combined_sample_C_difference'], 'comparison_end_s': min((r['event_root_s'] for r in rr)), 'sampling': 'Union of physical 21 fixed radii plus true surface at every second to 3h and every minute thereafter, and exact spatial maximum of piecewise-linear reconstruction differences over all radial-node unions at every minute.', 'finite_samples_not_continuous_PDE_error_bound': True, 'candidate_uses_tight_settings': True, 'spatial_pair_uses_matched_main_settings': True, 'detailed_comparisons': pairs, 'replay_sources': [{'directory': d.name, 'N': r['N'], 'settings': r['settings'], 'record_sha256': _compare_validation_replays__sha(d / 'record.json'), 'physical_samples_sha256': _compare_validation_replays__sha(d / 'physical_samples.npz'), 'radial_profiles_sha256': _compare_validation_replays__sha(d / 'radial_profiles.npz'), 'accepted_step_time_sha256': r['accepted_step_time_sha256'], 'root_s': r['event_root_s']} for d, r in zip(ds, rr)], 'source_identity': _compare_validation_replays__source_identity()}
            dest = a.out / case
            dest.mkdir(exist_ok=True)
            _compare_validation_replays__write(dest / 'convergence.json', conv)
            summary.append({'case': case, 'completed': True, 'pairs': pairs})
            print(_compare_validation_replays__json.dumps({'case': case, 'spatial_T': space['combined_sample_T_difference_K'], 'spatial_C': space['combined_sample_C_difference'], 'time_T': time['combined_sample_T_difference_K'], 'time_C': time['combined_sample_C_difference']}), flush=True)
        _compare_validation_replays__write(a.out / 'validation_replay_comparison.json', {'completed': True, 'cases': summary, 'continuous_PDE_error_certified': False})

    # ===== extensions/endpoint/reproduce_endpoint.py: readable implementation =====
    _reproduce_endpoint____doc__ = 'Portable orchestration of endpoint error-control research.\n\nThe original model package is read-only. All generated files go to --work or\n--scratch. --runs points to completed P23/Q4 production arrays. Ancillary standalone review dispatch has been removed from this delivery.'
    import os as _reproduce_endpoint__os
    import sys as _reproduce_endpoint__sys
    _reproduce_endpoint__sys.dont_write_bytecode = True
    for _reproduce_endpoint__k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        _reproduce_endpoint__os.environ[_reproduce_endpoint__k] = '1'
    from pathlib import Path as _reproduce_endpoint__Path
    import argparse as _reproduce_endpoint__argparse
    import json as _reproduce_endpoint__json
    import subprocess as _reproduce_endpoint__subprocess
    _reproduce_endpoint__dispatch_step = dispatch_step
    _reproduce_endpoint__source_identity = source_identity

    def _reproduce_endpoint__main():
        p = _reproduce_endpoint__argparse.ArgumentParser(description=_reproduce_endpoint____doc__)
        for k in ('base-code', 'inputs', 'contract', 'runs', 'work'):
            p.add_argument('--' + k, type=_reproduce_endpoint__Path, required=True)
        p.add_argument('--scratch', type=_reproduce_endpoint__Path)
        p.add_argument('--stage', choices=['refinement', 'candidate', 'balance', 'validation', 'all'], default='all')
        p.add_argument('--p23-refinements', type=_reproduce_endpoint__Path, help='Optional existing P23 reference directory; default is scratch/global_refinement.')
        p.add_argument('--q4-refinements', type=_reproduce_endpoint__Path, help='Optional existing Q4 reference directory; default is scratch/global_refinement.')
        p.add_argument('--case', choices=['p23', 'q4'], required=True)
        a = p.parse_args()
        scratch = a.scratch or a.work
        scratch.mkdir(parents=True, exist_ok=True)
        a.work.mkdir(parents=True, exist_ok=True)
        commands = []

        def run(name, args):
            command = _reproduce_endpoint__dispatch_step('extensions/endpoint/' + name, args)
            commands.append(command)
            _reproduce_endpoint__subprocess.run(command, check=True)
        if a.stage in ['refinement', 'all']:
            run('global_refinement_local_clock.py', ['--inputs', a.inputs, '--contract', a.contract, '--out', scratch / 'global_refinement', '--case', a.case])
        selected = [a.case]
        for case in selected:
            refs = (a.p23_refinements if case == 'p23' else a.q4_refinements) or scratch / 'global_refinement'
            common_new = ['--inputs', a.inputs, '--contract', a.contract, '--base-code', a.base_code, '--case', case]
            if a.stage in ['candidate', 'all']:
                run('finest_candidate.py', common_new + ['--refinements', refs, '--production', a.runs, '--out', scratch / 'finest_candidate'])
            if a.stage in ['balance', 'all']:
                run('finest_balance_audit.py', common_new + ['--refinements', refs, '--production', a.runs, '--candidate', scratch / 'finest_candidate', '--out', scratch / 'finest_balance'])
            if a.stage in ['validation', 'all']:
                run('validation_replay.py', common_new + ['--references', refs, '--out', scratch / 'validation_replay'])
                run('compare_validation_replays.py', ['--replays', scratch / 'validation_replay', '--out', scratch / 'validation_metrics', '--case', case])
        (a.work / 'reproduction_commands.json').write_text(_reproduce_endpoint__json.dumps({'completed': True, 'commands': commands, 'source_identity': _reproduce_endpoint__source_identity()}, ensure_ascii=False, indent=2), encoding='utf-8')

    _STEP_FUNCTIONS.update({
        'solvers/solve_all.py': _solve_all__main,
        'extensions/q1/portable_q1_disk.py': _portable_q1_disk__main,
        'extensions/q1/portable_q1_compare.py': _portable_q1_compare__main,
        'extensions/endpoint/reproduce_endpoint.py': _reproduce_endpoint__main,
        'extensions/endpoint/global_refinement_local_clock.py': _global_refinement_local_clock__main,
        'extensions/endpoint/finest_candidate.py': _finest_candidate__main,
        'extensions/endpoint/finest_balance_audit.py': _finest_balance_audit__main,
        'extensions/endpoint/validation_replay.py': _validation_replay__main,
        'extensions/endpoint/compare_validation_replays.py': _compare_validation_replays__main,
    })
    _science_initialized = True


if __name__ == "__main__":
    raise SystemExit(public_main())
