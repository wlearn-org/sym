import os
import subprocess
import sys
from setuptools import Extension, setup

if sys.platform == 'win32':
    sys.exit('wlearn-sym requires Linux. Windows is not supported.')

here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(here, 'csrc')
repo_root = os.path.dirname(here)
src_dir = os.path.join(repo_root, 'src')


def _repo_source_files():
    if not os.path.isdir(src_dir):
        return []
    return [
        name for name in sorted(os.listdir(src_dir))
        if name.endswith(('.c', '.h'))
    ]


repo_files = _repo_source_files()
sync_script = os.path.join(here, 'scripts', 'sync-csrc.py')
if repo_files:
    print('setup.py: syncing csrc/ from repo sources...')
    subprocess.check_call([sys.executable, sync_script])
elif not os.path.isdir(csrc):
    raise SystemExit(
        'csrc/ directory not found and repo src/ is unavailable.\n'
        'Run from the repo: python py/scripts/sync-csrc.py'
    )

sources = [
    os.path.join('csrc', name)
    for name in sorted(os.listdir(csrc))
    if name.endswith('.c') and not name.startswith('test')
]
sources.append(os.path.join('wlearn_sym', '_native.c'))

setup(
    ext_modules=[
        Extension(
            'wlearn_sym._native',
            sources=sources,
            include_dirs=[csrc],
            extra_compile_args=['-std=c11', '-O2', '-D_POSIX_C_SOURCE=200809L'],
            libraries=['m'],
        )
    ],
)
