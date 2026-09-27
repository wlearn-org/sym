import os
import subprocess
import sys
from setuptools import Extension, find_packages, setup

if sys.platform == 'win32':
    sys.exit('wlearn-sym requires Linux. Windows is not supported.')

here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(here, 'csrc')
repo_root = os.path.dirname(here)
src_dir = os.path.join(repo_root, 'src')
readme = os.path.join(here, 'README.md')


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
    name='wlearn-sym',
    version='0.1.0',
    description='Symbolic regression, classification, and formula features backed by the wlearn C11 core',
    long_description=open(readme, encoding='utf-8').read() if os.path.isfile(readme) else '',
    long_description_content_type='text/markdown',
    author='Anton Zemlyansky',
    url='https://github.com/wlearn-org/sym',
    license='Apache-2.0',
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.22',
        'wlearn>=0.1.0',
    ],
    packages=find_packages(),
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
