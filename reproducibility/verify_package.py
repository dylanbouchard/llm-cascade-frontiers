"""Check source integrity, syntax, and runnable-stage paths without loading data."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def main():
    errors = []
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for item in manifest:
        path = ROOT/item['path']
        if not path.is_file():
            errors.append(f'Missing source: {item["path"]}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['packaged_sha256']:
            errors.append(f'Checksum mismatch: {item["path"]}')
    paths = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.venv', '.cache', '__pycache__', '.git')]
        paths.extend(Path(directory)/n for n in names)
        paths.extend(Path(directory)/d for d in dirs if (Path(directory)/d).is_symlink())
    files = [p for p in paths if p.suffix == '.py']
    for path in files:
        if any(part in ('.venv', '.cache', '__pycache__') for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except SyntaxError as e:
            errors.append(str(e))
    from reproduce import commands
    stages = ['embeddings','exact','scorers','diff01','foc','synthetic','draws',
              'difficulty','costs','diagnostics','learning','reports']
    total = 0
    for stage in stages:
        for cwd, cmd in commands(stage, 4):
            total += 1
            if not Path(cmd[2]).is_file():
                errors.append(f'Missing {stage} entry point: {cmd[2]}')
            if not cwd.is_dir():
                errors.append(f'Missing working directory: {cwd}')
    for path in paths:
        if path.is_symlink():
            errors.append(f'Unexpected symbolic link: {path.relative_to(ROOT)}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(manifest)} copied-file hashes, {len(files)} Python files, {total} stage commands.')
    print('This static check does not execute experiments or verify empirical results.')

if __name__ == '__main__':
    main()
