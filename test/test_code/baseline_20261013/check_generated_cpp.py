"""Compile only the two reviewed add-function outputs with fixed assertions."""
import json
from pathlib import Path
import re
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/base-eval-20261013'
compiler = shutil.which('g++')
if compiler is None:
    raise RuntimeError('Install g++ and add it to PATH to run the optional C++ check')
r = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
results = {}
for row in r['generations']:
    if row['id'] != 'P06':
        continue
    function = re.search(r'int add\(int a, int b\)\s*\{[^}]*\}', row['output']).group(0)
    # Both bodies were reviewed before execution: a+b and a+b+c respectively.
    if function.split('{', 1)[1].strip() not in ('return a + b;\n}', 'return a + b + c;}'):
        raise RuntimeError('Unexpected generated code; requires review')
    cases = [(1,2,3), (0,0,0), (-1,1,0), (-2,-3,-5), (100,-50,50), (123,456,579), (-1000,1000,0)]
    source = OUT / f"generated_add_{row['mode']}.cpp"
    source.write_text('#include <cassert>\n' + function + '\nint main(){\n' +
                      '\n'.join(f'assert(add({a},{b})=={expected});' for a,b,expected in cases) +
                      '\nreturn 0;\n}\n', encoding='utf-8')
    exe = source.with_suffix('.exe')
    command = [compiler, '-std=c++17', '-O2', str(source), '-o', str(exe)]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
    result = dict(function=function, cases=cases, compiler=compiler,
                  compile_command=command, compile_returncode=compiled.returncode,
                  compile_stderr=compiled.stderr, executed=False, passed_cases=0)
    if compiled.returncode == 0:
        tested = subprocess.run([str(exe)], capture_output=True, text=True, timeout=5)
        result.update(executed=True, run_returncode=tested.returncode, run_stderr=tested.stderr,
                      passed_cases=len(cases) if tested.returncode==0 else None)
    results[row['mode']] = result
(OUT / 'code_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=True, indent=2))
