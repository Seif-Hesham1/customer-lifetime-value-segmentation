"""Convert a '# %%' / '# %% [markdown]' percent script into an executed .ipynb.
usage: python mknb.py src.py out.ipynb"""
import sys, re, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert.preprocessors import ExecutePreprocessor
src, out = sys.argv[1:3]
text = open(src).read()
parts = re.split(r'^# %%(.*)$', text, flags=re.M)
cells = []
for tag, body in zip(parts[1::2], parts[2::2]):
    body = body.strip('\n')
    if '[markdown]' in tag:
        md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
        cells.append(new_markdown_cell(md))
    elif body.strip():
        cells.append(new_code_cell(body))
nb = new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
ExecutePreprocessor(timeout=1800, kernel_name='python3').preprocess(nb, {'metadata': {'path': '.'}})
nbformat.write(nb, out)
print('wrote', out, len(cells), 'cells')
