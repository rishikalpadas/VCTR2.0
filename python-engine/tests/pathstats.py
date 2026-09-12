"""Count REAL path segments, honouring SVG implicit command repetition."""
import re

_TOKEN = re.compile(r'([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)')
_ARGS = {'m':2,'l':2,'h':1,'v':1,'c':6,'s':4,'q':4,'t':2,'a':7,'z':0}

def segment_counts(d: str) -> dict:
    tokens = []
    for m in _TOKEN.finditer(d):
        tokens.append(m.group(1) if m.group(1) else float(m.group(2)))
    counts = {'curve': 0, 'line': 0, 'move': 0}
    i = 0
    cmd = None
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
            if cmd.lower() == 'z':
                continue
        elif cmd is None:
            i += 1
            continue
        n = _ARGS[cmd.lower()]
        if n == 0:
            continue
        if i + n > len(tokens):
            break
        # consume one instance
        i += n
        k = cmd.lower()
        if k == 'm':
            counts['move'] += 1
            cmd = 'l' if cmd.islower() else 'L'   # implicit lineto after moveto
        elif k in ('c', 's', 'q', 't', 'a'):
            counts['curve'] += 1
        else:
            counts['line'] += 1
    return counts

def summarise(svg: str) -> dict:
    total = {'curve': 0, 'line': 0, 'move': 0}
    for d in re.findall(r'd="([^"]+)"', svg):
        c = segment_counts(d)
        for k in total:
            total[k] += c[k]
    seg = total['curve'] + total['line']
    total['line_share'] = total['line'] / seg if seg else 0.0
    total['segments'] = seg
    return total
