"""Quantitative quality harness.

Eyeballing "looks jagged" does not tell you whether a change helped. This
builds a self-contained page that rasterizes each generated SVG back to a
canvas and scores it against a *lossless* reference image, so preset and
preprocessing changes can be compared with a number instead of a vibe.

Metrics reported per run:
  RMSE        - overall pixel error vs the reference (lower is better)
  EDGE_RMSE   - error restricted to reference edge pixels; this is where
                jaggedness, bumps and rounded corners actually show up
  paths       - path count (lower is better at equal fidelity)
  bytes       - output size

Usage:
    python tests/quality_harness.py                 # default run set
    python tests/quality_harness.py --open          # also print the URL

It writes frontend/_quality.html, which Express already serves. Delete it when
finished - it is a dev tool, not part of the app.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vectorizer import vectorize_bytes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathstats import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
OUT_HTML = ROOT / "frontend" / "_quality.html"

# (label, input file, reference file, preset, overrides)
DEFAULT_RUNS = [
    ("badge BEFORE (first report)", "badge_repro.jpg", "badge_repro.png", "logo",
     {"boundary_smooth_sigma": 0.0, "corner_threshold": 40, "length_threshold": 4.0,
      "background_edge_bleed": 0}),
    ("badge AFTER", "badge_repro.jpg", "badge_repro.png", "auto", None),
    ("serif BEFORE (first report)", "logo_repro.jpg", "logo_repro.png", "logo",
     {"boundary_smooth_sigma": 0.0, "corner_threshold": 40, "length_threshold": 4.0,
      "background_edge_bleed": 0}),
    ("serif AFTER", "logo_repro.jpg", "logo_repro.png", "auto", None),
    ("flat_art sample", "sample_flat_art.png", "sample_flat_art.png", "flat_art",
     {"background": "never"}),
    ("typography sample", "sample_typography.png", "sample_typography.png",
     "typography", {"background": "never"}),
    ("line_art sample", "sample_line_art.png", "sample_line_art.png", "line_art", None),
]


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def svg_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def run_case(label, source, reference, preset, overrides) -> dict:
    source_path = SAMPLES / source
    reference_path = SAMPLES / reference
    if not source_path.exists():
        raise SystemExit(
            f"missing {source_path} - run `python tests/make_logo_repro.py` first"
        )

    outcome = vectorize_bytes(
        source_path.read_bytes(), preset_name=preset, overrides=overrides
    )
    meta = outcome.meta
    seg = summarise(outcome.svg)
    print(
        f"{label:28s} preset={meta['preset_used']:10s} "
        f"paths={meta['path_count']:5d} {meta['svg_bytes'] / 1024:7.1f}KB "
        f"{meta['processing_ms']:6.0f}ms "
        f"ss={meta['supersample']} lines={seg['line_share']:.0%} "
        f"seg={seg['segments']}"
    )
    return {
        "label": f"{label} [{seg['line_share']:.0%} lines]",
        "preset": meta["preset_used"],
        "paths": meta["path_count"],
        "bytes": meta["svg_bytes"],
        "ms": meta["processing_ms"],
        "steps": meta["preprocess_steps"],
        "reference": data_uri(reference_path),
        "svg": svg_data_uri(outcome.svg),
    }


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Vectorization quality harness</title>
<style>
 body{font:13px system-ui;margin:20px;background:#11131a;color:#e8ecf4}
 h1{font-size:16px;margin:0 0 4px}
 p.sub{color:#96a0b4;margin:0 0 18px}
 table{border-collapse:collapse;width:100%;margin-bottom:24px}
 th,td{border:1px solid #262b38;padding:6px 9px;text-align:right;font-variant-numeric:tabular-nums}
 th{background:#1a1e29;text-align:left;font-weight:600}
 td.l{text-align:left}
 .best{color:#66d9a0;font-weight:700}
 .row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}
 figure{margin:0}
 figcaption{color:#96a0b4;font-size:11px;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
 canvas,img{width:100%;border:1px solid #262b38;background:#2b2927;display:block}
 h2{font-size:13px;margin:18px 0 8px;color:#c8d2e4}
</style></head><body>
<h1>Vectorization quality harness</h1>
<p class="sub">SVG rasterized back to canvas and scored against the lossless reference.
RMSE = overall pixel error. EDGE RMSE = error on reference edge pixels only, which is
where jagged edges, bumps and rounded corners live. Lower is better.</p>
<div id="summary">measuring…</div>
<div id="detail"></div>
<script>
const RUNS = __RUNS__;
// Compare at the reference's NATIVE size. Resampling into a rounded box makes
// the SVG (preserveAspectRatio="xMidYMid meet") letterbox by a fraction of a
// pixel, which swamps the real tracing error with a constant offset.

function load(src){return new Promise((res,rej)=>{const i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=src;});}

function draw(img,w,h){
  const c=document.createElement('canvas');c.width=w;c.height=h;
  const x=c.getContext('2d',{willReadFrequently:true});
  // Flatten onto the logo's own dark field so transparent background pixels
  // are compared against the colour they replaced, not against black.
  x.fillStyle='#2b2927';x.fillRect(0,0,w,h);
  x.drawImage(img,0,0,w,h);
  return {canvas:c,data:x.getImageData(0,0,w,h).data};
}

function edgeMask(data,w,h){
  // Sobel magnitude on luminance, thresholded.
  const lum=new Float32Array(w*h);
  for(let i=0;i<w*h;i++){lum[i]=0.299*data[i*4]+0.587*data[i*4+1]+0.114*data[i*4+2];}
  const mask=new Uint8Array(w*h);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
    const i=y*w+x;
    const gx=-lum[i-w-1]-2*lum[i-1]-lum[i+w-1]+lum[i-w+1]+2*lum[i+1]+lum[i+w+1];
    const gy=-lum[i-w-1]-2*lum[i-w]-lum[i-w+1]+lum[i+w-1]+2*lum[i+w]+lum[i+w+1];
    if(Math.hypot(gx,gy)>60) mask[i]=1;
  }
  // Dilate by 2px so we score the neighbourhood of an edge, not one exact line.
  // Separable (H then V): O(w*h*5*2) instead of O(w*h*25).
  const tmp=new Uint8Array(w*h), out=new Uint8Array(w*h);
  for(let y=0;y<h;y++){const row=y*w;
    for(let x=0;x<w;x++){let v=0;
      for(let dx=-2;dx<=2;dx++){const xx=x+dx;if(xx>=0&&xx<w&&mask[row+xx]){v=1;break;}}
      tmp[row+x]=v;}}
  for(let x=0;x<w;x++){
    for(let y=0;y<h;y++){let v=0;
      for(let dy=-2;dy<=2;dy++){const yy=y+dy;if(yy>=0&&yy<h&&tmp[yy*w+x]){v=1;break;}}
      out[y*w+x]=v;}}
  return out;
}

function score(refData,testData,mask,w,h){
  let sum=0,n=0,esum=0,en=0;
  for(let i=0;i<w*h;i++){
    let d=0;
    for(let c=0;c<3;c++){const v=refData[i*4+c]-testData[i*4+c];d+=v*v;}
    d/=3;
    sum+=d;n++;
    if(mask[i]){esum+=d;en++;}
  }
  return {rmse:Math.sqrt(sum/n), edge:en?Math.sqrt(esum/en):0, edgePx:en};
}

(async()=>{
  const rows=[];
  const detail=document.getElementById('detail');
  for(const run of RUNS){
    window.__PROGRESS__=run.label;
    await new Promise(r=>setTimeout(r,0));
    const ref=await load(run.reference);
    const W=ref.naturalWidth, h=ref.naturalHeight;
    const r=draw(ref,W,h);
    const svgImg=await load(run.svg);
    const t=draw(svgImg,W,h);
    const mask=edgeMask(r.data,W,h);
    const s=score(r.data,t.data,mask,W,h);
    rows.push({...run,...s});

    const sec=document.createElement('div');
    sec.innerHTML='<h2>'+run.label+' — '+run.paths+' paths, '
      +(run.bytes/1024).toFixed(1)+' KB, RMSE '+s.rmse.toFixed(2)
      +', EDGE '+s.edge.toFixed(2)+'</h2>';
    const grid=document.createElement('div');grid.className='row';
    const mk=(cap,node)=>{const f=document.createElement('figure');
      f.innerHTML='<figcaption>'+cap+'</figcaption>';f.appendChild(node);return f;};
    grid.appendChild(mk('reference',r.canvas));
    grid.appendChild(mk('generated svg',t.canvas));
    // difference map
    const dc=document.createElement('canvas');dc.width=W;dc.height=h;
    const dx=dc.getContext('2d');const id=dx.createImageData(W,h);
    for(let i=0;i<W*h;i++){
      let d=0;for(let c=0;c<3;c++){const v=Math.abs(r.data[i*4+c]-t.data[i*4+c]);d=Math.max(d,v);}
      const v=Math.min(255,d*3);
      id.data[i*4]=v;id.data[i*4+1]=Math.round(v*0.25);id.data[i*4+2]=Math.round(v*0.4);id.data[i*4+3]=255;
    }
    dx.putImageData(id,0,0);
    grid.appendChild(mk('difference (brighter = worse)',dc));
    sec.appendChild(grid);detail.appendChild(sec);
  }

  const bestR=Math.min(...rows.map(r=>r.rmse));
  const bestE=Math.min(...rows.map(r=>r.edge));
  let html='<table><tr><th>run</th><th>preset</th><th>paths</th><th>KB</th><th>ms</th><th>RMSE</th><th>EDGE RMSE</th></tr>';
  for(const r of rows){
    html+='<tr><td class="l">'+r.label+'</td><td class="l">'+r.preset+'</td><td>'+r.paths
      +'</td><td>'+(r.bytes/1024).toFixed(1)+'</td><td>'+Math.round(r.ms)
      +'</td><td class="'+(r.rmse===bestR?'best':'')+'">'+r.rmse.toFixed(2)
      +'</td><td class="'+(r.edge===bestE?'best':'')+'">'+r.edge.toFixed(2)+'</td></tr>';
  }
  html+='</table>';
  document.getElementById('summary').innerHTML=html;
  window.__RESULTS__=rows.map(r=>({label:r.label,preset:r.preset,paths:r.paths,
    kb:+(r.bytes/1024).toFixed(1),ms:Math.round(r.ms),
    rmse:+r.rmse.toFixed(2),edge:+r.edge.toFixed(2)}));
  window.__DONE__=true;
})();
</script></body></html>
"""


def build_page(results: list[dict], out_path: Path = OUT_HTML) -> Path:
    """Write the comparison page for arbitrary runs.

    Each result needs: label, preset, paths, bytes, ms, reference, svg
    (the last two as data URIs).
    """
    out_path.write_text(PAGE.replace("__RUNS__", json.dumps(results)), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    parser.parse_args()
    logging.disable(logging.INFO)

    results = [run_case(*case) for case in DEFAULT_RUNS]
    build_page(results)
    print(f"\nwrote {OUT_HTML}  ->  http://127.0.0.1:5000/_quality.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
