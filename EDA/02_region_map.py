#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_region_map.py
================
Draw the study-region boundary on an Amap (高德) basemap together with the
dataset's geohash grid, so the boundary can be checked against the real roads
underneath it and against where the trips actually are.

Where the boundary comes from
-----------------------------
`data_select_configure_zgc_<date>.xlsx` (repo root) holds two scanline tables:

    sheet "Latitude_range"   indexed by Longitude -> [min_lat, max_lat]
    sheet "Longitude_range"  indexed by Latitude  -> [min_lon, max_lon]

(the sheet names are swapped relative to their content).  Both describe the
same polygon at 0.001 deg (~85 m in lon, ~111 m in lat) resolution, sampled
along different axes.  The ring is traced from the per-longitude table --
south edge west->east along min_lat, then north edge east->west along max_lat.

Why that table and not the other
--------------------------------
A per-latitude scanline reports one [min_lon, max_lon] per row, so it cannot
express a notch in a horizontal edge: a row crossing such a notch reports the
outer extremes and silently fills the gap.  The per-longitude table does
resolve it, and its south edge rises from 39.963 at lon 116.309 to 39.968 --
which is 北三环, sloping the way the real 3rd Ring Road slopes.  The
per-latitude table instead claims the full lon width (116.309..116.346) at
lat 39.963, i.e. it fills that notch.  Both tables are internally consistent;
they describe different shapes, and the per-longitude one matches the roads.
Their disagreement is reported at build time as a sanity check.

The datum -- settled empirically, and the answer is "no conversion"
-------------------------------------------------------------------
Amap publishes tiles in **GCJ-02**.  Geohash decoding conventionally yields
**WGS-84**; the two differ by ~500 m in Beijing, far more than any feature
here, so getting this wrong silently shifts the whole analysis.  Two
independent measurements say the data in this project is GCJ-02 end to end:

1. *Roads.*  Against the OSM (WGS-84) centrelines of 北三环 and 北四环 cached
   in `out/datum_evidence_osm_roads.json`, a 2-D translation fit puts the best
   alignment of the boundary at dlon = -0.0056 deg (-490 m), dlat = -0.0019 deg
   (-210 m).  A boundary already in WGS-84 would need a zero shift; the shift
   found is the GCJ-02 correction.  The ring's northern edge sits 15 m from
   北四环 as given (p90 63 m) and its southern edge 155 m from 北三环.

2. *Water.*  Nobody starts a bike trip in the middle of 昆明湖.  Over a
   32,519-cell mosaic covering the dense trip area, 4.1% of which is water,
   the share of geohash cells landing on water pixels is

       as-is                0.30%   (0.02% of departures)
       shifted by GCJ-02    0.98%
       shifted by -GCJ-02   1.40%

   A 2-D scan over +-300 m lat / +-800 m lon puts the global minimum exactly
   at (0, 0) and every other shift is worse.

So both the boundary and the trip coordinates are drawn **as given**.

What the "if WGS-84" switch is for
----------------------------------
The map still carries the converted variant, as a dashed reference layer and a
checkbox on the points.  It exists so the ~540 m error can be *seen* rather
than taken on trust -- switching it on is the fastest way to confirm the
finding above.  It is not a supported mode.

Usage:
    python 02_region_map.py                 # build out/region_map.html
    python 02_region_map.py --png           # ... and out/fig07_study_area.png
    python 02_region_map.py --open          # ... then open it in a browser

The HTML is the tool for *checking* the boundary (toggle the datum, clip to the
region, read the counts); the PNG is the rasterised version to put on a slide.
Both come from the same ring, so they cannot disagree.

The OSM road cache in `out/datum_evidence_osm_roads.json` is kept deliberately:
it is the raw evidence behind the datum claim above and the only way to
re-derive it.  Nothing in the pipeline reads it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

import region as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

XLSX = ROOT / "data_select_configure_zgc_20260909.xlsx"
CELLS = HERE / "out" / "cell_summary_keepzerodur_keepanomday.csv"
OUT_HTML = HERE / "out" / "region_map.html"
# Named for the plan's figure inventory (I1, #7) so a filename maps onto a
# slide without a lookup table.  The whole out/ folder follows that rule.
OUT_PNG = HERE / "out" / "fig07_study_area.png"

# Geohash cells are drawn over a bounding box a little larger than the region,
# so the boundary can be seen cutting *through* the grid rather than terminating
# at the last plotted point.
PAD_DEG = 0.004                      # ~350 m of margin on every side

AMAP_TILES = ("http://webrd0{s}.is.autonavi.com/appmaptile"
              "?size=1&scale=1&style=7&x={x}&y={y}&z={z}")
AMAP_SUBDOMAINS = "1234"
MAP_ZOOM = 14

LAT0 = 39.973                        # for local metres-per-degree

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))


# ---------------------------------------------------------------------------
# WGS-84 -> GCJ-02  (the standard public algorithm; used only for the
#                     reference layer that demonstrates the wrong datum)
# ---------------------------------------------------------------------------

_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lat, lon):
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _tf_lat(x, y):
    r = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
         + 0.2 * math.sqrt(abs(x)))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def _tf_lon(x, y):
    r = (300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
         + 0.1 * math.sqrt(abs(x)))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def wgs84_to_gcj02(lat, lon):
    if _out_of_china(lat, lon):
        return lat, lon
    dlat = _tf_lat(lon - 105.0, lat - 35.0)
    dlon = _tf_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sq = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sq) * math.pi)
    dlon = (dlon * 180.0) / (_A / sq * math.cos(radlat) * math.pi)
    return lat + dlat, lon + dlon


# ---------------------------------------------------------------------------
# The boundary itself -- loading the ring, the scanline self-check and the
# area -- lives in region.py, because 03_region_flow.py must test against
# exactly the same ring this map draws.


# ---------------------------------------------------------------------------
# Grid points
# ---------------------------------------------------------------------------

def load_points(cells: Path, bbox):
    """Geohash cells with at least one trip, inside the padded bounding box.

    Payload per point: [lat, lon, lat_shift, lon_shift, n_departures, n_arrivals]
    where lat/lon are the coordinates as the dataset holds them (GCJ-02, per the
    module docstring) and lat_shift/lon_shift are what you would get by wrongly
    re-applying the WGS-84 -> GCJ-02 offset.
    """
    df = pd.read_csv(cells)
    la0, lo0, la1, lo1 = bbox
    df = df[df.lat.between(la0, la1) & df.lon.between(lo0, lo1)]
    df = df[(df.n_departures + df.n_arrivals) > 0]

    pts = []
    for lat, lon, dep, arr in zip(df.lat, df.lon, df.n_departures, df.n_arrivals):
        slat, slon = wgs84_to_gcj02(float(lat), float(lon))
        pts.append([round(float(lat), 6), round(float(lon), 6),
                    round(slat, 5), round(slon, 5), int(dep), int(arr)])
    return pts


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

PANEL_CSS = """
.ctrl{position:fixed;top:12px;right:12px;z-index:9999;width:274px;
 background:rgba(255,255,255,.96);border-radius:8px;padding:12px 14px;
 font:13px/1.55 "Microsoft YaHei",system-ui,sans-serif;color:#222;
 box-shadow:0 2px 12px rgba(0,0,0,.25)}
.ctrl h3{margin:0 0 8px;font-size:14px}
.ctrl .k{font-size:12px;color:#666}
.ctrl .v{font-size:22px;font-weight:700;color:#1668dc;line-height:1.15}
.ctrl label{display:block;margin-top:7px;font-size:12.5px;cursor:pointer}
.ctrl hr{border:0;border-top:1px solid #e5e5e5;margin:10px 0}
.ctrl input[type=range]{width:100%;margin:4px 0 0}
.ctrl .hint{font-size:11.5px;color:#777;margin-top:8px;line-height:1.45}
.ctrl .sw{display:inline-block;width:22px;height:0;border-top:3px solid;
 vertical-align:middle;margin-right:6px}
.ctrl .row{display:flex;justify-content:space-between;font-size:12px;
 color:#444;margin-top:6px}
"""


def panel_html(n_pts, area_km2, offset_m, n_disagree, n_union):
    return f"""<h3>区域边界核对</h3>
<div class="v">{area_km2:.2f} km²</div>
<div class="k">xlsx 边界面积 · 格点 {n_pts:,} 个</div>
<hr>
<div class="k">基准（默认为已验证正确的读法）</div>
<label><input type="radio" name="datum" id="d-ok" checked>
  <span class="sw" style="border-color:#1668dc"></span>原样绘制 · GCJ-02</label>
<label><input type="radio" name="datum" id="d-err">
  <span class="sw" style="border-color:#fa8c16;border-top-style:dashed"></span>
  再转一次 WGS-84→GCJ-02</label>
<div class="row"><span>两版相距</span><b>{offset_m:.0f} m</b></div>
<hr>
<label><input type="checkbox" id="p-inside" checked> 只统计边界内格点</label>
<div class="k" style="margin-top:8px">最小出发量 <b id="thv">5</b></div>
<input type="range" id="p-th" min="0" max="200" step="5" value="5">
<hr>
<label><input type="checkbox" id="p-net"> 按净流量着色</label>
<div class="row"><span>界内格点</span><b id="n-in">–</b></div>
<div class="row"><span>界内出发 / 到达</span><b id="n-da">–</b></div>
<div class="row"><span>界内净流量</span><b id="n-net">–</b></div>
<div class="hint">
  基准已用两项独立证据确定：边界贴合 OSM 的北三环/北四环（平移拟合
  Δ经度 −0.0056°），且格点落水率在零位移处取全局最小（0.30% vs
  4.1% 的水面占比）。<br>
  扫描表自检：两表在 {n_union:,} 个格上不一致 {n_disagree} 个
  —— 是「横向扫描表达不了凹口」所致，不影响本区域。
</div>
"""


JS = r"""
<script>
window.addEventListener('load', function () {
  var MAP = __MAP__;
  var PTS = __PTS__;
  var RING_OK  = __RING_OK__;    // boundary drawn as given (GCJ-02)
  var RING_ERR = __RING_ERR__;   // same ring with GCJ-02 offset re-applied

  var el = document.createElement('div');
  el.className = 'ctrl';
  el.innerHTML = __PANEL__;
  document.body.appendChild(el);

  // ---------------------------------------------------------------- boundary
  var polyOk = L.polygon(RING_OK, {
      color: '#1668dc', weight: 3, opacity: .95, fill: true,
      fillColor: '#1668dc', fillOpacity: .06}).bindTooltip('边界（原样）');
  var polyErr = L.polygon(RING_ERR, {
      color: '#bbb', weight: 1.5, opacity: .85, dashArray: '4 4', fill: false})
      .bindTooltip('若再转换一次');
  polyOk.addTo(MAP); polyErr.addTo(MAP);

  function useWrong() { return el.querySelector('#d-err').checked; }
  function activeRing() { return useWrong() ? RING_ERR : RING_OK; }

  function paintBoundary() {
    var bad = useWrong();
    polyOk.setStyle(bad
      ? {color: '#bbb', weight: 1.5, opacity: .85, dashArray: '4 4', fill: false}
      : {color: '#1668dc', weight: 3, opacity: .95, dashArray: null,
         fill: true, fillOpacity: .06});
    polyErr.setStyle(bad
      ? {color: '#fa8c16', weight: 3, opacity: .95, dashArray: null}
      : {color: '#bbb', weight: 1.5, opacity: .85, dashArray: '4 4', fill: false});
  }

  // ------------------------------------------------------------ grid render
  // Canvas keeps ~20k circle markers responsive.
  var REND = L.canvas({padding: 0.5});
  var layer = L.layerGroup().addTo(MAP);
  var stat = {n: 0, dep: 0, arr: 0};

  function pointInRing(lat, lon, ring) {
    var inside = false;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      var yi = ring[i][0], xi = ring[i][1];
      var yj = ring[j][0], xj = ring[j][1];
      if (((yi > lat) !== (yj > lat)) &&
          (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }

  function colour(dep, net) {
    if (el.querySelector('#p-net').checked) {
      var s = Math.max(-60, Math.min(60, net));
      return s === 0 ? '#8c8c8c' : (s > 0 ? '#d4380d' : '#0958d9');
    }
    if (dep <= 0) return '#bfbfbf';
    if (dep < 20) return '#bae0ff';
    if (dep < 80) return '#69b1ff';
    if (dep < 250) return '#faad14';
    if (dep < 800) return '#fa541c';
    return '#a8071a';
  }

  function render() {
    layer.clearLayers();
    var thr = +el.querySelector('#p-th').value;
    var bad = useWrong();
    var ring = activeRing();
    var onlyIn = el.querySelector('#p-inside').checked;
    stat = {n: 0, dep: 0, arr: 0};
    el.querySelector('#thv').textContent = thr;

    PTS.forEach(function (p) {
      var lat = bad ? p[2] : p[0], lon = bad ? p[3] : p[1];
      var dep = p[4], arr = p[5], net = dep - arr;
      if (onlyIn && !pointInRing(lat, lon, ring)) return;
      if (dep < thr) return;
      stat.n++; stat.dep += dep; stat.arr += arr;
      L.circleMarker([lat, lon], {
        renderer: REND, radius: Math.min(9, 2.2 + Math.log10(1 + dep) * 1.3),
        color: '#000', weight: 0.3, opacity: 0.35,
        fillColor: colour(dep, net), fillOpacity: 0.82
      }).bindTooltip(dep + ' 出发 / ' + arr + ' 到达<br>净 ' + net +
                     '<br>' + lat.toFixed(5) + ', ' + lon.toFixed(5),
                     {direction: 'top'}).addTo(layer);
    });
    el.querySelector('#n-in').textContent = stat.n.toLocaleString();
    el.querySelector('#n-da').textContent =
      stat.dep.toLocaleString() + ' / ' + stat.arr.toLocaleString();
    el.querySelector('#n-net').textContent =
      (stat.dep - stat.arr > 0 ? '+' : '') + (stat.dep - stat.arr).toLocaleString();
  }

  ['d-ok', 'd-err'].forEach(function (id) {
    el.querySelector('#' + id).onchange = function () { paintBoundary(); render(); };
  });
  ['p-inside', 'p-net'].forEach(function (id) {
    el.querySelector('#' + id).onchange = render;
  });
  el.querySelector('#p-th').oninput = render;

  paintBoundary();
  render();
});
</script>
"""


def render_png(ring, pts, path: Path, zoom: int = 16, min_dep: int = 15):
    """Static companion to the interactive map, for slides.

    Same basemap and same overlay as `region_map.html`, rasterised: Amap tiles
    are fetched and mosaicked, then geohash cells (sized and coloured by
    departure volume) and the boundary ring are drawn on top.  The HTML is the
    tool for checking the boundary; this is the thing you put on a slide.
    """
    import io
    import urllib.request

    from PIL import Image, ImageDraw

    def lon2x(lon):
        return (lon + 180.0) / 360.0 * (2 ** zoom) * 256

    def lat2y(lat):
        s = math.sin(math.radians(lat))
        return (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * (2 ** zoom) * 256

    def fetch(tx, ty):
        for sub in AMAP_SUBDOMAINS:
            url = AMAP_TILES.format(s=sub, x=tx, y=ty, z=zoom)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=25).read()
                if len(data) > 200:
                    return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                continue
        return None

    pad = 0.005
    la0, lo0, la1, lo1 = R.ring_bbox(ring, pad)
    x0, x1 = int(lon2x(lo0) // 256), int(lon2x(lo1) // 256)
    y0, y1 = int(lat2y(la1) // 256), int(lat2y(la0) // 256)
    img = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), "white")
    got = 0
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            tile = fetch(tx, ty)
            if tile:
                img.paste(tile, ((tx - x0) * 256, (ty - y0) * 256))
                got += 1
    ox, oy = x0 * 256, y0 * 256

    d = ImageDraw.Draw(img, "RGBA")
    drawn = 0
    for lat, lon, _slat, _slon, dep, _arr in pts:
        if dep < min_dep:
            continue
        drawn += 1
        x, y = lon2x(lon) - ox, lat2y(lat) - oy
        r = min(7.0, 1.6 + math.log10(1 + dep))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(210, 20, 20, 110))

    ring_xy = [(lon2x(lon) - ox, lat2y(lat) - oy) for lat, lon in ring]
    d.line(ring_xy + [ring_xy[0]], fill=(0, 60, 255, 255), width=4)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"  tiles {got}/{(x1 - x0 + 1) * (y1 - y0 + 1)} · cells drawn {drawn:,} "
          f"(dep >= {min_dep}) · {img.size[0]}x{img.size[1]}")
    return path


def build_html(ring, ring_err, pts, center, zoom, offset_m, n_disagree, n_union) -> str:
    import folium
    from branca.element import Element

    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(tiles=AMAP_TILES, attr="© 高德地图 AutoNavi", name="高德路网",
                     subdomains=AMAP_SUBDOMAINS, max_zoom=19).add_to(m)
    m.get_root().header.add_child(Element(f"<style>{PANEL_CSS}</style>"))

    js = (JS
          .replace("__MAP__", m.get_name())
          .replace("__PTS__", json.dumps(pts, separators=(",", ":")))
          .replace("__RING_OK__", json.dumps(ring, separators=(",", ":")))
          .replace("__RING_ERR__", json.dumps([[round(a, 6), round(b, 6)]
                                               for a, b in ring_err],
                                              separators=(",", ":")))
          .replace("__PANEL__", json.dumps(panel_html(len(pts), R.ring_area_km2(ring),
                                                      offset_m, n_disagree, n_union))))
    m.get_root().html.add_child(Element(js))
    return m.get_root().render()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the map in a browser")
    ap.add_argument("--png", nargs="?", const=str(OUT_PNG), default=None,
                    metavar="PATH", help="also write a static PNG (for slides)")
    args = ap.parse_args(argv)

    for p in (XLSX, CELLS):
        if not p.exists():
            print(f"FATAL: {p} not found.", file=sys.stderr)
            return 2

    ring, by_lon, by_lat = R.load_boundary(XLSX)
    lats = [p[0] for p in ring]
    lons = [p[1] for p in ring]
    bbox = (min(lats) - PAD_DEG, min(lons) - PAD_DEG,
            max(lats) + PAD_DEG, max(lons) + PAD_DEG)

    print(f"boundary : {len(ring)} vertices | lat {min(lats):.3f}..{max(lats):.3f}"
          f" | lon {min(lons):.3f}..{max(lons):.3f}")
    print(f"area     : {R.ring_area_km2(ring):.3f} km2")

    _, _, only_a, only_b = R.cross_check(by_lon, by_lat)
    n_dis = len(only_a) + len(only_b)
    n_union = len(R._spell_out(by_lon, "Longitude", "Min_latitude", "Max_latitude", False)
                  | R._spell_out(by_lat, "Latitude", "Min_longitude", "Max_longitude", True))
    print(f"scanline self-check: the two tables disagree on {n_dis} of {n_union} "
          f"cells ({100 * n_dis / n_union:.2f}%) -- the per-latitude table fills "
          f"the south-east notch")

    pts = load_points(CELLS, bbox)
    print(f"grid pts : {len(pts):,} geohash cells with trips in the padded box")

    ring_err = [wgs84_to_gcj02(la, lo) for la, lo in ring]
    dlat = float(np.mean([g[0] - r[0] for g, r in zip(ring_err, ring)]))
    dlon = float(np.mean([g[1] - r[1] for g, r in zip(ring_err, ring)]))
    offset_m = math.hypot(dlat * M_PER_DEG_LAT, dlon * M_PER_DEG_LON)
    print(f"the error both variants would introduce: {offset_m:.0f} m "
          f"({dlat * M_PER_DEG_LAT:+.0f} m N, {dlon * M_PER_DEG_LON:+.0f} m E)")

    clat, clon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    html = build_html(ring, ring_err, pts, (clat, clon), MAP_ZOOM,
                      offset_m, n_dis, n_union)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.0f} KB)")

    if args.png:
        png = Path(args.png)
        render_png(ring, pts, png)
        print(f"wrote {png}  ({png.stat().st_size / 1024:.0f} KB)")

    if args.open:
        webbrowser.open(OUT_HTML.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
