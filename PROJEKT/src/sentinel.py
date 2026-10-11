"""Sentinel-2: wskaźniki dla każdego roku, GeoTIFF, XYZ i mapy.
Nie wymaga QGIS ani programów gdal_translate/gdal2tiles w PATH.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import transform_geom, transform_bounds, reproject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_BANDS = (2, 3, 4, 8, 11)
INDEX_BANDS = {'NDVI': (8, 4), 'NDBI': (11, 8), 'MNDWI': (3, 11)}
CMAPS = {'NDVI': 'RdYlGn', 'NDBI': 'PuOr_r', 'MNDWI': 'BrBG'}


def find_band_files(year_dir: Path) -> dict:
    files = {}
    pattern = re.compile(r'_B(02|03|04|08|11)_\(Raw\)\.tiff$', re.IGNORECASE)
    for path in year_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if match:
            band = int(match.group(1))
            if band in files:
                raise ValueError(f'Więcej niż jeden plik B{band:02d} w {year_dir}.')
            files[band] = path
    missing = sorted(set(REQUIRED_BANDS) - files.keys())
    if missing:
        labels = ', '.join(f'B{band:02d}' for band in missing)
        raise ValueError(f'Brak wymaganych pasm {labels} w {year_dir}.')
    return files


def available_years(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Nie znaleziono katalogu danych Sentinel-2: {data_dir}')
    years = sorted((p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if not years:
        raise ValueError(f'Nie znaleziono katalogów z latami w {data_dir}.')
    return [year.name for year in years]


def load_scene(data_dir: Path, year: str, aoi_path: Path = ROOT / 'aoi.geojson') -> dict:
    """Czyta pasma Sentinel-2 na siatce B04 i stosuje maskę granicy oraz NoData."""
    year_dir = Path(data_dir) / str(year)
    band_files = find_band_files(year_dir)
    aoi = json.loads(Path(aoi_path).read_text(encoding='utf-8'))
    raw = {}
    with rasterio.open(band_files[4]) as reference:
        if reference.count != 1:
            raise ValueError(f'Oczekiwano rastra jednopasmowego: {band_files[4]}')
        profile = reference.profile.copy()
        reference_transform = reference.transform
        reference_crs = reference.crs
        reference_shape = (reference.height, reference.width)
    for band in REQUIRED_BANDS:
        with rasterio.open(band_files[band]) as src:
            if src.count != 1:
                raise ValueError(f'Oczekiwano rastra jednopasmowego: {band_files[band]}')
            destination = np.zeros(reference_shape, dtype='float32')
            if (src.crs == reference_crs and src.transform == reference_transform
                    and (src.height, src.width) == reference_shape):
                destination[:] = src.read(1).astype('float32')
            else:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata if src.nodata is not None else 0,
                    dst_transform=reference_transform,
                    dst_crs=reference_crs,
                    dst_nodata=0,
                    resampling=Resampling.bilinear,
                )
            raw[band] = destination
    geoms = [transform_geom('EPSG:4326', profile['crs'], f['geometry']) for f in aoi['features']]
    inside = geometry_mask(geoms, out_shape=reference_shape, transform=profile['transform'], invert=True)
    if not inside.any():
        raise ValueError('Granica nie przecina sceny. Sprawdź CRS i zasięg.')
    valid = inside.copy()
    for band in REQUIRED_BANDS:
        valid &= np.isfinite(raw[band]) & (raw[band] > 0)
    if not valid.any():
        raise ValueError('Po maskowaniu nie pozostały ważne piksele.')
    bands = {band: np.where(valid, raw[band], np.nan) for band in REQUIRED_BANDS}
    first_raster = band_files[REQUIRED_BANDS[0]].name
    date_match = re.match(r'\d{4}-\d{2}-\d{2}', first_raster)
    if not date_match:
        raise ValueError(f'Nie można odczytać daty z nazwy rastra: {first_raster}')
    metadata = {'product': 'Sentinel-2 L2A', 'date': date_match.group(0), 'year': str(year)}
    return {'metadata': metadata, 'profile': profile, 'raw': raw, 'inside': inside,
            'valid': valid, 'bands': bands, 'aoi': aoi}


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zachowuje zera i wartości ujemne. Zerowy mianownik daje NaN."""
    a, b = np.asarray(a, dtype='float32'), np.asarray(b, dtype='float32')
    out = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype='float32')
    denominator = a + b
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > 1e-6)
    np.divide(a - b, denominator, out=out, where=valid)
    return out


def calculate_indices(bands: dict) -> dict:
    return {name: normalized_difference(bands[a], bands[b]) for name, (a, b) in INDEX_BANDS.items()}


def stretch_rgb(bands: dict) -> np.ndarray:
    """RGB B04/B03/B02: osobne rozciągnięcie percentyli wyłącznie do prezentacji."""
    channels = []
    for b in (4, 3, 2):
        a = bands[b]
        lo, hi = np.nanpercentile(a, (2, 98))
        scaled = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        channels.append((np.nan_to_num(scaled) * 255).astype('uint8'))
    alpha = (np.isfinite(bands[4]) * 255).astype('uint8')
    return np.dstack([*channels, alpha])


def index_rgba(array: np.ndarray, name: str) -> np.ndarray:
    rgba = matplotlib.colormaps[CMAPS[name]](np.clip((array + 1) / 2, 0, 1), bytes=True)
    rgba[..., 3] = np.isfinite(array) * 255
    return rgba


def write_geotiff(path: Path, data: np.ndarray, profile: dict, rgba=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    if rgba:
        data = np.moveaxis(data, -1, 0)
        p.update(count=4, dtype='uint8', nodata=None, photometric='RGB')
    else:
        data = np.where(np.isfinite(data), data, -9999).astype('float32')[None]
        p.update(count=1, dtype='float32', nodata=-9999)
        p.pop('photometric', None)
    p.pop('blockxsize', None)
    p.pop('blockysize', None)
    p.update(driver='GTiff', compress='deflate', tiled=True)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(data)
        if rgba:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        else:
            dst.set_band_description(1, path.stem)
            dst.update_tags(meaning='normalized difference; no numeric clipping; NoData=-9999')


def xyz_tiles(rgba_tif: Path, target: Path, minzoom=9, maxzoom=12) -> int:
    """Kafelki XYZ 256 px w EPSG:3857. Oś y rośnie na południe (tms=False)."""
    if not 0 <= minzoom <= maxzoom <= 15:
        raise ValueError('W ćwiczeniu dozwolone 0 <= minzoom <= maxzoom <= 15.')
    radius = 20037508.342789244
    count = 0
    with rasterio.open(rgba_tif) as src:
        west, south, east, north = transform_bounds(src.crs, 'EPSG:3857', *src.bounds, densify_pts=21)
        for z in range(minzoom, maxzoom + 1):
            n = 2 ** z
            span = 2 * radius / n
            x0, x1 = max(0, math.floor((west + radius) / span)), min(n-1, math.floor((east + radius) / span))
            y0, y1 = max(0, math.floor((radius - north) / span)), min(n-1, math.floor((radius - south) / span))
            for x in range(x0, x1 + 1):
                folder = target / str(z) / str(x)
                folder.mkdir(parents=True, exist_ok=True)
                for y in range(y0, y1 + 1):
                    dst = np.zeros((4, 256, 256), dtype='uint8')
                    bounds = (x*span-radius, radius-(y+1)*span, (x+1)*span-radius, radius-y*span)
                    reproject(source=rasterio.band(src, [1,2,3,4]), destination=dst,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=from_bounds(*bounds,256,256), dst_crs='EPSG:3857',
                              resampling=Resampling.nearest, src_alpha=4, dst_alpha=4)
                    Image.fromarray(np.moveaxis(dst,0,-1)).save(folder / f'{y}.png')
                    count += 1
    return count


def make_figures(scene: dict, indices: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.titlesize':15,'figure.facecolor':'white'})
    extent = array_bounds(scene['profile']['height'], scene['profile']['width'], scene['profile']['transform'])
    west,south,east,north=extent
    extent_km=[west/1000,east/1000,south/1000,north/1000]
    def plot(a, title, name, cmap=None, limits=None, label=None):
        fig,ax=plt.subplots(figsize=(8.8,6.2),layout='constrained')
        im=ax.imshow(a,extent=extent_km,cmap=cmap,**({'vmin':limits[0],'vmax':limits[1]} if limits else {}))
        ax.set(title=title,xlabel='Easting [km], EPSG:32634',ylabel='Northing [km]')
        if cmap: fig.colorbar(im,ax=ax,shrink=.8,label=label or '')
        fig.savefig(out/name,dpi=155)
        plt.close(fig)
    date=scene['metadata']['date']
    plot(stretch_rgb(scene['bands']), f'Sentinel-2 • RGB (B04/B03/B02) • {date}','rgb.png')
    plot(np.where(scene['raw'][8] > 0, scene['raw'][8], np.nan),
         'Kanał B08 przed maskowaniem','band08_dn.png','gray',label='Wartość źródłowa')
    plot(scene['bands'][8],'B08 po masce NoData i granicy','band08_masked.png',
         'gray',label='Wartość źródłowa')
    quality=np.where(~scene['inside'],np.nan,np.where(scene['valid'],1,0))
    plot(quality,'Maska: 1 = piksel z danymi, 0 = NoData','quality.png','RdYlGn',(0,1),'Ważność')
    for name,a in indices.items():
        plot(a,f'{name} • {date}',name.lower()+'.png',CMAPS[name],(-1,1),'Wartość wskaźnika')
    fig,axs=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    for ax,(name,a) in zip(axs,indices.items()):
        ax.hist(a[np.isfinite(a)],bins=60,range=(-1,1),color='#167f83')
        ax.set(title=name,xlabel='Wartość',ylabel='Liczba pikseli')
    fig.savefig(out/'histograms.png',dpi=155); plt.close(fig)


def export_results(scene: dict, indices: dict, output_dir: Path, minzoom=9, maxzoom=12) -> dict:
    start=time.perf_counter()
    output_dir=Path(output_dir)
    raster_dir,figure_dir,site=output_dir/'rasters',output_dir/'figures',output_dir/'site'
    for p in [raster_dir,figure_dir,site]: p.mkdir(parents=True,exist_ok=True)
    bounds=array_bounds(scene['profile']['height'],scene['profile']['width'],scene['profile']['transform'])
    w,s,e,n=transform_bounds(scene['profile']['crs'],'EPSG:4326',*bounds)
    ramps={name:['#'+''.join(f'{v:02x}' for v in matplotlib.colormaps[CMAPS[name]](float(t),bytes=True)[:3]) for t in np.linspace(0,1,9)] for name in indices}
    summary={'year':scene['metadata']['year'],'product':scene['metadata']['product'],'date':scene['metadata']['date'],
             'shape':[scene['profile']['height'],scene['profile']['width']],
             'crs':str(scene['profile']['crs']),'inside_pixels':int(scene['inside'].sum()),
             'valid_pixels':int(scene['valid'].sum()),'indices':{},'tiles':{},
             'minzoom':minzoom,'maxzoom':maxzoom,'bounds':[[s,w],[n,e]],'ramps':ramps}
    for name,a in indices.items():
        finite=a[np.isfinite(a)]
        summary['indices'][name]={'count':int(finite.size),'min':float(finite.min()),'max':float(finite.max()),
            'mean':float(finite.mean()),'median':float(np.median(finite)),
            'outside_minus1_plus1':int((np.abs(finite)>1).sum())}
        write_geotiff(raster_dir/f'{name}.tif',a,scene['profile'])
    layers={'RGB':stretch_rgb(scene['bands']),**{name:index_rgba(a,name) for name,a in indices.items()}}
    for name,rgba in layers.items():
        p=raster_dir/f'{name}_display.tif'
        write_geotiff(p,rgba,scene['profile'],rgba=True)
        # Zakres zoomów jest zapisany w mapie; istniejące pliki innych zoomów są ignorowane.
        summary['tiles'][name]=xyz_tiles(p,site/'tiles'/name,minzoom,maxzoom)
    make_figures(scene,indices,figure_dir)
    shutil.copytree(Path(__file__).parent/'vendor',site/'vendor',dirs_exist_ok=True)
    (site/'aoi.geojson').write_text(json.dumps(scene['aoi']),encoding='utf-8')
    summary['export_seconds']=round(time.perf_counter()-start,2)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (site/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def write_annual_index(output_dir: Path, summaries: list[dict]) -> dict:
    output_dir = Path(output_dir)
    summaries_by_year = {}
    for year_dir in output_dir.iterdir() if output_dir.exists() else ():
        summary_path = year_dir / 'summary.json'
        if year_dir.is_dir() and year_dir.name.isdigit() and summary_path.is_file():
            stored = json.loads(summary_path.read_text(encoding='utf-8'))
            if (stored.get('year') != year_dir.name
                    or stored.get('product') != 'Sentinel-2 L2A'
                    or 'bounds' not in stored or 'ramps' not in stored):
                raise ValueError(f'Niezgodny rok w podsumowaniu {summary_path}.')
            summaries_by_year[year_dir.name] = stored
    summaries_by_year.update({summary['year']: summary for summary in summaries})
    summaries = [summaries_by_year[year] for year in sorted(summaries_by_year, key=int)]
    site = Path(output_dir) / 'site'
    site.mkdir(parents=True, exist_ok=True)
    template=(Path(__file__).parent/'map_template.html').read_text(encoding='utf-8')
    for selected_year, page_dir, base_pattern in [
        (None, site, '../{year}/site'),
        *((summary['year'], output_dir/summary['year']/'site', '../../{year}/site')
          for summary in summaries),
    ]:
        scenes=[{
            'year':summary['year'],'date':summary['date'],
            'base':base_pattern.format(year=summary['year']),
            'url':base_pattern.format(year=summary['year'])+'/index.html',
            'bounds':summary['bounds'],'ramps':summary['ramps'],
            'minzoom':summary['minzoom'],'maxzoom':summary['maxzoom'],
        } for summary in summaries]
        config={'scenes':scenes,'selectedYear':selected_year or summaries[0]['year']}
        page_dir.mkdir(parents=True,exist_ok=True)
        (page_dir/'index.html').write_text(
            template.replace('__MAP_CONFIG__',json.dumps(config,ensure_ascii=False)),
            encoding='utf-8')
        shutil.copytree(Path(__file__).parent/'vendor',page_dir/'vendor',dirs_exist_ok=True)
    summary = {'product': 'Sentinel-2 L2A', 'years': summaries}
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    (Path(output_dir) / 'summary.json').write_text(text, encoding='utf-8')
    (site / 'summary.json').write_text(text, encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data'/'sentinel')
    parser.add_argument('--aoi',type=Path,default=ROOT/'aoi.geojson')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--year',help='Przetwarzaj tylko wskazany rok; domyślnie wszystkie dostępne lata.')
    parser.add_argument('--minzoom',type=int,default=9)
    parser.add_argument('--maxzoom',type=int,default=12)
    args=parser.parse_args()
    years=available_years(args.data)
    if args.year is not None:
        if args.year not in years:
            raise ValueError(f'Rok {args.year} nie jest dostępny. Dostępne lata: {", ".join(years)}')
        years=[args.year]
    results=[]
    for year in years:
        scene=load_scene(args.data,year,args.aoi)
        indices=calculate_indices(scene['bands'])
        results.append(export_results(
            scene,indices,args.output/year,args.minzoom,args.maxzoom))
        del scene, indices
    result=write_annual_index(args.output,results)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Podgląd: python -m http.server 8000 --bind 127.0.0.1 --directory '+str(args.output/'site'))

if __name__=='__main__': main()
