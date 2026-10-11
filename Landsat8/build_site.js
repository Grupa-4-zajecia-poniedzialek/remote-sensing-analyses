const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

const root = __dirname;
const outputDirectory = path.join(root, "vendor");
const notebookPath = path.join(root, "Landsat_part2.ipynb");
const rasterPath = path.join(root, "clipped", "LC_clipped_band5.tif");
const aoiShapePath = path.join(root, "vector", "krakow_krakowskie.shp");
const leafletDirectory = path.join(
  root,
  "..",
  "LABOLATORIUM",
  "outputs",
  "site",
  "vendor",
);

function notebookImages() {
  const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
  const cell = notebook.cells.find((item) =>
    textSource(item.source).includes("calculate_index('ndvi'"),
  );
  const images = (cell?.outputs ?? [])
    .filter((output) => output.data?.["image/png"])
    .map((output) => {
      const image = output.data["image/png"];
      return Buffer.from(Array.isArray(image) ? image.join("") : image, "base64");
    });
  if (images.length < 3) {
    throw new Error(
      "W Landsat_part2.ipynb nie znaleziono zapisanych wykresów NDVI, NDWI i NDBI.",
    );
  }
  return images;
}

function textSource(source) {
  return Array.isArray(source) ? source.join("") : String(source ?? "");
}

function readPng(buffer) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (!buffer.subarray(0, 8).equals(signature)) {
    throw new Error("Zapisany wynik NDVI nie jest prawidłowym plikiem PNG.");
  }

  let width;
  let height;
  let colorType;
  let bitDepth;
  let interlace;
  const imageData = [];
  for (let offset = 8; offset < buffer.length; ) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      interlace = data[12];
    } else if (type === "IDAT") {
      imageData.push(data);
    } else if (type === "IEND") {
      break;
    }
    offset += length + 12;
  }

  const channels = colorType === 6 ? 4 : colorType === 2 ? 3 : 0;
  if (!width || !height || bitDepth !== 8 || !channels || interlace !== 0) {
    throw new Error(
      `Nieobsługiwany PNG NDVI (typ=${colorType}, głębia=${bitDepth}, przeplot=${interlace}).`,
    );
  }

  const stride = width * channels;
  const filtered = zlib.inflateSync(Buffer.concat(imageData));
  const pixels = Buffer.alloc(height * stride);
  let inputOffset = 0;
  for (let row = 0; row < height; row += 1) {
    const filter = filtered[inputOffset++];
    const rowOffset = row * stride;
    const previousOffset = rowOffset - stride;
    for (let column = 0; column < stride; column += 1) {
      const raw = filtered[inputOffset++];
      const left = column >= channels ? pixels[rowOffset + column - channels] : 0;
      const above = row > 0 ? pixels[previousOffset + column] : 0;
      const upperLeft =
        row > 0 && column >= channels
          ? pixels[previousOffset + column - channels]
          : 0;
      let predictor = 0;
      if (filter === 1) predictor = left;
      else if (filter === 2) predictor = above;
      else if (filter === 3) predictor = Math.floor((left + above) / 2);
      else if (filter === 4) {
        const base = left + above - upperLeft;
        const leftDistance = Math.abs(base - left);
        const aboveDistance = Math.abs(base - above);
        const upperLeftDistance = Math.abs(base - upperLeft);
        predictor =
          leftDistance <= aboveDistance && leftDistance <= upperLeftDistance
            ? left
            : aboveDistance <= upperLeftDistance
              ? above
              : upperLeft;
      } else if (filter !== 0) {
        throw new Error(`Nieobsługiwany filtr PNG: ${filter}.`);
      }
      pixels[rowOffset + column] = (raw + predictor) & 255;
    }
  }
  return { width, height, channels, pixels };
}

function cropRasterImage(image, rasterAspect) {
  const { width, height, channels, pixels } = image;
  const pixelCount = width * height;
  const candidate = new Uint8Array(pixelCount);
  const visited = new Uint8Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * channels;
    const red = pixels[offset];
    const green = pixels[offset + 1];
    const blue = pixels[offset + 2];
    const minimum = Math.min(red, green, blue);
    const maximum = Math.max(red, green, blue);
    const alpha = channels === 4 ? pixels[offset + 3] : 255;
    if (alpha > 0 && maximum - minimum > 24 && maximum < 245) {
      candidate[index] = 1;
    }
  }

  const queue = new Int32Array(pixelCount);
  let largest = null;
  for (let start = 0; start < pixelCount; start += 1) {
    if (!candidate[start] || visited[start]) continue;
    let readIndex = 0;
    let writeIndex = 1;
    queue[0] = start;
    visited[start] = 1;
    let minX = width;
    let maxX = -1;
    let minY = height;
    let maxY = -1;

    while (readIndex < writeIndex) {
      const index = queue[readIndex++];
      const x = index % width;
      const y = Math.floor(index / width);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nextX = x + dx;
          const nextY = y + dy;
          if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
            continue;
          }
          const next = nextY * width + nextX;
          if (candidate[next] && !visited[next]) {
            visited[next] = 1;
            queue[writeIndex++] = next;
          }
        }
      }
    }
    if (!largest || writeIndex > largest.size) {
      largest = { size: writeIndex, minX, maxX, minY, maxY };
    }
  }

  if (!largest) throw new Error("Nie udało się znaleźć obrazu NDVI na wykresie.");
  const cropWidth = largest.maxX - largest.minX + 1;
  const cropHeight = largest.maxY - largest.minY + 1;
  const cropAspect = cropWidth / cropHeight;
  if (
    largest.size < pixelCount * 0.05 ||
    Math.abs(cropAspect - rasterAspect) > 0.12
  ) {
    throw new Error(
      `Nie udało się wiarygodnie wyciąć rastra NDVI (proporcje ${cropAspect.toFixed(
        3,
      )}; oczekiwano ${rasterAspect.toFixed(3)}).`,
    );
  }

  const rgba = Buffer.alloc(cropWidth * cropHeight * 4);
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const sourceOffset =
        ((largest.minY + y) * width + largest.minX + x) * channels;
      const targetOffset = (y * cropWidth + x) * 4;
      const red = pixels[sourceOffset];
      const green = pixels[sourceOffset + 1];
      const blue = pixels[sourceOffset + 2];
      const sourceAlpha = channels === 4 ? pixels[sourceOffset + 3] : 255;
      rgba[targetOffset] = red;
      rgba[targetOffset + 1] = green;
      rgba[targetOffset + 2] = blue;
      rgba[targetOffset + 3] =
        sourceAlpha === 0 || (red > 238 && green > 238 && blue > 238)
          ? 0
          : sourceAlpha;
    }
  }
  return encodePng(cropWidth, cropHeight, rgba);
}

const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) {
    crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return crc >>> 0;
});

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const name = Buffer.from(type);
  const result = Buffer.alloc(12 + data.length);
  result.writeUInt32BE(data.length, 0);
  name.copy(result, 4);
  data.copy(result, 8);
  result.writeUInt32BE(crc32(Buffer.concat([name, data])), 8 + data.length);
  return result;
}

function encodePng(width, height, rgba) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  const scanlines = Buffer.alloc(height * (width * 4 + 1));
  for (let row = 0; row < height; row += 1) {
    const start = row * (width * 4 + 1);
    scanlines[start] = 0;
    rgba.copy(scanlines, start + 1, row * width * 4, (row + 1) * width * 4);
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header),
    chunk("IDAT", zlib.deflateSync(scanlines)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function readRasterExtent() {
  const tiff = fs.readFileSync(rasterPath);
  const byteOrder = tiff.toString("ascii", 0, 2);
  if (tiff.readUInt16LE(2) !== 42 || byteOrder !== "II") {
    throw new Error("Raster B5 musi być klasycznym GeoTIFF-em little-endian.");
  }
  const ifdOffset = tiff.readUInt32LE(4);
  const entryCount = tiff.readUInt16LE(ifdOffset);
  const tags = new Map();
  for (let index = 0; index < entryCount; index += 1) {
    const entry = ifdOffset + 2 + index * 12;
    const tag = tiff.readUInt16LE(entry);
    const type = tiff.readUInt16LE(entry + 2);
    const count = tiff.readUInt32LE(entry + 4);
    const typeSize = { 3: 2, 4: 4, 12: 8 }[type];
    if (!typeSize) continue;
    const valueOffset =
      count * typeSize <= 4 ? entry + 8 : tiff.readUInt32LE(entry + 8);
    const values = [];
    for (let valueIndex = 0; valueIndex < count; valueIndex += 1) {
      const position = valueOffset + valueIndex * typeSize;
      if (type === 3) values.push(tiff.readUInt16LE(position));
      else if (type === 4) values.push(tiff.readUInt32LE(position));
      else values.push(tiff.readDoubleLE(position));
    }
    tags.set(tag, values);
  }

  const width = tags.get(256)?.[0];
  const height = tags.get(257)?.[0];
  const scale = tags.get(33550);
  const tiePoint = tags.get(33922);
  const keys = tags.get(34735);
  if (
    !width ||
    !height ||
    !scale?.[0] ||
    !scale?.[1] ||
    !tiePoint ||
    !keys?.includes(32634)
  ) {
    throw new Error("GeoTIFF nie zawiera oczekiwanych rozmiarów, CRS i geotransformacji.");
  }
  const [rasterX, rasterY, , modelX, modelY] = tiePoint;
  return {
    width,
    height,
    minX: modelX - rasterX * scale[0],
    maxY: modelY + rasterY * scale[1],
    maxX: modelX + (width - rasterX) * scale[0],
    minY: modelY - (height - rasterY) * scale[1],
  };
}

function utm34ToLatLng(easting, northing) {
  const a = 6378137;
  const eccentricitySquared = 0.00669437999014;
  const secondEccentricitySquared =
    eccentricitySquared / (1 - eccentricitySquared);
  const k0 = 0.9996;
  const x = easting - 500000;
  const meridionalArc = northing / k0;
  const mu =
    meridionalArc /
    (a *
      (1 -
        eccentricitySquared / 4 -
        (3 * eccentricitySquared ** 2) / 64 -
        (5 * eccentricitySquared ** 3) / 256));
  const e1 =
    (1 - Math.sqrt(1 - eccentricitySquared)) /
    (1 + Math.sqrt(1 - eccentricitySquared));
  const phi1 =
    mu +
    ((3 * e1) / 2 - (27 * e1 ** 3) / 32) * Math.sin(2 * mu) +
    ((21 * e1 ** 2) / 16 - (55 * e1 ** 4) / 32) * Math.sin(4 * mu) +
    ((151 * e1 ** 3) / 96) * Math.sin(6 * mu) +
    ((1097 * e1 ** 4) / 512) * Math.sin(8 * mu);
  const sinPhi = Math.sin(phi1);
  const cosPhi = Math.cos(phi1);
  const tanPhi = Math.tan(phi1);
  const n1 = a / Math.sqrt(1 - eccentricitySquared * sinPhi ** 2);
  const t1 = tanPhi ** 2;
  const c1 = secondEccentricitySquared * cosPhi ** 2;
  const r1 =
    (a * (1 - eccentricitySquared)) /
    (1 - eccentricitySquared * sinPhi ** 2) ** 1.5;
  const d = x / (n1 * k0);
  const latitude =
    phi1 -
    ((n1 * tanPhi) / r1) *
      (d ** 2 / 2 -
        ((5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * secondEccentricitySquared) *
          d ** 4) /
          24 +
        ((61 +
          90 * t1 +
          298 * c1 +
          45 * t1 ** 2 -
          252 * secondEccentricitySquared -
          3 * c1 ** 2) *
          d ** 6) /
          720);
  const longitude =
    ((d -
      ((1 + 2 * t1 + c1) * d ** 3) / 6 +
      ((5 -
        2 * c1 +
        28 * t1 -
        3 * c1 ** 2 +
        8 * secondEccentricitySquared +
        24 * t1 ** 2) *
        d ** 5) /
        120) /
      cosPhi +
      (21 * Math.PI) / 180) *
    (180 / Math.PI);
  return [latitude * (180 / Math.PI), longitude];
}

function readAoiBoundary() {
  const shape = fs.readFileSync(aoiShapePath);
  if (shape.readInt32BE(0) !== 9994 || shape.readInt32LE(32) !== 5) {
    throw new Error("Plik granicy AOI nie jest prawidłowym shapefile typu Polygon.");
  }

  const recordOffset = 108;
  const partCount = shape.readInt32LE(recordOffset + 36);
  const pointCount = shape.readInt32LE(recordOffset + 40);
  if (partCount < 1 || pointCount < 4) {
    throw new Error("Shapefile AOI nie zawiera pierścienia granicy.");
  }

  const firstPart = shape.readInt32LE(recordOffset + 44);
  const nextPart = partCount > 1 ? shape.readInt32LE(recordOffset + 48) : pointCount;
  const pointsOffset = recordOffset + 44 + partCount * 4;
  const boundary = [];
  for (let index = firstPart; index < nextPart; index += 1) {
    const pointOffset = pointsOffset + index * 16;
    boundary.push(
      utm34ToLatLng(
        shape.readDoubleLE(pointOffset),
        shape.readDoubleLE(pointOffset + 8),
      ),
    );
  }
  if (boundary.length < 4) {
    throw new Error("Kontur AOI zawiera za mało punktów.");
  }
  const [firstLat, firstLng] = boundary[0];
  const [lastLat, lastLng] = boundary[boundary.length - 1];
  if (Math.abs(firstLat - lastLat) > 1e-10 || Math.abs(firstLng - lastLng) > 1e-10) {
    boundary.push(boundary[0]);
  }
  return boundary;
}

const extent = readRasterExtent();
const aoiBoundary = readAoiBoundary();
const indexImages = notebookImages();
const resultImages = indexImages.slice(0, 3).map((image) =>
  cropRasterImage(readPng(image), extent.width / extent.height),
);
fs.mkdirSync(outputDirectory, { recursive: true });
for (const [index, name] of ["ndvi", "ndwi", "ndbi"].entries()) {
  fs.writeFileSync(path.join(root, `${name}-overlay.png`), resultImages[index]);
}
for (const asset of ["leaflet.js", "leaflet.css", "LICENSE-Leaflet.txt"]) {
  fs.copyFileSync(path.join(leafletDirectory, asset), path.join(outputDirectory, asset));
}
const southWest = utm34ToLatLng(extent.minX, extent.minY);
const northEast = utm34ToLatLng(extent.maxX, extent.maxY);

const html = `<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Landsat 8 • indeksy</title><link rel="stylesheet" href="vendor/leaflet.css">
<style>
*{box-sizing:border-box}body{margin:0;font:16px system-ui,sans-serif;color:#173541;background:#f5f7f6}header{position:relative;z-index:1001;padding:16px 24px;background:#133e4a;color:white}h1{font-size:24px;margin:0 0 4px}header p{margin:0;font-size:14px;color:#d8e7e8}main{position:relative;height:calc(100vh - 89px);min-height:450px}#map{position:absolute;inset:0;background:#e6eceb}aside{position:absolute;z-index:1000;top:16px;right:16px;padding:12px 14px;border:1px solid #17354122;border-radius:8px;background:rgba(255,255,255,.94);box-shadow:0 2px 12px #17354130}.buttons{display:flex;gap:5px}button{padding:6px 10px;border:1px solid #b9c9c8;border-radius:5px;background:#fff;color:#173541;font:600 13px system-ui,sans-serif;cursor:pointer}button[aria-pressed="true"]{border-color:#176b57;background:#176b57;color:#fff}button:focus-visible{outline:2px solid #176b57;outline-offset:2px}.legend{margin-top:9px;padding:10px 11px;border:1px solid #17354118;border-radius:6px;background:#fff}.legend[hidden]{display:none}.legend h2{margin:0 0 8px;font-size:14px}.legend-name{margin:0 0 7px;font-size:12px;color:#45616b}.ramp{height:12px;border-radius:3px;background:linear-gradient(to right,#440154,#472d7b,#3b528b,#2c728e,#21918c,#27ad81,#5cc863,#aadc32,#fde725)}.ticks{display:flex;justify-content:space-between;margin-top:4px;font-size:11px;color:#45616b}@media(max-width:700px){header{padding:12px 16px}h1{font-size:20px}main{height:calc(100vh - 77px);min-height:350px}aside{top:12px;right:12px;padding:10px 12px}}
</style></head><body><header><h1>Landsat 8: Indeksy</h1><p>7 sierpnia 2013 • wskaźniki spektralne</p></header>
<main><div id="map" aria-label="Mapa OpenStreetMap z warstwami NDVI, NDWI i NDBI"></div><aside>
<div class="buttons" role="group" aria-label="Warstwa wynikowa">
<button type="button" data-index="NDVI" aria-pressed="true">NDVI</button>
<button type="button" data-index="NDWI" aria-pressed="false">NDWI</button>
<button type="button" data-index="NDBI" aria-pressed="false">NDBI</button>
</div>
<section class="legend" id="legend" aria-live="polite">
<h2>Legenda</h2>
<p class="legend-name" id="legend-name">NDVI</p>
<div class="ramp" aria-hidden="true"></div>
<div class="ticks" id="legend-ticks"><span>0</span><span>0,5</span><span>1</span></div>
</section>
</aside></main><script src="vendor/leaflet.js"></script><script>
const bounds=${JSON.stringify([southWest, northEast])};
const aoiBoundary=${JSON.stringify(aoiBoundary)};
const map=L.map('map',{maxZoom:17}).fitBounds(bounds);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
const overlays={
  NDVI:L.imageOverlay('ndvi-overlay.png',bounds,{opacity:0.82,alt:'Wynik NDVI z Landsat_part2.ipynb',interactive:false}),
  NDWI:L.imageOverlay('ndwi-overlay.png',bounds,{opacity:0.82,alt:'Wynik NDWI z Landsat_part2.ipynb',interactive:false}),
  NDBI:L.imageOverlay('ndbi-overlay.png',bounds,{opacity:0.82,alt:'Wynik NDBI z Landsat_part2.ipynb',interactive:false})
};
const aoi=L.polyline(aoiBoundary,{color:'#143843',weight:2,opacity:0.95,interactive:false});
let activeIndex=null;
const legendTicks={NDVI:['0','0,5','1'],NDWI:['−1','0','1'],NDBI:['−1','0','1']};
function setActiveIndex(name){if(activeIndex)map.removeLayer(overlays[activeIndex]);activeIndex=name;if(activeIndex)overlays[activeIndex].addTo(map);document.querySelectorAll('[data-index]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.index===activeIndex)));const legend=document.querySelector('#legend');legend.hidden=!activeIndex;if(activeIndex){document.querySelector('#legend-name').textContent=activeIndex;document.querySelector('#legend-ticks').innerHTML=legendTicks[activeIndex].map(value=>'<span>'+value+'</span>').join('');}}
aoi.addTo(map);
setActiveIndex('NDVI');
document.querySelectorAll('[data-index]').forEach(button=>button.addEventListener('click',()=>setActiveIndex(activeIndex===button.dataset.index?null:button.dataset.index)));
</script></body></html>
`;

fs.writeFileSync(path.join(root, "index.html"), html, "utf8");
console.log(
  `Utworzono index.html, 3 nakładki (${resultImages.map((image) => image.length).join(", ")} B) i kontur AOI (${aoiBoundary.length} punktów).`,
);
