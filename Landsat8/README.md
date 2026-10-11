# Landsat 8 — mapa NDVI

Otwórz [`index.html`](index.html) w przeglądarce. Strona pokazuje podkład OpenStreetMap z jedną opcjonalną warstwą NDVI, którą można włączyć lub wyłączyć polem wyboru.

Warstwa jest przygotowywana na podstawie zapisanego wykresu NDVI w `Landsat_part2.ipynb`; jej położenie geograficzne pochodzi z georeferencji rastra w `clipped/LC_clipped_band5.tif`. Generator odświeża `index.html`, `ndvi-overlay.png` i lokalne pliki Leaflet:

```text
node build_site.js
```

OpenStreetMap wymaga połączenia z internetem. Pliki biblioteki Leaflet i warstwa NDVI są przechowywane lokalnie.
