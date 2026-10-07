"""
Pull paired thermal (Band 10) and RGB (Bands 4/3/2) imagery from Landsat 8/9
Collection 2 Level-2 via Google Earth Engine, and export to Google Drive as
GeoTIFFs.

Run this in Google Colab after:
    !pip install -q earthengine-api
and authenticating (handled automatically by initialize_ee() below).
"""

import sys
import os
import ee

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import THERMAL_BAND, RGB_BANDS, CLOUD_COVER_MAX, SCENE_CENTERS, DATE_RANGE

EXPORT_FOLDER = "ir2rgb_raw"      # Drive folder the exports land in
EXPORT_SCALE = 30                 # meters/pixel, matches Landsat native res
EXPORT_SIZE_M = 256 * 30 * 4      # ~30km box per scene, enough to tile into many 256px patches


def initialize_ee(project_id=None):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def get_best_scene(lon, lat, date_range):
    """Return the least-cloudy Landsat 8/9 L2 scene covering (lon, lat)."""
    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(point)
        .filterDate(date_range[0], date_range[1])
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX))
        .sort("CLOUD_COVER")
    )
    return collection.first()


def scale_bands(image):
    """Apply Landsat Collection 2 Level-2 scale factors to get physical units."""
    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)  # Kelvin
    return image.addBands(optical, None, True).addBands(thermal, None, True)


def export_scene(scene_dict, date_range):
    name = scene_dict["name"]
    lon, lat = scene_dict["lon"], scene_dict["lat"]

    image = ee.Image(get_best_scene(lon, lat, date_range))
    image = scale_bands(image)

    region = ee.Geometry.Point([lon, lat]).buffer(EXPORT_SIZE_M / 2).bounds()
    rgb_thermal = image.select(RGB_BANDS + [THERMAL_BAND])

    task = ee.batch.Export.image.toDrive(
        image=rgb_thermal,
        description=f"ir2rgb_{name}",
        folder=EXPORT_FOLDER,
        fileNamePrefix=name,
        region=region,
        scale=EXPORT_SCALE,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    print(f"Started export task for scene '{name}' -> Drive/{EXPORT_FOLDER}/{name}.tif")
    return task


def main():
    initialize_ee()
    tasks = [export_scene(scene, DATE_RANGE) for scene in SCENE_CENTERS]
    print(f"\nSubmitted {len(tasks)} export tasks.")
    print("Check progress at https://code.earthengine.google.com/tasks")
    print(f"Exported files will land in your Google Drive under '{EXPORT_FOLDER}/'.")


if __name__ == "__main__":
    main()
