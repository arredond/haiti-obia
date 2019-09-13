import os

import rasterio
import numpy as np
import geopandas as gpd

from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape

from urllib.parse import urljoin
from urllib.request import urlretrieve

def fetch_image(img_name, img_type, save_folder, base_path="ftp://dirsftp.cis.rit.edu/Haiti/2010-01-21-haiti/"):
    if img_type == 'vnir':
        download_name = img_name.replace('.tif', '_flatfield.tif')
    elif img_type == 'swir':
        download_name = img_name

    full_path = urljoin(urljoin(base_path, f'{img_type}-ortho/'), download_name)

    return urlretrieve(full_path, os.path.join(save_folder, img_name))

def get_image_metadata(input_path):
    with rasterio.open(input_path, 'r') as src:
        return src.width, src.height, list(src.bounds)

def remove_borders(input_path, save_path, d=0.00002):
    with rasterio.open(input_path, 'r') as src:
        img = src.read()[0]

        footprint = img.copy()
        footprint[footprint != 0] = 1

        props = src.profile

        results = (
            {'properties': {'segment_id': int(v), 'class': ''}, 'geometry': s}
            for i, (s, v)
            in enumerate(
                shapes(footprint.astype('uint8'), mask=None, transform=src.affine)
            )
        )
        geom = list(results)

        # Convert list 
        shapely_mask = shape(
            [x for x in geom if x['properties']['segment_id'] == 1][0]['geometry']
        ).simplify(0.002).buffer(-d)

        shape_mask = gpd.GeoSeries([shapely_mask]).__geo_interface__['features'][0]['geometry']

        masked = mask(raster=src, shapes=[shape_mask], nodata=0)[0].data[0]
    
    props.update(dtype='uint16', count=1, compress='lzw', nodata=0, transform=props['affine'])

    with rasterio.open(save_path, 'w', **props) as dst:
        dst.write(masked, 1)