import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes

from matplotlib import pyplot as plt
from matplotlib import colors
from parse import search

from skimage.filters import roberts

def get_fid(sample_name):
    return search('sample_{}_', sample_name)[0]

def read_stratum_images(stratum, base_path, blacklist=None, debug=False):
    img_paths = [x for x in os.listdir(base_path) if stratum in x and x.endswith('.tif')]
    imgs = {}
    
    for name in img_paths:
        if blacklist and (get_fid(name) in blacklist):
            continue
        if debug:
            print(f'Reading image {name}')
        with rasterio.open(os.path.join(base_path, name), 'r') as src:
            imgs[name] = {
                'img': np.dstack(src.read()),
                'masks': src.read_masks(),
                'props': {
                    'affine': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'res': src.res
                }
            }
    
    return imgs

def plot_segments(imgs_8, segments, sample):
    img_8 = imgs_8[sample]
    cmap = colors.ListedColormap(np.random.rand(len(np.unique(segments[sample])), 3))

    edges = roberts(segments[sample])
    edges[edges == 0] = np.nan

    plt.figure(figsize=(15, 15))

    plt.imshow(img_8[:,:,:3])
    plt.imshow(edges, cmap='autumn')

def tiff_export(segment, props, output_name, debug=False):
    seg_to_save = segment.astype(np.int16)
    if debug:
        print(f'Exporting to {output_name}...')
    with rasterio.open(
        output_name, 'w',
        driver='GTiff', count=1,
        dtype=np.int16,
        height=props['height'],
        width=props['width'],
        crs=props['crs'],
        transform=props['affine']
    ) as dst:
        dst.write(seg_to_save, 1)

def gpkg_export(segment, props, output_name, debug=False):
    if debug:
        print(f'Exporting to {output_name}...')
    results = (
        {'properties': {'segment_id': int(v), 'class': ''}, 'geometry': s}
        for i, (s, v)
        in enumerate(
            shapes(segment.astype('int16'), mask=None, transform=props['affine'])
        )
    )
    geom = list(results)
    segments_gdf = gpd.GeoDataFrame.from_features(geom)
    segments_gdf.crs = {'init': 'epsg:4326'}
    segments_gdf.to_file(output_name, driver='GPKG')

import rasterio
import numpy as np
import geopandas as gpd

from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape

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
                shapes(footprint.astype('uint8'), mask=None, transform=src.transform)
            )
        )
        geom = list(results)

        # Convert list 
        shapely_mask = shape(
            [x for x in geom if x['properties']['segment_id'] == 1][0]['geometry']
        ).simplify(0.01).buffer(-d)

        shape_mask = gpd.GeoSeries([shapely_mask]).__geo_interface__['features'][0]['geometry']

        masked = mask(
            raster=src, shapes=[shape_mask], nodata=0, crop=True
        )[0].data[0]
    
    props.update(dtype='uint16', count=1, compress='lzw', nodata=0, transform=props['affine'])

    with rasterio.open(save_path, 'w', **props) as dst:
        dst.write(masked, 1)