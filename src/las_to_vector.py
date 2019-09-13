#!/usr/bin/env/python3

"""
Convert one or multiple LAS files to a vector format of choice. Options
include: gpkg (GeoPackage), shp (ESRI Shapefile) and json/geojson (GeoJSON).

Output name is the same as input (except the extension, of course) and is saved
in a directory called `las_to_vector`, created in the original LAS data's directory.

Mandatory arguments:
- --input_files (-i): One or more input files, separated by spaces.

Optional arguments:
- -src_srs: Spatial Reference System of the LAS files. Must be a valid PROJ4 string
  or EPSG code accepted by GeoPandas. **Defaults to WGS84 UTM Zone 18N**
- -dst_srs: Spatial Reference System of the output VECTOR files.
  Must be a valid PROJ4 string or EPSG code accepted by GeoPandas.
  **Defaults to WGS84 (EPSG 4326)**
- --extension (-e): Extension of vector file for saving. **Defaults to `gpkg`**
- --bounds (-b): Minimum/maximum height in output vector files. All points outside
  these bounds will be discarded. Defaults to [0, 25].
- --number_of_cores (-nc): Number of CPU cores to use. Defaults to all the available
  cores on the machine.

This script is designed for Command Line use. Example usage:

python3 las_to_vector.py -i /Users/arredond/Desktop/haiti/data/lidar/FID_0_PointHeight.LAS
python3 las_to_vector.py -i ../data/lidar/FID_0_PointHeight.LAS -e shp
python3 las_to_vector.py -i ../data/lidar/*.LAS

"""

import argparse
import os

from multiprocessing import Pool
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd

from laspy.file import File
from shapely.geometry import Point

# Conversion function
def las_to_vector(path, src_srs, dst_srs, extension, bounds):
    print(f'Working on {path}')
    
    # Get driver if possible. If not, use GPKG as default
    possible_drivers = {
        'shp': 'ESRI Shapefile',
        'gpkg': 'GPKG',
        'json': 'GeoJSON',
        'geojson': 'GeoJSON'
    }
    ext = extension if extension in possible_drivers.keys() else 'gpkg'
    driver = possible_drivers[extension]

    # Read Lidar
    f = File(path, mode='r')
    
    # Get data into Dataframe
    df = pd.DataFrame.from_records({
        'x': f.x,
        'y': f.y,
        'z': f.z,
        'new_class': f.Classification,
        'raw_class': f.raw_classification,
        'intensity': f.Intensity
    })
    # Filter by bounds
    min_bound, max_bound = bounds
    df = df.loc[(df['z'] >= min_bound) & (df['z'] <= max_bound), :]
    
    # Transform to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    gdf.crs = src_srs
    
    # Re-project to destination SRS (by default, EPSG:4326)
    gdf = gdf.to_crs(epsg=dst_srs)
    
    # Create output folder if necessary and save
    save_folder = os.path.join(os.path.dirname(path), 'las_to_vector')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(
        save_folder,
        os.path.basename(path).replace('.LAS', f'.{ext}')
    )
    gdf.to_file(save_path, driver=driver)

# Parse input arguments
parser = argparse.ArgumentParser(description='LAS to Vector converter. Defaults to saving in GPKG format')
parser.add_argument('-i', '--input_files', nargs='+', required=True,
                    help='List of paths to files with scores, separated by spaces')
parser.add_argument('-src_srs', type=str, required=False, \
                    default='+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs',
                    help='Proj. 4 or valid EPSG code to use as **input** Spatial Reference System')
parser.add_argument('-dst_srs', type=str, required=False, default=4326,
                    help='Proj. 4 or valid EPSG code to use as **output** Spatial Reference System')
parser.add_argument('-e', '--extension', type=str, choices=['shp', 'gpkg', 'json', 'geojson'],
                    required=False, default='gpkg', help='Vector format to save output file(s) in')
parser.add_argument('-b', '--bounds', type=int, nargs=2, required=False, default=[0,25],
                    help='Minimum/maximum values of z coordinate')
parser.add_argument('-nc', '--number_of_cores', type=int, required=False,
                    help='Minimum/maximum values of z coordinate')

args = parser.parse_args()

# Run
if __name__ == '__main__':
    # Prepare 
    nc = args.number_of_cores if args.number_of_cores else (os.cpu_count() - 1)
    process_image = partial(
        las_to_vector,
        src_srs=args.src_srs,
        dst_srs=args.dst_srs,
        extension=args.extension,
        bounds=args.bounds)
    
    with Pool(nc) as pool:
        pool.map(process_image, args.input_files)
