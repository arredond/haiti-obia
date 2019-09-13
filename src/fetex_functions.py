import os

from math import pi

import numpy as np
import pandas as pd

from scipy import stats
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import perimeter

def mode(x):
    mode, count = stats.mode(x)
    return mode[0]

def get_features_cols(nbands):

    # Get feature names
    cols = ['id'] # Initializing with ID

    # Shape features -- per segment
    shape_features = ['area', 'perimeter', 'compacity_index']
    for i in shape_features:
        cols.append(i)

    # Radiometric features
    stat = ['min', 'max', 'mean', 'median', 'var', 'kurtosis', 'mode']
    for i in stat:
        for band in range(nbands):
            cols.append('{0}_{1}'.format(i, band+1))

    # Texture features
    texture_features = ['contrast', 'dissimilarity', 'homogeneity', 'ASM']
    for feat in texture_features:
        for band in range(nbands):
            cols.append('{0}_{1}'.format(feat, band+1))
    
    return cols

def extract_feature_df(img, segments, cols, pixel_size=1, debug=True):
    """
    Extract features from image. Image and segments must both be NumPy arrays.
    """
    
    # Initialize feature dict
    feature_dict = {}
    for col in cols:
        feature_dict[col] = []

    unique_segments = np.unique(segments)
    total_segments = len(unique_segments)
    print('{} segments to process'.format(total_segments))
    for seg_id in unique_segments:
        if (debug==True) and (seg_id % 100 == 0):
            print('Segments processed: {0}, {1} % '.format(seg_id, round(100*seg_id/total_segments)))

        # Filter segments on ID and get bounding box
        idx = (segments == seg_id).reshape((segments.shape[0], segments.shape[1]))
        filt = np.where(idx)
        bbox = [min(filt[0]), max(filt[0]), min(filt[1]), max(filt[1])] # min_x, max_x, min_y, max_y

        # Generate empty mask, fill with segment and crop to bbox
        # These are used to calculate shape features
        seg = np.zeros(img.shape[0:2], dtype=bool)
        seg[idx] = 1
        seg = seg[bbox[0]:bbox[1], bbox[2]:bbox[3]]

        # Extract image and crop to bbox
        # This is used to calculate radiometric and texture features
        img_seg = np.zeros(img.shape, dtype=img.dtype)
        img_seg[idx] = img[idx]
        img_seg = img_seg[bbox[0]:bbox[1], bbox[2]:bbox[3]]

        # Check segment is not only no data for all bands
        drop_segment = False
        if len(seg[seg != 0]) == 0:
            drop_segment = True

        for band in range(img_seg.shape[2]):
            if len(img_seg[:, :, band][seg != 0]) == 0:
                drop_segment = True
        
        if drop_segment:
            continue

        # Adding segment_id as separate field
        feature_dict['id'].append(seg_id)   

        # Shape features (per segment)
        feature_dict['area'].append(len(seg[seg == 1]) * pixel_size ** 2)
        feature_dict['perimeter'].append(perimeter(seg) * pixel_size)
        feature_dict['compacity_index'].append(
            4*pi*feature_dict['area'][-1]/(feature_dict['perimeter'][-1]**2)
            # feature_dict['perimeter'][-1]/(np.sqrt(feature_dict['area'][-1])) OLD
        )

        # Radiometric and texture features (per band)
        for band in range(img_seg.shape[2]):
            img_band = img_seg[:, :, band]
            band_flat = img_band[seg != 0] # Keep only segment data, not all bbox data

            # Radiometric
            feature_dict[f'max_{band+1}'].append(band_flat.max())
            feature_dict[f'min_{band+1}'].append(band_flat.min())
            feature_dict[f'mean_{band+1}'].append(band_flat.mean())
            feature_dict[f'median_{band+1}'].append(np.median(band_flat))
            feature_dict[f'var_{band+1}'].append(band_flat.var())
            feature_dict[f'kurtosis_{band+1}'].append(stats.kurtosis(band_flat))
            feature_dict[f'mode_{band+1}'].append(mode(band_flat))
            # RGB differences
            

            # Texture
            n_bins = 16
            bins = np.linspace(img_band.min(), img_band.max(), num=n_bins)
            binned_band = np.digitize(img_band, bins)
            glcm = greycomatrix(binned_band, [1], [0, 90, 180, 270], levels=n_bins+1)
            glcm = glcm[1:, 1:, :, :]
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'ASM']:
                feature_dict['{0}_{1}'.format(prop, band+1)].append(np.mean(greycoprops(glcm, prop)))
                
    return pd.DataFrame.from_dict(feature_dict).reset_index()
