#!/usr/bin/env Rscript

library(raster)

# This script simply takes two or more raster files,
# aligns them and writes out a new TIFF file.
#
# The last argument will be the file to save to.
# All others are the rasters to align and stack, 
# where the first one is the reference to align them to.
args = commandArgs(trailingOnly = TRUE)

n = length(args)
save_name = args[n]
ref = stack(args[1])
rest = lapply(args[2:(n-1)], stack)

for (r in rest) {
  ref = addLayer(ref, resample(r, ref, method = 'bilinear'))
}

writeRaster(ref, filename = args[n], format = 'GTiff')
