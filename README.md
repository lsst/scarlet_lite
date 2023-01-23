# Scarlet Lite

This package is a re-imagining of the original [scarlet](https://github.com/lsst/scarlet) deblending package specificaly designed for single instrument surveys where all of the observed images have been reprojected onto the same WCS, with the same pixel grid.
Scarlet Lite uses the same algorithm as the original scarlet, as described in [Melchior et al. (2018)](https://doi.org/10.1016/j.ascom.2018.07.001), with a more streamlined architecture that is faster and uses less memory due to the assumptions regarding the observed images.

This can be installed from the command line using

```
pip install .
```

and soon will also support installation with scons in the [LSST science pipelines](https://pipelines.lsst.io/). A full set of documentation is also in the works, so see this page soon for updates.