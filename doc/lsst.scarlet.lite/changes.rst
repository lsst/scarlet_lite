.. _lsst.scarlet.lite-changes:

=================
v30.0.0 Changes
=================

Improved Slicing
----------------

In v29.0 slicing was only supported for the ``Image`` class. In ``v30.0`` slicing has been extended to ``Blend``, ``Source``, ``Component``, and ``Observation`` classes. This allows uers to use a subset of bands or change the band order of each of these classes. ``Observation`` can be sliced along the spatial dimension as well.

New Image methods
-----------------
- ``Image.trimmed`` method added to remove data below a threshold from an Image.
- ``Image.at`` method added to extract a single pixel from an Image.

Serialization Improvments
-------------------------
Serialization received a major update in ``v30.0`` to improve performance and usability. The major upgrade is a ``Migration Registry`` that registers all scarlet serializable classes and allows users to register their own custom classes along with function to migrate between different versions of the class.
This allows scarlet to automatically handle versioning of serialized objects and migrate them to the latest version when deserializing.

As part of this update the classes used for serialization were expanded to included base classes such as ``BlendBaseData``, ``SourceBaseData``, and ``ComponentBaseData`` to make it easier for users to extend serialization to their own custom classes.

Copying and Deep Copying
------------------------
To support the serialization improvements, ``__copy__`` and ``__deepcopy__`` methods were added to nearly all scarlet lite classes. This allows users to create copies of scarlet objects using the standard library ``copy`` and ``deepcopy`` modules.

Initialization Changes
----------------------
Initialization has a few changes to make it both more customizable and more standardized
- The ineffective ``FactorizedWaveletInitialization`` was deprecated and the ``FactorizedChirInitialization`` was changed to ``FactorizedInitialization`` as testing has shown that it is a superior algorithm.
- A large number of new parameters were added to ``FactorizedInitialization`` to make it easier for users to initialize sources using more tunable constraints and detection images that may differ from the observation image.

Other Changes
-------------
- The detection algorithms have been updated and while not ready for production users should see improved performance.
- A more customizable ``conserve_flux`` function was added to ``scarlet``
- Positions are rounded instead of truncated during initialization to remove a bias from inaccurate sources positions.
