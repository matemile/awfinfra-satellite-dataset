#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modis_ist_preprocess.py
========================

Pre-process MODIS Ice Surface Temperature (IST) swath products (MOD29/MYD29,
collection 6.1) together with their companion geolocation files (MOD03/MYD03),
split the combined IST SDS into a physical-temperature field and a
classification field, resample both onto the CARRA2 polar-stereographic
grid with pyresample, merge Terra + Aqua observations that fall within a
+/- 30 minute window of a requested hour, and write the result as a single
CF-compliant NetCDF4 file.

--------------------------------------------------------------------------
Ice_Surface_Temperature encoding (MOD29/MYD29, collection 6.1)
--------------------------------------------------------------------------
The SDS "Ice_Surface_Temperature" is stored as uint16 counts with
scale_factor = 0.01, add_offset = 0.0, _FillValue = 65535. After scaling,
the value (call it `v`, in "Kelvin-like" units) is interpreted as:

    v == 0.0    -> missing data
    v == 1.0    -> no decision
    v == 11.0   -> night
    v == 25.0   -> land
    v == 37.0   -> inland water
    v == 39.0   -> open ocean
    v == 50.0   -> cloud
    v in [210, 313]  -> physical IST value in Kelvin (243-273 K expected range)
    v == 655.35 (raw 65535) -> fill value

(Source: MODIS/VIIRS Snow and Ice Product User Guide, Table 4/5, NASA GSFC
 -- https://modis-snow-ice.gsfc.nasa.gov/uploads/siug_c5.pdf)

This script maps that key onto two independent NetCDF variables:

  ist_phys  (float32, K)   - the physical temperature where v is in the
                             valid physical range, NaN everywhere else.
  ist_class (int16, flag)  - 0=missing/fill(*) 1=no_decision 2=night
                             3=land 4=inland_water 5=open_ocean 6=cloud
                             7=fill 255->not used, -1=unknown/outside-swath

  (*) DESIGN DECISION: code 0 is reused both for MODIS's own "missing data"
      raw value AND as the sentinel written wherever ist_phys already holds
      a real physical temperature (no classification needed there). This is
      intentional and consistent with ist_class's _FillValue = 0s: pixels
      that don't need a classification code (because they carry a real
      temperature, or because the input truly had no data) collapse onto
      the same "empty" flag. Non-zero codes 1-7 always mean "no physical
      temperature is available here, and this is why". -1 is reserved for
      pixels with invalid/missing geolocation or that fall outside every
      swath's coverage during resampling. If you need to tell "real missing
      data" and "physical value present" apart, keep ist_phys as the source
      of truth (NaN vs. not-NaN) -- ist_class is a *reason* code, not an
      exhaustive partition.

--------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------
    pip install numpy netCDF4 pyresample pyproj pyhdf
(pyhdf needs the HDF4 C library; on HPC modules/conda-forge are usually the
 easiest route: `conda install -c conda-forge pyhdf`)

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    python modis_ist_preprocess.py \
        --year 2025 --doy 121 --hour 2 \
        --mod29-dir /data/modis/MOD29 \
        --myd29-dir /data/modis/MYD29 \
        --mod03-dir /data/modis/MOD03 \
        --myd03-dir /data/modis/MYD03 \
        --out-dir /data/modis/out \
        --carra2-sample /data/grids/carra2_sea.d.nc

This produces one file per run, e.g.:
    /data/modis/out/MODIS_IST_CARRA2_20250501_0200.nc
(time dimension length 1; run the script once per hour and concatenate with
 `xarray.concat`/`ncrcat` afterwards if you want a multi-hour file such as
 the time=4 example in the spec.)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pyhdf.SD import SD, SDC
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyhdf is required to read HDF4 MODIS files. "
        "Install with: conda install -c conda-forge pyhdf"
    ) from exc

try:
    from pyresample import geometry, kd_tree
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyresample is required. Install with: pip install pyresample"
    ) from exc

from netCDF4 import Dataset

LOG = logging.getLogger("modis_ist_preprocess")


# =========================================================================
# Configuration / constants
# =========================================================================

# MODIS Ice_Surface_Temperature "key" values (already scale-factor applied,
# i.e. scaled = raw_uint16 * 0.01), see module docstring / Table 4-5 of the
# MODIS snow/ice user guide.
IST_KEY_TO_CLASS = {
    0.0: 0,   # missing data
    1.0: 1,   # no decision
    11.0: 2,  # night
    25.0: 3,  # land
    37.0: 4,  # inland water
    39.0: 5,  # open ocean
    50.0: 6,  # cloud
}
IST_FILL_RAW = 65535           # raw uint16 fill value -> scaled 655.35
IST_PHYS_VALID_MIN = 210.0     # valid_range (scaled) lower bound
IST_PHYS_VALID_MAX = 273.0     # expected_range upper bound, more meaningful than valid max
IST_KEY_TOL = 1.0e-3           # float tolerance when matching key codes

CLASS_FLAG_VALUES = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int16)
CLASS_FLAG_MEANINGS = (
    "missing no_decision night land inland_water open_ocean cloud fill unknown"
)
CLASS_FILL_VALUE = np.int16(0)
CLASS_UNKNOWN = np.int16(-1)

# Default pyresample parameters (metres, matches a 1 km source -> 2.5 km
# target regridding). Tune with the CLI flags if results look too smoothed
# or too patchy.
DEFAULT_RADIUS_IST = 5000.0        # radius_of_influence for Gaussian resample
DEFAULT_SIGMA = 2000.0             # Gaussian sigma
DEFAULT_NEIGHBOURS = 8
DEFAULT_RADIUS_CLASS = 3500.0      # radius_of_influence for nearest-neighbour

# MODIS filenames, e.g. MOD29.A2025121.0225.061.2025121141036.hdf
FNAME_RE = re.compile(
    r"^(?P<sat>MOD29|MYD29|MOD03|MYD03)\.A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<hhmm>\d{4})\.\d{3}\.\d{13,}\.hdf$"
)
IST_TO_GEO_SAT = {"MOD29": "MOD03", "MYD29": "MYD03"}
SAT_LABEL = {"MOD29": "Terra", "MYD29": "Aqua"}


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class FileEntry:
    path: str
    sat: str
    year: int
    doy: int
    hhmm: str
    obs_dt: datetime


@dataclass
class MatchedGranule:
    satellite: str          # "Terra" / "Aqua"
    ist_path: str
    geo_path: str
    obs_dt: datetime


@dataclass
class ResampledGranule:
    satellite: str
    obs_dt: datetime
    ist_phys: np.ndarray    # (ny, nx) float32, NaN where invalid
    ist_class: np.ndarray   # (ny, nx) int16, -1 where no coverage


# =========================================================================
# Step 1: locate + pair input files
# =========================================================================

def _index_hdf_dir(directory: str, wanted_prefixes: set) -> Dict[str, List[FileEntry]]:
    """Index all MODIS HDF files in a directory by satellite/product prefix."""
    index: Dict[str, List[FileEntry]] = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    for fname in sorted(os.listdir(directory)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        sat = m.group("sat")
        if sat not in wanted_prefixes:
            continue
        year = int(m.group("year"))
        doy = int(m.group("doy"))
        hhmm = m.group("hhmm")
        obs_dt = (
            datetime(year, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=doy - 1, hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
        )
        index.setdefault(sat, []).append(
            FileEntry(os.path.join(directory, fname), sat, year, doy, hhmm, obs_dt)
        )
    return index


def find_and_match_files(
    year: int,
    doy: int,
    hour: int,
    mod29_dir: str,
    myd29_dir: str,
    mod03_dir: str,
    myd03_dir: str,
    window_minutes: int = 30,
) -> Tuple[List[MatchedGranule], datetime]:
    """
    Find all MOD29/MYD29 granules for the given year/day-of-year whose
    observation time falls within +/- window_minutes of the requested hour,
    and pair each with its MOD03/MYD03 geolocation file (matched by
    satellite + year + doy + hhmm; the trailing production-time stamp in the
    filenames is intentionally ignored). MOD29, MYD29, MOD03 and MYD03 are
    each read from their own directory, since Terra/Aqua IST and
    geolocation files are commonly stored separately.
    """
    target_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour)
    window_start = target_dt - timedelta(minutes=window_minutes)
    window_end = target_dt + timedelta(minutes=window_minutes)

    ist_index: Dict[str, List[FileEntry]] = {}
    ist_index.update(_index_hdf_dir(mod29_dir, {"MOD29"}))
    ist_index.update(_index_hdf_dir(myd29_dir, {"MYD29"}))

    geo_index: Dict[str, List[FileEntry]] = {}
    geo_index.update(_index_hdf_dir(mod03_dir, {"MOD03"}))
    geo_index.update(_index_hdf_dir(myd03_dir, {"MYD03"}))

    matches: List[MatchedGranule] = []
    for ist_sat, entries in ist_index.items():
        geo_sat = IST_TO_GEO_SAT[ist_sat]
        geo_entries = geo_index.get(geo_sat, [])
        for e in entries:
            if not (window_start <= e.obs_dt < window_end):
                continue
            geo_match = next(
                (g for g in geo_entries if g.year == e.year and g.doy == e.doy and g.hhmm == e.hhmm),
                None,
            )
            if geo_match is None:
                LOG.warning(
                    "No matching %s geolocation file found for %s (year=%d doy=%03d hhmm=%s) - skipping.",
                    geo_sat, os.path.basename(e.path), e.year, e.doy, e.hhmm,
                )
                continue
            matches.append(
                MatchedGranule(
                    satellite=SAT_LABEL[ist_sat],
                    ist_path=e.path,
                    geo_path=geo_match.path,
                    obs_dt=e.obs_dt,
                )
            )

    matches.sort(key=lambda m: m.obs_dt)
    LOG.info(
        "Target hour %s UTC: found %d matched granule(s) within +/-%d min.",
        target_dt.isoformat(), len(matches), window_minutes,
    )
    for m in matches:
        LOG.info("  %-5s %s  (dt=%+.1f min)", m.satellite, m.obs_dt.isoformat(),
                  (m.obs_dt - target_dt).total_seconds() / 60.0)
    return matches, target_dt


# =========================================================================
# Step 2: read HDF4 inputs + split IST into physical / classification
# =========================================================================

def read_modis_ist_hdf(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the "Ice_Surface_Temperature" SDS from a MOD29/MYD29 HDF4 file.
    Returns (scaled_values, raw_counts) as float64 / uint16 arrays with the
    same shape, where scaled_values = raw_counts * scale_factor + add_offset.
    """
    hdf = SD(path, SDC.READ)
    try:
        sds = hdf.select("Ice_Surface_Temperature")
        raw = sds.get()
        attrs = sds.attributes()
        scale_factor = float(attrs.get("scale_factor", 0.01))
        add_offset = float(attrs.get("add_offset", 0.0))
        sds.endaccess()
    finally:
        hdf.end()

    raw = raw.astype(np.uint16)
    scaled = raw.astype(np.float64) * scale_factor + add_offset
    return scaled, raw


def read_modis_geo_1km_hdf(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read 1 km Latitude/Longitude arrays from a MOD03/MYD03 HDF4 file."""
    hdf = SD(path, SDC.READ)
    try:
        lat = hdf.select("Latitude").get().astype(np.float64)
        lon = hdf.select("Longitude").get().astype(np.float64)
    finally:
        hdf.end()

    lat = np.where((lat < -90.0) | (lat > 90.0), np.nan, lat)
    lon = np.where((lon < -180.0) | (lon > 180.0), np.nan, lon)
    return lat, lon


def derive_ist_and_classes(
    ist_scaled: np.ndarray, raw_counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the combined MOD29/MYD29 IST SDS into:
      ist_phys  : float32 physical temperature (K), NaN where not valid
      ist_class : int16 classification code (see IST_KEY_TO_CLASS / module
                  docstring for the exact mapping)
    """
    ist_phys = np.full(ist_scaled.shape, np.nan, dtype=np.float32)
    ist_class = np.full(ist_scaled.shape, CLASS_UNKNOWN, dtype=np.int16)

    # 1) Non-physical key codes (missing, no_decision, night, land, ...)
    for key_val, code in IST_KEY_TO_CLASS.items():
        mask = np.isclose(ist_scaled, key_val, atol=IST_KEY_TOL)
        ist_class[mask] = code

    # 2) Fill value (raw counts == 65535, scaled == 655.35)
    fill_mask = raw_counts == IST_FILL_RAW
    ist_class[fill_mask] = 7

    # 3) Physical IST values -> real temperature, class collapses to the
    #    fill/no-classification-needed sentinel (0).
    phys_mask = (
        (ist_scaled >= IST_PHYS_VALID_MIN)
        & (ist_scaled <= IST_PHYS_VALID_MAX)
        & ~fill_mask
    )
    ist_phys[phys_mask] = ist_scaled[phys_mask].astype(np.float32)
    ist_class[phys_mask] = 0

    return ist_phys, ist_class


# =========================================================================
# Step 3: build the CARRA2 target area + resample one granule
# =========================================================================

@dataclass
class Carra2Grid:
    area_def: "geometry.AreaDefinition"
    x: np.ndarray
    y: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    proj4: str
    proj_attrs: Dict[str, object]
    flip_y: bool
    flip_x: bool
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    domain_margin_deg: float = 2.0


def build_carra2_grid(sample_nc_path: str, domain_margin_deg: float = 2.0) -> Carra2Grid:
    """
    Build a pyresample AreaDefinition from the CARRA2 sample NetCDF file
    (its regular projected x/y coordinates + polar_stereographic mapping).
    Also returns flip flags so that resampled arrays end up in the same
    row/column order as the x/y/longitude/latitude arrays copied straight
    from the sample file into the output.
    """
    with Dataset(sample_nc_path, "r") as ds:
        x = ds.variables["x"][:].astype(np.float64)
        y = ds.variables["y"][:].astype(np.float64)
        longitude = ds.variables["longitude"][:, :].astype(np.float64)
        latitude = ds.variables["latitude"][:, :].astype(np.float64)
        proj_var = ds.variables["projection_polar_stereographic"]
        proj4 = proj_var.getncattr("proj4")
        proj_attrs = {k: proj_var.getncattr(k) for k in proj_var.ncattrs()}

    nx = x.size
    ny = y.size
    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))

    x_min, x_max = float(x.min()) - dx / 2.0, float(x.max()) + dx / 2.0
    y_min, y_max = float(y.min()) - dy / 2.0, float(y.max()) + dy / 2.0
    area_extent = (x_min, y_min, x_max, y_max)

    area_def = geometry.AreaDefinition(
        area_id="carra2_2p5km",
        description="CARRA2 domain, polar stereographic, 2.5 km",
        proj_id="carra2_2p5km",
        projection=proj4,
        width=nx,
        height=ny,
        area_extent=area_extent,
    )

    # pyresample convention: row 0 = max-y (north-up), col 0 = min-x.
    # Determine whether the sample file's own x/y ordering matches that,
    # so the resampled fields line up with the x/y/lon/lat we copy verbatim
    # into the output file.
    y_ascending = y[0] < y[-1]
    x_descending = x[0] > x[-1]
    flip_y = bool(y_ascending)     # need to flip rows if source y goes min->max
    flip_x = bool(x_descending)    # need to flip cols if source x goes max->min

    # Bounding box of the target domain (from the actual lon/lat arrays we
    # already loaded) - used to cheaply pre-crop each MODIS swath before
    # resampling (see crop_swath_to_domain). This is the single biggest
    # runtime win: a full 1 km swath is ~2030x1354 (~2.7M) points, but for a
    # regional domain like the Nordic Seas only a small fraction of that
    # typically overlaps, so building the KD-tree on the cropped swath
    # instead of the full swath is dramatically faster.
    lat_min = float(np.nanmin(latitude))
    lat_max = float(np.nanmax(latitude))
    lon_min = float(np.nanmin(longitude))
    lon_max = float(np.nanmax(longitude))

    return Carra2Grid(
        area_def=area_def, x=x, y=y, longitude=longitude, latitude=latitude,
        proj4=proj4, proj_attrs=proj_attrs, flip_y=flip_y, flip_x=flip_x,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        domain_margin_deg=domain_margin_deg,
    )


def crop_swath_to_domain(
    lat: np.ndarray,
    lon: np.ndarray,
    extra_arrays: Tuple[np.ndarray, ...],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    margin_deg: float,
) -> Optional[Tuple[np.ndarray, ...]]:
    """
    Cheap spatial pre-filter: find the minimal bounding rectangle (in the
    swath's own row/column indices) that contains every 1 km pixel whose
    lat/lon falls inside the target domain bbox (+ margin), and crop
    lat/lon plus any extra co-located arrays to it.

    Returns None if the granule does not overlap the domain at all (caller
    should skip resampling entirely in that case). NOTE: this assumes the
    target domain does not straddle the +/-180 degree dateline, which holds
    for the CARRA2 Nordic Seas domain (lon ~ 5.5-25.7 E).
    """
    mask = (
        (lat >= lat_min - margin_deg) & (lat <= lat_max + margin_deg)
        & (lon >= lon_min - margin_deg) & (lon <= lon_max + margin_deg)
    )
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    cropped = tuple(a[r0:r1, c0:c1] for a in (lat, lon) + tuple(extra_arrays))
    return cropped


def resample_one(
    ist_hdf: str,
    geo_hdf: str,
    grid: Carra2Grid,
    radius_of_influence_ist: float,
    sigma: float,
    neighbours: int,
    radius_of_influence_class: float,
    nprocs: int = 4,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Read + split + resample one MOD29/MOD03 (or MYD29/MYD03) pair onto the
    CARRA2 grid. Returns None if the granule has no spatial overlap with the
    target domain at all (nothing to resample).
    """
    t0 = time.perf_counter()
    ist_scaled, raw_counts = read_modis_ist_hdf(ist_hdf)
    lat_1km, lon_1km = read_modis_geo_1km_hdf(geo_hdf)

    ny = min(ist_scaled.shape[0], lat_1km.shape[0], lon_1km.shape[0])
    nx = min(ist_scaled.shape[1], lat_1km.shape[1], lon_1km.shape[1])
    ist_scaled = ist_scaled[:ny, :nx]
    raw_counts = raw_counts[:ny, :nx]
    lat_1km = lat_1km[:ny, :nx]
    lon_1km = lon_1km[:ny, :nx]

    ist_phys_1km, class_codes_1km = derive_ist_and_classes(ist_scaled, raw_counts)

    geo_mask = np.isnan(lat_1km) | np.isnan(lon_1km)
    ist_phys_1km = np.where(geo_mask, np.nan, ist_phys_1km)
    class_codes_1km = np.where(geo_mask, CLASS_UNKNOWN, class_codes_1km).astype(np.int16)
    t_read = time.perf_counter() - t0

    # --- Spatial pre-filter (biggest speedup): crop the swath down to the
    # bounding box of the CARRA2 domain before ever building a KD-tree.
    # A full 1 km granule is ~2.7M points; a regional domain like the
    # Nordic Seas often only overlaps a small fraction of that, so this can
    # cut resampling cost by an order of magnitude. Granules that don't
    # overlap the domain at all (common when the time window catches a
    # pass over a completely different part of the globe) are skipped
    # entirely instead of being resampled uselessly.
    cropped = crop_swath_to_domain(
        lat_1km, lon_1km, (ist_phys_1km, class_codes_1km),
        grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max, grid.domain_margin_deg,
    )
    if cropped is None:
        LOG.info("  no spatial overlap with the CARRA2 domain - skipping resampling for this granule.")
        return None
    lat_1km, lon_1km, ist_phys_1km, class_codes_1km = cropped
    LOG.debug(
        "  cropped swath from full granule to %dx%d pixels overlapping the domain (+%.1f deg margin).",
        lat_1km.shape[0], lat_1km.shape[1], grid.domain_margin_deg,
    )

    swath_def = geometry.SwathDefinition(lons=lon_1km, lats=lat_1km)

    t1 = time.perf_counter()
    ist_gauss = kd_tree.resample_gauss(
        swath_def,
        ist_phys_1km,
        grid.area_def,
        radius_of_influence=radius_of_influence_ist,
        sigmas=sigma,
        neighbours=neighbours,
        fill_value=np.nan,
        nprocs=nprocs,
    )
    t_gauss = time.perf_counter() - t1

    t2 = time.perf_counter()
    class_nn = kd_tree.resample_nearest(
        swath_def,
        class_codes_1km.astype(np.float64),  # resample_nearest wants float for fill_value=-1 semantics
        grid.area_def,
        radius_of_influence=radius_of_influence_class,
        fill_value=-1,
        nprocs=nprocs,
    )
    t_nn = time.perf_counter() - t2
    class_nn = np.round(class_nn).astype(np.int16)

    if grid.flip_y:
        ist_gauss = ist_gauss[::-1, :]
        class_nn = class_nn[::-1, :]
    if grid.flip_x:
        ist_gauss = ist_gauss[:, ::-1]
        class_nn = class_nn[:, ::-1]

    LOG.info(
        "  timing: read+split=%.2fs gauss_resample=%.2fs nn_resample=%.2fs",
        t_read, t_gauss, t_nn,
    )
    return ist_gauss.astype(np.float32), class_nn


# =========================================================================
# Step 4: merge Terra + Aqua observations into one hourly slot
# =========================================================================

def composite_hourly(
    resampled: List[ResampledGranule], target_dt: datetime, grid_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge all resampled granules (possibly several 5-minute Terra AND Aqua
    granules) that fall in the +/- 30 minute window into a single hourly
    ist_phys / ist_class field.

    Merge strategy - "nearest observation time wins", evaluated per pixel:
      For every grid cell we keep the value coming from whichever granule's
      acquisition time is closest to the requested hour, among the granules
      that actually have data at that cell (ist_class != -1, i.e. within
      resampling radius of a real swath pixel). This:
        - naturally resolves Terra/Aqua overlap without blending two
          different-time, different-viewing-geometry measurements together
          (important for a temperature retrieval - averaging swaths taken
          10-20 minutes apart over drifting/melting ice can create
          non-physical intermediate values),
        - fills gaps in one satellite's swath with the other satellite's
          data whenever available,
        - keeps ist_phys and ist_class consistent (they always come from
          the same source granule at a given pixel).

    Alternative strategies you may prefer depending on your application:
      - "mean" of all valid ist_phys per pixel (denoise, but blurs fronts
        and mixes different overpass times - not applied to ist_class,
        which has no meaningful average),
      - satellite priority (e.g. always prefer Terra over Aqua, or vice
        versa) instead of / in addition to nearest-time.
    To switch to a simple mean, replace the per-pixel argmin logic below
    with a masked nanmean over the stacked ist_phys arrays.
    """
    ny, nx = grid_shape
    combined_phys = np.full((ny, nx), np.nan, dtype=np.float32)
    combined_class = np.zeros((ny, nx), dtype=np.int16)  # default: 0 = missing (no coverage at all)
    best_dt_diff = np.full((ny, nx), np.inf, dtype=np.float64)

    if not resampled:
        LOG.warning("No granules to composite - output will be all-missing.")
        return combined_phys, combined_class

    for gran in resampled:
        dt_diff = abs((gran.obs_dt - target_dt).total_seconds())
        has_data = gran.ist_class != CLASS_UNKNOWN
        take = has_data & (dt_diff < best_dt_diff)
        if not np.any(take):
            continue
        combined_phys[take] = gran.ist_phys[take]
        combined_class[take] = gran.ist_class[take]
        best_dt_diff[take] = dt_diff

    n_covered = int(np.sum(np.isfinite(best_dt_diff)))
    n_total = ny * nx
    LOG.info(
        "Composited %d granule(s): %d/%d grid cells (%.1f%%) received swath coverage.",
        len(resampled), n_covered, n_total, 100.0 * n_covered / n_total,
    )
    return combined_phys, combined_class


# =========================================================================
# Step 5: write output NetCDF
# =========================================================================

def write_output_netcdf(
    out_path: str,
    target_dt: datetime,
    grid: Carra2Grid,
    ist_phys: np.ndarray,
    ist_class: np.ndarray,
    matches: List[MatchedGranule],
) -> None:
    ny, nx = ist_phys.shape

    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)  # unlimited, so hourly files can later be concatenated
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)

        v_time = ds.createVariable("time", "f8", ("time",))
        v_time.standard_name = "time"
        v_time.long_name = "time"
        v_time.units = "seconds since 1970-01-01 00:00:00 +00:00"
        v_time.calendar = "standard"
        v_time.axis = "T"
        v_time[:] = [target_dt.replace(tzinfo=timezone.utc).timestamp()]

        v_x = ds.createVariable("x", "f8", ("x",))
        v_x.standard_name = "projection_x_coordinate"
        v_x.long_name = "x-coordinate in Cartesian system"
        v_x.units = "m"
        v_x.axis = "X"
        v_x[:] = grid.x

        v_y = ds.createVariable("y", "f8", ("y",))
        v_y.standard_name = "projection_y_coordinate"
        v_y.long_name = "y-coordinate in Cartesian system"
        v_y.units = "m"
        v_y.axis = "Y"
        v_y[:] = grid.y

        v_lon = ds.createVariable("longitude", "f8", ("y", "x"))
        v_lon.standard_name = "longitude"
        v_lon.long_name = "longitude"
        v_lon.units = "degree_east"
        v_lon[:, :] = grid.longitude

        v_lat = ds.createVariable("latitude", "f8", ("y", "x"))
        v_lat.standard_name = "latitude"
        v_lat.long_name = "latitude"
        v_lat.units = "degree_north"
        v_lat[:, :] = grid.latitude

        v_proj = ds.createVariable("projection_polar_stereographic", "i4")
        for k, val in grid.proj_attrs.items():
            v_proj.setncattr(k, val)

        v_ist = ds.createVariable(
            "ist_phys", "f4", ("time", "y", "x"),
            fill_value=np.float32(np.nan), zlib=True, complevel=4,
        )
        v_ist.standard_name = "surface_temperature"
        v_ist.long_name = "Ice Surface Temperature"
        v_ist.units = "K"
        v_ist.coordinates = "latitude longitude"
        v_ist.grid_mapping = "projection_polar_stereographic"
        v_ist[0, :, :] = ist_phys

        v_cls = ds.createVariable(
            "ist_class", "i2", ("time", "y", "x"),
            fill_value=CLASS_FILL_VALUE, zlib=True, complevel=4,
        )
        v_cls.standard_name = "class_code"
        v_cls.long_name = "IST class code derived from Ice_Surface_Temperature"
        v_cls.flag_values = CLASS_FLAG_VALUES
        v_cls.flag_meanings = CLASS_FLAG_MEANINGS
        v_cls.coordinates = "latitude longitude"
        v_cls.grid_mapping = "projection_polar_stereographic"
        v_cls[0, :, :] = ist_class

        ds.Conventions = "CF-1.8"
        ds.title = "MODIS MOD29/MYD29 IST on CARRA2 polar stereographic grid"
        ds.source = "Resampled from MODIS HDF4 (MOD29/MYD29 + MOD03/MYD03) with pyresample"
        ds.history = (
            f"Created {datetime.now(timezone.utc).isoformat()} by modis_ist_preprocess.py"
        )
        ds.references = "https://modis-snow-ice.gsfc.nasa.gov/uploads/siug_c5.pdf"
        input_list = "; ".join(
            f"{m.satellite}:{os.path.basename(m.ist_path)}+{os.path.basename(m.geo_path)}"
            for m in matches
        )
        ds.input_granules = input_list if input_list else "none (no coverage in window)"


# =========================================================================
# Main driver
# =========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-process MODIS MOD29/MYD29 IST onto the CARRA2 grid for one target hour."
    )
    p.add_argument("--year", type=int, required=True, help="Year, e.g. 2025")
    p.add_argument("--doy", type=int, required=True, help="Day of year (1-366)")
    hour_group = p.add_mutually_exclusive_group(required=True)
    hour_group.add_argument("--hour", type=int, help="Single target UTC hour (0-23)")
    hour_group.add_argument(
        "--hours", type=str,
        help=(
            "Comma-separated list of target UTC hours to process in one run, "
            "e.g. '0,1,2,3,...,23'. Processing multiple hours per invocation "
            "reuses the CARRA2 grid and avoids re-paying Python/library "
            "startup cost for every hour - much faster than launching one "
            "process per hour in a shell loop."
        ),
    )
    p.add_argument("--mod29-dir", required=True, help="Directory containing Terra MOD29 (IST) HDF4 files")
    p.add_argument("--myd29-dir", required=True, help="Directory containing Aqua MYD29 (IST) HDF4 files")
    p.add_argument("--mod03-dir", required=True, help="Directory containing Terra MOD03 (geolocation) HDF4 files")
    p.add_argument("--myd03-dir", required=True, help="Directory containing Aqua MYD03 (geolocation) HDF4 files")
    p.add_argument("--out-dir", required=True, help="Output directory for the NetCDF file")
    p.add_argument(
        "--carra2-sample", required=True,
        help="Sample CARRA2 NetCDF file providing the target grid (e.g. carra2_sea.d.nc)",
    )
    p.add_argument("--window-minutes", type=int, default=30, help="Half-width of the merge window (default 30)")
    p.add_argument("--radius-ist", type=float, default=DEFAULT_RADIUS_IST, help="Gaussian resample radius of influence (m)")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Gaussian resample sigma (m)")
    p.add_argument("--neighbours", type=int, default=DEFAULT_NEIGHBOURS, help="Gaussian resample neighbour count")
    p.add_argument("--radius-class", type=float, default=DEFAULT_RADIUS_CLASS, help="Nearest-neighbour resample radius (m)")
    p.add_argument(
        "--domain-margin-deg", type=float, default=2.0,
        help="Lat/lon margin (degrees) added around the CARRA2 domain bbox before cropping each swath (default 2.0)",
    )
    p.add_argument(
        "--nprocs", type=int, default=min(os.cpu_count() or 1, 8),
        help="Number of processes pyresample uses per resample call (default: min(cpu_count, 8))",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def process_one_hour(args: argparse.Namespace, grid: Carra2Grid, hour: int) -> str:
    """Run the full pipeline for a single target hour and return the output path."""
    t_hour0 = time.perf_counter()
    matches, target_dt = find_and_match_files(
        args.year, args.doy, hour,
        args.mod29_dir, args.myd29_dir, args.mod03_dir, args.myd03_dir,
        args.window_minutes,
    )

    resampled: List[ResampledGranule] = []
    for m in matches:
        LOG.info("Resampling %s granule at %s ...", m.satellite, m.obs_dt.isoformat())
        try:
            result = resample_one(
                m.ist_path, m.geo_path, grid,
                radius_of_influence_ist=args.radius_ist,
                sigma=args.sigma,
                neighbours=args.neighbours,
                radius_of_influence_class=args.radius_class,
                nprocs=args.nprocs,
            )
        except Exception:
            LOG.exception("Failed to resample %s / %s - skipping this granule.", m.ist_path, m.geo_path)
            continue
        if result is None:
            continue  # no spatial overlap with the domain - nothing to add
        ist_phys, ist_class = result
        resampled.append(ResampledGranule(m.satellite, m.obs_dt, ist_phys, ist_class))

    combined_phys, combined_class = composite_hourly(
        resampled, target_dt, (grid.y.size, grid.x.size)
    )

    out_name = f"MODIS_IST_CARRA2_{target_dt:%Y%m%d}_{target_dt:%H%M}.nc"
    out_path = os.path.join(args.out_dir, out_name)
    write_output_netcdf(out_path, target_dt, grid, combined_phys, combined_class, matches)
    LOG.info("Wrote %s (hour %02d took %.1fs)", out_path, hour, time.perf_counter() - t_hour0)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    os.makedirs(args.out_dir, exist_ok=True)

    hours = [int(h) for h in args.hours.split(",")] if args.hours else [args.hour]

    t_start = time.perf_counter()
    # Build the CARRA2 grid (and its domain bounding box used for swath
    # cropping) only once, even when processing several hours in this run -
    # avoids repeating that setup cost and, more importantly, avoids
    # relaunching a fresh Python process (with pyhdf/pyresample/netCDF4
    # import overhead) per hour if you loop over --hours instead of calling
    # this script once per hour from a shell/job-array loop.
    grid = build_carra2_grid(args.carra2_sample, domain_margin_deg=args.domain_margin_deg)

    for hour in hours:
        process_one_hour(args, grid, hour)

    LOG.info("Finished %d hour(s) in %.1fs total.", len(hours), time.perf_counter() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
