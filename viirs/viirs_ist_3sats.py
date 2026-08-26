#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viirs_ist_to_carra2.py
======================

Pre-process VIIRS Ice Surface Temperature (IST) L2 swath products
(NSIDC / VIIRS Land SIPS  VJ130 = JPSS-1/NOAA-20, VJ230 = JPSS-2/NOAA-21, VNP30 = SNPP)
into hourly fields on the CARRA2 polar-stereographic grid (2.5 km).

Pipeline
--------
1. Find all VJ130 / VJ230 /VNP30 granules whose observation time falls inside a
   +/- WINDOW minute time slot centred on YEAR / DOY / HOUR.
2. Read IST_Data/IST (ushort, scale_factor 0.01) + IST_Data/IST_Basic_QA and
   Geolocation_Data/{latitude,longitude}.  Keep only pixels that carry a
   *physical* temperature (default 210-273 K); every flag value
   (missing / no_decision / night / land / inland_water / open_ocean,
   bow-tie trim, fill) becomes NaN.
3. Resample each granule (750 m swath) onto the CARRA2 area definition read
   from a CARRA2 sample file, using pyresample Gaussian weighting
   (kd_tree.resample_gauss, with_uncert=True so we also get neighbour counts).
4. Merge all granules of the slot (both satellites, overlapping swaths) with a
   selectable strategy: gaussian-count/time weighted mean, plain mean,
   nearest-in-time, or fixed satellite priority.
5. Write one CF-1.8 netCDF file per hour with a MODIS-like structure
   (time, y, x, latitude, longitude, projection_polar_stereographic, ist_phys)
   with the time coordinate set to the *nominal* hour.

Example
-------
python viirs_ist_to_carra2.py \
    --year 2025 --doy 121 --hour 6 \
    --vj130-dir /data/viirs/VJ130 \
    --vj230-dir /data/viirs/VJ230 \
    --vnp30-dir /data/viirs/VNP30 \
    --carra2-grid /data/grids/carra2_sea.d.nc \
    --outdir /data/out \
    --merge weighted

Performance / memory notes (full 2869 x 2869 CARRA2 grid)
--------------------------------------------------------
* The kd-tree query allocates target_cells x neighbours float64 distances AND
  int64 indices.  For the whole grid at neighbours=32 that is ~4 GB, which is
  why the grid is resampled in row blocks (--block-rows, default 256:
  ~1.9 GB peak, ~5-7 s per block, 12 blocks per granule).
* Keep --nprocs 1.  pyresample's multiprocessing path loses child-process
  exceptions and reports the misleading
      AttributeError: 'bytes' object has no attribute '_object_hook'
  (in pyresample/_spatial_mp.py::_run_jobs) instead of the real error, which is
  usually a MemoryError.
* pyresample's own data reduction (--reduce-data) is OFF by default: its
  lon/lat bounding-box pruning is unreliable for polar sub-areas that span all
  longitudes and silently drops valid swath pixels.
* "UserWarning: Possible more than 32 neighbours within 3000.0 m" is expected
  for 750 m pixels on a 2.5 km grid: only the 32 nearest neighbours are used.
  With sigma=1250 m the discarded ones carry negligible weight (the field
  changes by <0.05 % between neighbours=16 and neighbours=96).

Requires: numpy, netCDF4, pyresample (pyproj).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from netCDF4 import Dataset
from pyresample import geometry
from pyresample.kd_tree import resample_gauss

LOG = logging.getLogger("viirs_ist")

# --------------------------------------------------------------------------- #
# Constants / defaults
# --------------------------------------------------------------------------- #

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

#: IST raw valid_range from the product (21000..31300 -> 210.00..313.00 K)
IST_RAW_VALID = (21000, 31300)
IST_SCALE = 0.01
IST_FILL = 65535

#: Physical window requested by the user
IST_PHYS_MIN = 210.0
IST_PHYS_MAX = 273.0

#: IST_Basic_QA: 0-best, 1-day_good, 2-day_cloud, 3-night_good, 4-night_cloud,
#: 5-other, 6-poor ; 237 inland_water, 253 land, 254 bowtie_trim, 255 fill
DEFAULT_ALLOWED_QA = (0, 1, 3)

#: granule file name:  VJ130.A2025121.0548.002.2025121190705.nc
FNAME_RE = re.compile(
    r"^(?P<short>V[NJ][123P]30)\.A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<hhmm>\d{4})\.(?P<ver>\d{3})\.(?P<prod>\d+)\.nc$"
)

SAT_OF_SHORTNAME = {"VJ130": "JPSS-1", "VJ230": "JPSS-2", "VNP30": "SNPP"}
SAT_CODE = {"JPSS-1": 1, "JPSS-2": 2, "SNPP": 3}


# --------------------------------------------------------------------------- #
# Granule bookkeeping
# --------------------------------------------------------------------------- #

@dataclass
class Granule:
    path: str
    shortname: str            # VJ130 / VJ230 / VNP30
    satellite: str            # JPSS-1 / JPSS-2 / SNPP
    start: datetime           # granule start time (UTC)
    end: datetime             # granule end time (UTC)
    dt_seconds: float = 0.0   # signed offset of granule centre vs. slot centre

    @property
    def centre(self) -> datetime:
        return self.start + (self.end - self.start) / 2


def _parse_granule_name(path: str) -> Granule | None:
    m = FNAME_RE.match(os.path.basename(path))
    #print("_parse_granule_name", m)
    if not m:
        return None
    year = int(m.group("year"))
    doy = int(m.group("doy"))
    hhmm = m.group("hhmm")
    start = (datetime(year, 1, 1, tzinfo=timezone.utc)
             + timedelta(days=doy - 1,
                         hours=int(hhmm[:2]), minutes=int(hhmm[2:])))
    short = m.group("short")
    return Granule(path=path, shortname=short,
                   satellite=SAT_OF_SHORTNAME.get(short, short),
                   start=start, end=start + timedelta(minutes=6))


def _refine_times_from_metadata(g: Granule) -> Granule:
    """Use RangeBeginning*/RangeEnding* global attributes when available."""
    try:
        with Dataset(g.path, "r") as nc:
            bd = getattr(nc, "RangeBeginningDate", None)
            bt = getattr(nc, "RangeBeginningTime", None)
            ed = getattr(nc, "RangeEndingDate", None)
            et = getattr(nc, "RangeEndingTime", None)
        if bd and bt:
            g.start = datetime.strptime(f"{bd} {bt[:8]}",
                                        "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if ed and et:
            g.end = datetime.strptime(f"{ed} {et[:8]}",
                                      "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if g.end <= g.start:                      # granule crossing midnight
            g.end += timedelta(days=1)
    except OSError as exc:
        LOG.warning("cannot read time metadata of %s (%s)", g.path, exc)
    return g


def find_granules(directory: str, shortname: str, slot_centre: datetime,
                  window_minutes: float, use_metadata_times: bool = True
                  ) -> list[Granule]:
    """All granules of one satellite whose *centre* is inside the time slot."""
    if not directory:
        return []
    half = timedelta(minutes=window_minutes)
    t0, t1 = slot_centre - half, slot_centre + half

    # glob the day of t0 and of t1 (slot may cross midnight / year boundary)
    days = {(t0.year, int(t0.strftime("%j"))), (t1.year, int(t1.strftime("%j")))}
    candidates: set[str] = set()
    for year, doy in days:
        pat = os.path.join(directory, f"{shortname}.A{year}{doy:03d}.*.nc")
        candidates.update(glob.glob(pat))
        # also allow files stored in YYYY/DDD/ sub-trees
        candidates.update(glob.glob(os.path.join(
            directory, f"{year}", f"{doy:03d}", f"{shortname}.A{year}{doy:03d}.*.nc")))

    out: list[Granule] = []
    for path in sorted(candidates):
        g = _parse_granule_name(path)
        if g is None:
            continue
        if use_metadata_times:
            g = _refine_times_from_metadata(g)
        if t0 <= g.centre <= t1:
            g.dt_seconds = (g.centre - slot_centre).total_seconds()
            out.append(g)
        else:
            LOG.debug("skip %s (centre %s outside slot)",
                      os.path.basename(path), g.centre)
    out.sort(key=lambda g: (abs(g.dt_seconds), g.shortname))
    return out


# --------------------------------------------------------------------------- #
# Reading VIIRS IST
# --------------------------------------------------------------------------- #

def read_viirs_ist(path: str,
                   allowed_qa: tuple[int, ...] = DEFAULT_ALLOWED_QA,
                   use_qa: bool = True,
                   ist_min: float = IST_PHYS_MIN,
                   ist_max: float = IST_PHYS_MAX,
                   variable: str = "IST"
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ist_phys, lat, lon) as float32/float64 arrays with NaN for
    everything that is not a physically meaningful ice surface temperature.

    Auto mask/scale is switched OFF so that the ushort flag values
    (0, 1, 11, 25, 37, 39, 50) and the fill value can be handled explicitly.
    """
    with Dataset(path, "r") as nc:
        geo = nc.groups["Geolocation_Data"]
        istg = nc.groups["IST_Data"]

        for v in (geo.variables["latitude"], geo.variables["longitude"]):
            v.set_auto_maskandscale(False)
        lat = np.asarray(geo.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(geo.variables["longitude"][:], dtype=np.float64)
        lat_fill = float(getattr(geo.variables["latitude"], "_FillValue", -999.0))
        lon_fill = float(getattr(geo.variables["longitude"], "_FillValue", -999.0))

        var = istg.variables[variable]
        var.set_auto_maskandscale(False)
        raw = np.asarray(var[:])
        scale = float(getattr(var, "scale_factor", IST_SCALE))
        vrange = getattr(var, "valid_range", IST_RAW_VALID)
        vmin, vmax = int(vrange[0]), int(vrange[1])

        qa = None
        if use_qa and "IST_Basic_QA" in istg.variables:
            qav = istg.variables["IST_Basic_QA"]
            qav.set_auto_maskandscale(False)
            qa = np.asarray(qav[:])

    # trim to the common shape (defensive, as in the MODIS version)
    ny = min(raw.shape[0], lat.shape[0], lon.shape[0])
    nx = min(raw.shape[1], lat.shape[1], lon.shape[1])
    raw, lat, lon = raw[:ny, :nx], lat[:ny, :nx], lon[:ny, :nx]
    if qa is not None:
        qa = qa[:ny, :nx]

    # --- geolocation mask (bow-tie trim rows have fill lat/lon) ------------- #
    bad_geo = (
        ~np.isfinite(lat) | ~np.isfinite(lon)
        | np.isclose(lat, lat_fill) | np.isclose(lon, lon_fill)
        | (np.abs(lat) > 90.0) | (np.abs(lon) > 180.0)
    )

    # --- physical IST ------------------------------------------------------- #
    good = (raw >= vmin) & (raw <= vmax)          # excludes flags & fill
    ist = np.full(raw.shape, np.nan, dtype=np.float32)
    ist[good] = raw[good].astype(np.float32) * scale
    with np.errstate(invalid="ignore"):
        ist[(ist < ist_min) | (ist > ist_max)] = np.nan

    # --- QA filtering ------------------------------------------------------- #
    if qa is not None:
        qa_ok = np.isin(qa, np.asarray(allowed_qa, dtype=qa.dtype))
        ist[~qa_ok] = np.nan

    ist[bad_geo] = np.nan
    # pyresample cannot handle NaN geolocation -> substitute harmless values
    lat = np.where(bad_geo, 0.0, lat)
    lon = np.where(bad_geo, 0.0, lon)
    return ist, lat, lon


# --------------------------------------------------------------------------- #
# CARRA2 target grid
# --------------------------------------------------------------------------- #

@dataclass
class TargetGrid:
    area_def: geometry.AreaDefinition
    x: np.ndarray                 # as stored in the CARRA2 file (1-D, m)
    y: np.ndarray
    lat: np.ndarray               # as stored in the CARRA2 file (2-D)
    lon: np.ndarray
    proj4: str
    y_ascending: bool             # True -> file rows run south -> north
    proj_attrs: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape


def read_carra2_grid(path: str) -> TargetGrid:
    """Build a pyresample AreaDefinition from the CARRA2 sample file."""
    with Dataset(path, "r") as nc:
        x = np.asarray(nc.variables["x"][:], dtype=np.float64)
        y = np.asarray(nc.variables["y"][:], dtype=np.float64)
        lat = np.asarray(nc.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(nc.variables["longitude"][:], dtype=np.float64)

        pvar_name = None
        for cand in ("projection_polar_stereographic", "projection_lambert",
                     "crs", "polar_stereographic"):
            if cand in nc.variables:
                pvar_name = cand
                break
        if pvar_name is None:
            raise KeyError("no grid-mapping variable found in %s" % path)
        pvar = nc.variables[pvar_name]
        proj_attrs = {a: pvar.getncattr(a) for a in pvar.ncattrs()}
        proj4 = proj_attrs.get(
            "proj4",
            "+proj=stere +lat_0={lat0} +lon_0={lon0} +lat_ts={latts} "
            "+no_defs +R={R}".format(
                lat0=proj_attrs.get("latitude_of_projection_origin", 90.0),
                lon0=proj_attrs.get("straight_vertical_longitude_from_pole", -30.0),
                latts=proj_attrs.get("latitude_of_projection_origin", 90.0),
                R=proj_attrs.get("earth_radius", 6371000.0)))

    ny, nx = lat.shape
    dx = float(np.abs(np.diff(x)).mean())
    dy = float(np.abs(np.diff(y)).mean())
    y_ascending = bool(y[-1] > y[0])

    # pyresample area_extent = (ll_x, ll_y, ur_x, ur_y) of the outer cell edges
    area_extent = (float(min(x) - dx / 2.0), float(min(y) - dy / 2.0),
                   float(max(x) + dx / 2.0), float(max(y) + dy / 2.0))

    area_def = geometry.AreaDefinition(
        area_id="carra2", description="CARRA2 polar stereographic",
        proj_id="carra2", projection=proj4,
        width=nx, height=ny, area_extent=area_extent)

    LOG.info("CARRA2 grid: %d x %d, dx=%.1f m, dy=%.1f m, y_ascending=%s",
             nx, ny, dx, dy, y_ascending)
    return TargetGrid(area_def=area_def, x=x, y=y, lat=lat, lon=lon,
                      proj4=proj4, y_ascending=y_ascending,
                      proj_attrs=proj_attrs)


# --------------------------------------------------------------------------- #
# Resampling one granule
# --------------------------------------------------------------------------- #

def _block_area(area_def: geometry.AreaDefinition, r0: int, r1: int
                ) -> geometry.AreaDefinition:
    """Sub-AreaDefinition covering target rows [r0, r1).

    pyresample rows run north -> south, so row r spans
    y in [ur_y - (r+1)*dy , ur_y - r*dy].
    """
    ll_x, ll_y, ur_x, ur_y = area_def.area_extent
    dy = (ur_y - ll_y) / area_def.height
    # .crs on pyresample >= 1.15, .proj_dict on older releases
    proj = getattr(area_def, "crs", None) or area_def.proj_dict
    return geometry.AreaDefinition(
        area_id=area_def.area_id + "_blk",
        description=area_def.description,
        proj_id=area_def.proj_id,
        projection=proj,
        width=area_def.width, height=int(r1 - r0),
        area_extent=(ll_x, ur_y - r1 * dy, ur_x, ur_y - r0 * dy))


def resample_one(granule: Granule, grid: TargetGrid, args
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """Resample one granule to the CARRA2 grid.

    Returns (ist_grid, count_grid) in *pyresample row order*
    (row 0 = northernmost), or None when the granule has no usable data.
    """
    ist, lat, lon = read_viirs_ist(
        granule.path, allowed_qa=tuple(args.allowed_qa), use_qa=not args.no_qa,
        ist_min=args.ist_min, ist_max=args.ist_max, variable=args.ist_variable)

    n_valid = int(np.isfinite(ist).sum())
    if n_valid == 0:
        LOG.info("  %s: no valid IST pixels", os.path.basename(granule.path))
        return None

    swath_def = geometry.SwathDefinition(lons=lon, lats=lat)

    # ------------------------------------------------------------------ #
    # Gaussian resampling of gappy data.
    #
    # Passing a masked array to resample_gauss is NOT usable here: pyresample
    # also resamples the mask, so every target cell whose neighbourhood
    # touches one cloud/land/bow-tie pixel becomes masked - with a realistic
    # `neighbours` setting that wipes out essentially the whole field.
    #
    # Instead we resample two channels with identical weights,
    #     ch0 = IST (invalid pixels set to 0)
    #     ch1 = validity indicator (1 where IST is valid, else 0)
    # resample_gauss returns  sum(w*ch)/sum(w)  over ALL neighbours, hence
    #     ch0_out / ch1_out = sum(w*IST)/sum(w)  over the VALID neighbours,
    # i.e. exactly the Gaussian-weighted mean of the good observations, while
    # ch1_out is the Gaussian-weighted fraction of valid neighbours and is a
    # natural quality/coverage measure.
    # ------------------------------------------------------------------ #
    valid = np.isfinite(ist)
    stack = np.empty(ist.shape + (2,), dtype=np.float64)
    stack[..., 0] = np.where(valid, ist, 0.0)
    stack[..., 1] = valid.astype(np.float64)

    # The target grid is processed in row blocks.  The kd-tree query allocates
    # (target_cells x neighbours) float64 distances AND int64 indices, i.e. for
    # the full 2869 x 2869 CARRA2 grid with 32 neighbours already ~4 GB - which
    # is what makes pyresample die inside its (buggy) multiprocessing error
    # handler.  Blocking bounds the peak memory and, together with
    # reduce_data=True, also speeds things up because each block only sees the
    # part of the swath that can reach it.
    n_rows = grid.area_def.height
    block = max(1, int(args.block_rows)) if args.block_rows > 0 else n_rows

    num = np.zeros((n_rows, grid.area_def.width), dtype=np.float64)
    frac = np.zeros_like(num)
    n_neigh = np.zeros_like(num)

    for r0 in range(0, n_rows, block):
        r1 = min(r0 + block, n_rows)
        sub_area = _block_area(grid.area_def, r0, r1)
        try:
            values, _stddev, counts = resample_gauss(
                swath_def, stack, sub_area,
                radius_of_influence=args.radius_of_influence,
                sigmas=[args.sigma, args.sigma],
                neighbours=args.neighbours,
                fill_value=0.0,
                with_uncert=True,
                reduce_data=args.reduce_data,
                nprocs=args.nprocs,
            )
        except AttributeError as exc:
            # pyresample._spatial_mp._run_jobs mangles child-process errors
            # ("'bytes' object has no attribute '_object_hook'").
            raise RuntimeError(
                "pyresample failed inside its multiprocessing worker and hid "
                "the real error (%s). Re-run with --nprocs 1 and, if it is a "
                "MemoryError, lower --block-rows / --neighbours." % exc
            ) from exc
        except MemoryError as exc:
            raise MemoryError(
                "kd-tree query ran out of memory for rows %d:%d - lower "
                "--block-rows (now %d) or --neighbours (now %d)"
                % (r0, r1, block, args.neighbours)) from exc

        values = np.ma.filled(np.asarray(values, dtype=np.float64), 0.0)
        counts = np.ma.filled(np.asarray(counts), 0).astype(np.float64)
        num[r0:r1] = values[..., 0]
        frac[r0:r1] = values[..., 1]
        n_neigh[r0:r1] = counts[..., 0]
        LOG.debug("    rows %5d:%-5d done", r0, r1)

    ist_grid = np.full(frac.shape, np.nan, dtype=np.float32)
    ok = (frac >= args.min_valid_fraction) & (n_neigh > 0)
    ist_grid[ok] = (num[ok] / frac[ok]).astype(np.float32)

    # effective number of good source pixels behind each target cell
    eff_counts = np.rint(n_neigh * frac).astype(np.int32)
    eff_counts[~ok] = 0
    ok &= eff_counts >= args.min_valid_pixels
    ist_grid[~ok] = np.nan
    eff_counts[~ok] = 0
    counts = eff_counts

    LOG.info("  %s (%s, dt=%+.1f min): %d swath px -> %d grid cells",
             os.path.basename(granule.path), granule.satellite,
             granule.dt_seconds / 60.0, n_valid,
             int(np.isfinite(ist_grid).sum()))
    return ist_grid, counts


# --------------------------------------------------------------------------- #
# Merging granules / satellites inside the hourly slot
# --------------------------------------------------------------------------- #

def time_weight(dt_seconds: float, tau_minutes: float) -> float:
    """Gaussian temporal weight, 1.0 at the nominal hour."""
    if tau_minutes <= 0:
        return 1.0
    return float(np.exp(-0.5 * (dt_seconds / (tau_minutes * 60.0)) ** 2))


def merge_slot(per_granule: list[tuple[Granule, np.ndarray, np.ndarray]],
               args) -> dict[str, np.ndarray]:
    """Combine the resampled granules of one hourly slot.

    Strategies (--merge):
      weighted : mean weighted by (Gaussian neighbour count) x (temporal weight)
                 x (satellite weight).  Default - smooth, uses every
                 observation, gives most weight to obs closest to the hour.
      mean     : plain arithmetic mean of all granules that saw the cell.
      nearest  : value of the granule whose centre time is closest to the
                 nominal hour ("winner takes all", keeps native values).
      priority : first satellite in --sat-priority wins where both observe;
                 within a satellite the nearest-in-time granule wins.
    """
    ny, nx = per_granule[0][1].shape
    ist = np.full((ny, nx), np.nan, dtype=np.float32)
    num = np.zeros((ny, nx), dtype=np.float64)
    den = np.zeros((ny, nx), dtype=np.float64)
    n_obs = np.zeros((ny, nx), dtype=np.int32)
    dt_num = np.zeros((ny, nx), dtype=np.float64)
    sat_flag = np.zeros((ny, nx), dtype=np.int16)      # 0 none, 1 J1, 2 J2, 3 both
    best_rank = np.full((ny, nx), np.inf, dtype=np.float64)
    src_sat = np.zeros((ny, nx), dtype=np.int16)

    sat_prio = [s.strip().upper() for s in args.sat_priority.split(",")]
    sat_weights = {"JPSS-1": args.weight_jpss1, "JPSS-2": args.weight_jpss2}

    def prio_index(sat: str) -> int:
        key = SAT_OF_SHORTNAME.get(sat, sat).upper()
        return sat_prio.index(key) if key in sat_prio else len(sat_prio)

    for g, values, counts in per_granule:
        ok = np.isfinite(values)
        if not ok.any():
            continue
        n_obs[ok] += 1
        sat_flag[ok] |= SAT_CODE.get(g.satellite, 0)
        dt_num[ok] += g.dt_seconds

        wt = time_weight(g.dt_seconds, args.tau_minutes)
        wsat = sat_weights.get(g.satellite, 1.0)

        if args.merge == "mean":
            w = np.where(ok, 1.0, 0.0)
        elif args.merge == "weighted":
            w = np.where(ok, np.maximum(counts, 1).astype(np.float64) * wt * wsat, 0.0)
        else:
            w = None

        if w is not None:
            num[ok] += w[ok] * values[ok].astype(np.float64)
            den[ok] += w[ok]
        else:
            if args.merge == "nearest":
                rank = abs(g.dt_seconds)
            else:                                    # priority
                rank = prio_index(g.satellite) * 1e6 + abs(g.dt_seconds)
            better = ok & (rank < best_rank)
            ist[better] = values[better]
            best_rank[better] = rank
            src_sat[better] = SAT_CODE.get(g.satellite, 0)

    if args.merge in ("mean", "weighted"):
        good = den > 0
        ist[good] = (num[good] / den[good]).astype(np.float32)
        ist[~good] = np.nan
        src_sat = sat_flag.copy()

    with np.errstate(invalid="ignore"):
        ist[(ist < args.ist_min) | (ist > args.ist_max)] = np.nan

    dt_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    m = n_obs > 0
    dt_mean[m] = (dt_num[m] / n_obs[m]).astype(np.float32)

    return {"ist_phys": ist, "num_obs": n_obs, "obs_time_offset": dt_mean,
            "source_satellite": src_sat.astype(np.int16)}


# --------------------------------------------------------------------------- #
# Output netCDF
# --------------------------------------------------------------------------- #

def orient_to_file(field2d: np.ndarray, grid: TargetGrid) -> np.ndarray:
    """pyresample rows go north->south; flip if the CARRA2 y axis ascends."""
    return field2d[::-1, :] if grid.y_ascending else field2d


def write_output(outpath: str, slot_centre: datetime, grid: TargetGrid,
                 merged: dict[str, np.ndarray],
                 granules: list[Granule], args) -> None:
    ny, nx = grid.shape
    tval = (slot_centre - EPOCH).total_seconds()

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with Dataset(outpath, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)

        v = nc.createVariable("time", "f8", ("time",))
        v.standard_name = "time"
        v.long_name = "time"
        v.units = "seconds since 1970-01-01 00:00:00 +00:00"
        v.calendar = "standard"
        v.axis = "T"
        v.comment = ("nominal validity time of the hourly slot; all VIIRS "
                     "observations within +/- %g minutes are assigned to it"
                     % args.window)
        v[:] = [tval]

        v = nc.createVariable("x", "f8", ("x",))
        v.standard_name = "projection_x_coordinate"
        v.long_name = "x-coordinate in Cartesian system"
        v.units = "m"
        v.axis = "X"
        v[:] = grid.x

        v = nc.createVariable("y", "f8", ("y",))
        v.standard_name = "projection_y_coordinate"
        v.long_name = "y-coordinate in Cartesian system"
        v.units = "m"
        v.axis = "Y"
        v[:] = grid.y

        v = nc.createVariable("longitude", "f8", ("y", "x"), zlib=args.zlib)
        v.standard_name = "longitude"
        v.long_name = "longitude"
        v.units = "degree_east"
        v[:] = grid.lon

        v = nc.createVariable("latitude", "f8", ("y", "x"), zlib=args.zlib)
        v.standard_name = "latitude"
        v.long_name = "latitude"
        v.units = "degree_north"
        v[:] = grid.lat

        p = nc.createVariable("projection_polar_stereographic", "i4")
        defaults = {
            "grid_mapping_name": "polar_stereographic",
            "scale_factor_at_projection_origin": 1.0,
            "straight_vertical_longitude_from_pole": -30.0,
            "latitude_of_projection_origin": 90.0,
            "earth_radius": 6371000.0,
            "proj4": grid.proj4,
        }
        for k, dv in defaults.items():
            p.setncattr(k, grid.proj_attrs.get(k, dv))

        v = nc.createVariable("ist_phys", "f4", ("time", "y", "x"),
                              zlib=args.zlib, complevel=4,
                              fill_value=np.float32(np.nan))
        v.standard_name = "surface_temperature"
        v.long_name = "Ice Surface Temperature"
        v.units = "K"
        v.valid_range = np.array([args.ist_min, args.ist_max], dtype="f4")
        v.coordinates = "latitude longitude"
        v.grid_mapping = "projection_polar_stereographic"
        v.cell_methods = "time: point area: mean"
        v[0, :, :] = orient_to_file(merged["ist_phys"], grid)

        if args.extra_vars:
            v = nc.createVariable("num_obs", "i2", ("time", "y", "x"),
                                  zlib=args.zlib, fill_value=np.int16(-1))
            v.long_name = "number of VIIRS granules contributing to the grid cell"
            v.units = "1"
            v.coordinates = "latitude longitude"
            v.grid_mapping = "projection_polar_stereographic"
            v[0, :, :] = orient_to_file(
                merged["num_obs"].astype(np.int16), grid)

            v = nc.createVariable("obs_time_offset", "f4", ("time", "y", "x"),
                                  zlib=args.zlib,
                                  fill_value=np.float32(np.nan))
            v.long_name = ("mean offset of the contributing observation times "
                           "relative to the nominal hour")
            v.units = "s"
            v.coordinates = "latitude longitude"
            v.grid_mapping = "projection_polar_stereographic"
            v[0, :, :] = orient_to_file(merged["obs_time_offset"], grid)

            v = nc.createVariable("source_satellite", "i2", ("time", "y", "x"),
                                  zlib=args.zlib, fill_value=np.int16(0))
            v.long_name = "satellite(s) contributing to the grid cell"
            v.flag_values = np.array([0, 1, 2, 3], dtype="i2")
            v.flag_meanings = "none JPSS-1 JPSS-2 both"
            v.coordinates = "latitude longitude"
            v.grid_mapping = "projection_polar_stereographic"
            v[0, :, :] = orient_to_file(merged["source_satellite"], grid)

        # ---------------- global attributes ---------------- #
        nc.Conventions = "CF-1.8"
        nc.title = "VIIRS VJ130/VJ230/VNP30 IST on CARRA2 polar stereographic grid"
        nc.source = ("Resampled from VIIRS Ice Surface Temperature L2 swath "
                     "netCDF (VJ130/VJ230/VNP30) with pyresample")
        nc.institution = "Norwegian Meteorological Institute, MET Norway"
        nc.summary = ("Hourly VIIRS ice surface temperature composited from "
                      "JPSS-1 (VJ130) and JPSS-2 (VJ230), "
                      "SNPP (VNP30) 750 m L2 swaths"
                      "resampled to the CARRA2 2.5 km grid.")
        nc.time_slot_centre = slot_centre.strftime("%Y-%m-%dT%H:%M:%SZ")
        nc.time_window_minutes = float(args.window)
        nc.merge_method = args.merge
        nc.satellite_priority = args.sat_priority
        nc.temporal_weight_tau_minutes = float(args.tau_minutes)
        nc.ist_physical_range = "%.2f %.2f K" % (args.ist_min, args.ist_max)
        nc.ist_source_variable = "IST_Data/%s" % args.ist_variable
        nc.qa_filter = ("IST_Basic_QA in {%s}" % ",".join(map(str, args.allowed_qa))
                        if not args.no_qa else "none")
        nc.resampling = ("pyresample kd_tree.resample_gauss "
                         "(radius_of_influence=%g m, sigma=%g m, neighbours=%d, "
                         "min_valid_fraction=%g, min_valid_pixels=%d)"
                         % (args.radius_of_influence, args.sigma,
                            args.neighbours, args.min_valid_fraction,
                            args.min_valid_pixels))
        nc.number_of_input_granules = len(granules)
        nc.input_granules = ", ".join(os.path.basename(g.path) for g in granules)
        nc.history = "%s: created by viirs_ist_to_carra2.py" % \
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG.info("wrote %s", outpath)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-process VIIRS (VJ130/VJ230/VNP30) IST swaths into hourly "
                    "fields on the CARRA2 polar-stereographic grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--year", type=int, required=True, help="year, e.g. 2025")
    p.add_argument("--doy", "--day-of-year", type=int, required=True,
                   dest="doy", help="day of year (1-366)")
    p.add_argument("--hour", type=int, required=True,
                   help="nominal UTC hour of the output slot (0-23)")
    p.add_argument("--vj130-dir", default="",
                   help="directory with JPSS-1 VJ130 IST netCDF files")
    p.add_argument("--vj230-dir", default="",
                   help="directory with JPSS-2 VJ230 IST netCDF files")
    p.add_argument("--vnp30-dir", default="",
                   help="directory with SNPP VNP30 IST netCDF files")
    p.add_argument("--carra2-grid", required=True,
                   help="CARRA2 sample netCDF defining the target grid "
                        "(e.g. carra2_sea.d.nc)")
    p.add_argument("--outdir", required=True, help="output directory")
    p.add_argument("--outfile", default=None,
                   help="explicit output file name (default: "
                        "VIIRS_IST_CARRA2_YYYYMMDD_HH00.nc)")

    # time slot
    p.add_argument("--window", type=float, default=30.0,
                   help="half width of the time window in minutes (+/-)")
    p.add_argument("--tau-minutes", type=float, default=20.0,
                   help="e-folding time of the temporal weight "
                        "(merge=weighted); <=0 disables temporal weighting")
    p.add_argument("--filename-times-only", action="store_true",
                   help="trust granule file names, do not open files to read "
                        "RangeBeginning/Ending attributes")

    # physics / QA
    p.add_argument("--ist-min", type=float, default=IST_PHYS_MIN,
                   help="minimum accepted physical IST [K]")
    p.add_argument("--ist-max", type=float, default=IST_PHYS_MAX,
                   help="maximum accepted physical IST [K]")
    p.add_argument("--ist-variable", default="IST", choices=["IST", "IST_map"],
                   help="source variable inside group IST_Data")
    p.add_argument("--allowed-qa", default=",".join(map(str, DEFAULT_ALLOWED_QA)),
                   help="comma separated IST_Basic_QA values to keep "
                        "(0-best,1-day_good,2-day_cloud,3-night_good,"
                        "4-night_cloud,5-other,6-poor)")
    p.add_argument("--no-qa", action="store_true",
                   help="do not filter on IST_Basic_QA")

    # resampling
    p.add_argument("--radius-of-influence", type=float, default=3000.0,
                   help="pyresample radius of influence [m] "
                        "(~1.2 x target grid spacing works well for 750 m -> 2.5 km)")
    p.add_argument("--sigma", type=float, default=1250.0,
                   help="Gaussian sigma [m], typically half the target spacing")
    p.add_argument("--neighbours", type=int, default=32,
                   help="max. number of swath neighbours per target cell; must "
                        "exceed pi*R^2/(750 m)^2 (~50 for R=3000 m) or "
                        "pyresample warns about truncated neighbourhoods")
    p.add_argument("--nprocs", type=int, default=1,
                   help="processes for the kd-tree query. Keep at 1: "
                        "pyresample's multiprocessing path swallows child "
                        "errors and raises a misleading "
                        "\"'bytes' object has no attribute '_object_hook'\"")
    p.add_argument("--reduce-data", action="store_true",
                   help="enable pyresample source-data reduction. OFF by "
                        "default: its lon/lat bounding-box pruning is not "
                        "reliable for polar sub-areas spanning all longitudes "
                        "and silently drops valid swath pixels")
    p.add_argument("--block-rows", type=int, default=256,
                   help="number of CARRA2 rows resampled per kd-tree query; "
                        "bounds peak memory (roughly "
                        "block_rows * width * neighbours * 16 bytes). "
                        "0 = whole grid at once")
    p.add_argument("--min-valid-fraction", type=float, default=0.10,
                   help="minimum Gaussian-weighted fraction of valid (clear, "
                        "in-range) swath neighbours required to fill a target "
                        "cell; larger values give fewer but more reliable cells")
    p.add_argument("--min-valid-pixels", type=int, default=1,
                   help="minimum effective number of valid swath pixels per "
                        "target cell")

    # merging
    p.add_argument("--merge", default="weighted",
                   choices=["weighted", "mean", "nearest", "priority"],
                   help="how to combine granules/satellites in the slot")
    p.add_argument("--sat-priority", default="JPSS-2,JPSS-1",
                   help="satellite priority order for merge=priority")
    p.add_argument("--weight-jpss1", type=float, default=1.0,
                   help="extra weight for JPSS-1 (merge=weighted)")
    p.add_argument("--weight-jpss2", type=float, default=1.0,
                   help="extra weight for JPSS-2 (merge=weighted)")

    # output options
    p.add_argument("--no-extra-vars", dest="extra_vars", action="store_false",
                   help="write only ist_phys (no num_obs / obs_time_offset / "
                        "source_satellite diagnostics)")
    p.add_argument("--no-zlib", dest="zlib", action="store_false",
                   help="disable netCDF compression")
    p.add_argument("--allow-empty", action="store_true",
                   help="write an all-NaN file when no granule is found")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(extra_vars=True, zlib=True)

    args = p.parse_args(argv)
    args.allowed_qa = tuple(int(t) for t in str(args.allowed_qa).split(",")
                            if str(t).strip() != "")
    if not (0 <= args.hour <= 23):
        p.error("--hour must be in 0..23")
    if not (1 <= args.doy <= 366):
        p.error("--doy must be in 1..366")
    if not args.vj130_dir and not args.vj230_dir and not args.vnp30_dir:
        p.error("at least one of --vj130-dir / --vj230-dir / --vnp30-dir must be given")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")

    slot_centre = (datetime(args.year, 1, 1, tzinfo=timezone.utc)
                   + timedelta(days=args.doy - 1, hours=args.hour))
    LOG.info("target slot: %s (+/- %g min)",
             slot_centre.strftime("%Y-%m-%d %H:%M UTC"), args.window)

    grid = read_carra2_grid(args.carra2_grid)
    if args.block_rows > 0:
        mb = (min(args.block_rows, grid.shape[0]) * grid.shape[1]
              * args.neighbours * 16) / 1024.0 ** 2
        LOG.info("kd-tree blocks: %d rows -> ~%.0f MB peak per query "
                 "(neighbours=%d, nprocs=%d)",
                 args.block_rows, mb, args.neighbours, args.nprocs)

    granules: list[Granule] = []
    for directory, short in ((args.vj130_dir, "VJ130"), (args.vj230_dir, "VJ230"), (args.vnp30_dir, "VNP30")):
        found = find_granules(directory, short, slot_centre, args.window,
                              use_metadata_times=not args.filename_times_only)
        LOG.info("%s: %d granule(s) in the slot", short, len(found))
        granules.extend(found)

    if not granules and not args.allow_empty:
        LOG.error("no granules found for the requested slot - nothing to do "
                  "(use --allow-empty to write an empty field)")
        return 2

    per_granule = []
    for g in granules:
        try:
            res = resample_one(g, grid, args)
        except (OSError, KeyError, ValueError) as exc:
            LOG.warning("granule %s failed: %s", os.path.basename(g.path), exc)
            continue
        if res is not None:
            per_granule.append((g, res[0], res[1]))

    if per_granule:
        merged = merge_slot(per_granule, args)
    else:
        ny, nx = grid.shape
        LOG.warning("no usable data in the slot - writing empty field")
        merged = {"ist_phys": np.full((ny, nx), np.nan, np.float32),
                  "num_obs": np.zeros((ny, nx), np.int32),
                  "obs_time_offset": np.full((ny, nx), np.nan, np.float32),
                  "source_satellite": np.zeros((ny, nx), np.int16)}

    n_cells = int(np.isfinite(merged["ist_phys"]).sum())
    LOG.info("merged (%s): %d grid cells with IST (%.3f %% of the domain)",
             args.merge, n_cells,
             100.0 * n_cells / merged["ist_phys"].size)
    if n_cells:
        vals = merged["ist_phys"][np.isfinite(merged["ist_phys"])]
        LOG.info("IST statistics: min=%.2f mean=%.2f max=%.2f K",
                 vals.min(), vals.mean(), vals.max())

    fname = args.outfile or ("VIIRS_IST_CARRA2_%s_%02d00.nc"
                             % (slot_centre.strftime("%Y%m%d"), args.hour))
    write_output(os.path.join(args.outdir, fname), slot_centre, grid,
                 merged, [g for g, _, _ in per_granule] or granules, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
