# awfinfra-satellite-dataset
This is repository to pre-process satellite observation for ML training with Anemoi

1. Clone the repository
```bash
git clone git@github.com:matemile/awfinfra-satellite-dataset.git
```

2. MODIS pre-processing (under modis folder) as an example
```bash
for y in ${year}; do for doy in ${dlist}; do for h in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do python3 modis_ist_preprocess.py --year ${y} --doy ${doy} --hour ${h} --mod29-dir /path/to/MOD29/${y} --myd29-dir /path/to/MYD29/${y} --mod03-dir /path/to/MOD03/${y} --myd03-dir /path/to/MYD03/${y} --out-dir /path/to/data/netcdf/MODIS --carra2-sample /path/to/carra.nc; done; done; done
```

3. VIIRS pre-processing (under viirs folder) as an example
```bash
for y in ${year}; do for doy in ${dlist}; do for h in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do python3 viirs_ist_3sats.py --year ${y} --doy ${doy} --hour ${h} --vj130-dir /path/to/VJ130/${y} --vj230-dir /path/to/VJ230/${y} --vnp30-dir /path/to/VNP30/${y} --carra2-grid /path/to/carra.nc --outdir /path/to/data/netcdf/VIIRS --merge weighted; done; done; done
```

4. Check missing dates for building zarr dataset
```bash
python3 check-date-time-missing.py
```

5. Config yaml for MODIS ZARR dataset generation
modis-1h-v3.yaml
