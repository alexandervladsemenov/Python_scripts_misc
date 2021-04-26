#!/usr/bin/python3
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import stat
import time
import tempfile
import numpy as np
import hashlib
import os
import shutil
import logging
import re
import netCDF4
import errno
import subprocess

import matplotlib as mpl
mpl.use('Agg')  # Prevents the "QXcbConnection: Could not connect to display" error
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import xml.etree.ElementTree as ET
import io
from PIL import Image

import db_setup

# inherit airflow logger if it exists
logger = logging.root.manager.loggerDict.get('airflow.task', logging.getLogger(__name__))
pysquam_dir = os.path.dirname(__file__)

valid_squam_reftarg_pairs = ['CMC_reg',
                             'CMC_deb',
                             'OSTIA_reg',
                             'OSTIA_deb',
                             'REY_reg',
                             'REY_deb',
                             'REF_reg',
                             'REF_deb',
                             'CCI_reg',
                             'CCI_deb',
                             'SST_reg',
                             'SST_deb',
                             'IQ-DR-TM_reg',
                             'IQ-DR-TM_deb',
                             'IQ-AG_reg',
                             'IQ-AG_deb',
                             'IQ-DR-TM-SH_reg',
                             'IQ-DR-TM-SH_deb', ]

valid_micros_reftarg_pairs = ['BT_IR37',
                              'BT_IR86',
                              'BT_IR10',
                              'BT_IR11',
                              'BT_IR12',
                              'IR37_RTM',
                              'IR86_RTM',
                              'IR10_RTM',
                              'IR11_RTM',
                              'IR12_RTM',
                              'SNS_reg',
                              'SNS_hyb',
                              'REF_reg',
                              'REF_hyb']

all_valid_reftarg_pairs = valid_squam_reftarg_pairs + valid_micros_reftarg_pairs

valid_sensors = ('VIIRS', 'MODIS', 'FRAC', 'GAC', 'GHRR', 'ATSR', 'ACSENS')
valid_aggregations = ('daily', 'monthly', 'yearly')
valid_coverages = ('NGT', 'DAY')
valid_products = ('L2P', 'L3U', 'L3C', 'L3U1KM', 'L3S',)
valid_processors = ('ACSPO', 'NAVO', 'CCI', 'PATHFINDER')

MISA_to_db_plat = {
    'NOAA01': 'N01',
    'NOAA02': 'N02',
    'NOAA03': 'N03',
    'NOAA04': 'N04',
    'NOAA05': 'N05',
    'NOAA06': 'N06',
    'NOAA07': 'N07',
    'NOAA08': 'N08',
    'NOAA09': 'N09',
    'NOAA10': 'N10',
    'NOAA11': 'N11',
    'NOAA12': 'N12',
    'NOAA13': 'N13',
    'NOAA14': 'N14',
    'NOAA15': 'N15',
    'NOAA16': 'N16',
    'NOAA17': 'N17',
    'NOAA18': 'N18',
    'NOAA19': 'N19',
}
db_to_MISA_plat = {v: k for k, v in MISA_to_db_plat.items()}

MISA_to_db_sensor = {
    'GHRR': 'GAC',
}
db_to_MISA_sensor = {v: k for k, v in MISA_to_db_sensor.items()}


# this is an interface to cython functions that aggregate/smooth maps arrays
def downres_maps(mean, sdev, clear, masked, lat, lon):

    try:
        import cython_common
    except ImportError as e:
        logger.error('Unable to import cython functions from cython commmon. You must run setup.py to compile the cython code.')
        logger.error('Reason: {}'.format(str(e)))
        raise

    assert mean.shape == sdev.shape
    assert mean.shape == clear.shape
    assert mean.shape == clear.shape

    if masked is not None:
        assert mean.shape == masked.shape

    w = np.where(clear > 1, clear, 0.0).astype(np.float32)

    assert mean.dtype == np.float32
    assert sdev.dtype == np.float32
    assert clear.dtype == np.int64

    if masked is not None:
        assert masked.dtype == np.int64
    
    mean_lowres = cython_common.downres_float_cython(mean, w)
    sdev_lowres = cython_common.downres_float_cython(sdev, w)
    clear_lowres = cython_common.downres_int_cython(clear)
    
    if masked is not None:
        masked_lowres = cython_common.downres_int_cython(masked)
    else:
        masked_lowres = None

    Ny, Nx = mean.shape
    lat_lowres = np.linspace(lat[0], lat[-1], Ny)
    lon_lowres = np.linspace(lon[0], lon[-1], Nx)
    
    return mean_lowres, sdev_lowres, clear_lowres, masked_lowres, lat_lowres, lon_lowres


def smooth_map_array(A):

    Ny_old, Nx_old = A.shape

    if Ny_old % 2 == 0:
        Ny = Ny_old//2
    else:
        Ny = (Ny_old-1)//2 + 1

    if Nx_old % 2 == 0:
        Nx = Nx_old//2
    else:
        Nx = (Nx_old-1)//2 + 1

    A_downres = np.zeros(shape=(Ny, Nx), dtype=np.float)
    w = np.where(A == A, 1.0, 0.0)
    A_copy = np.where(A == A, A, 0.0)

    for ny in range(0, Ny):
        ny_sh = 2*ny
        ny_sh_pl = min(ny_sh + 1, Ny_old-1)
        for nx in range(0, Nx):
            nx_sh = 2*nx
            nx_sh_pl = min(nx_sh + 1, Nx_old - 1)

            rsum = 0.0
            rsum += A_copy[ny_sh, nx_sh] * w[ny_sh, nx_sh]
            rsum += A_copy[ny_sh_pl, nx_sh] * w[ny_sh_pl, nx_sh]
            rsum += A_copy[ny_sh, nx_sh_pl] * w[ny_sh, nx_sh_pl]
            rsum += A_copy[ny_sh_pl, nx_sh_pl] * w[ny_sh_pl, nx_sh_pl]

            wsum = 0.0
            wsum += w[ny_sh, nx_sh]
            wsum += w[ny_sh_pl, nx_sh]
            wsum += w[ny_sh, nx_sh_pl]
            wsum += w[ny_sh_pl, nx_sh_pl]

            if wsum > 0.0:
                A_downres[ny, nx] = rsum/wsum
            else:
                A_downres[ny, nx] = A[ny_sh, nx_sh]

    return A_downres


def downres_maps_py(mean, sdev, nobs, masked, lat, lon):

    mean_smoothed = smooth_map_array(mean)
    sdev_smoothed = smooth_map_array(sdev)
    nobs_smoothed = smooth_map_array(nobs)
    if masked is not None:
        masked_smoothed = smooth_map_array(masked)
    else:
        masked_smoothed = None
    
    lat_smoothed = np.linspace(lat[0], lat[-1], (lat.shape[0]-1)//2+1)
    lon_smoothed = lon[0::2]
    
    return mean_smoothed, sdev_smoothed, nobs_smoothed, masked_smoothed, lat_smoothed, lon_smoothed


def get_cmap_from_xml(xml_name):

    tree = ET.parse(xml_name)
    root = tree.getroot()
    
    N = len(root)
    cmap_array = np.zeros((N, 4), dtype=np.float)
    
    for n in range(0, N):
        r = float(root[n].attrib['r'])/255.0
        g = float(root[n].attrib['g'])/255.0
        b = float(root[n].attrib['b'])/255.0
        cmap_array[n, :] = np.array([r, g, b, 1.0])

    cmap = ListedColormap(cmap_array)
    
    return cmap


def save_to_text(path, mat, delimiter=',', header='', comments='', fmt='%.18e'):

    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.error('Unable to remove file: "{}". Reason: {}'.format(path, str(e)))
            raise

    try:
        np.savetxt(path, mat, delimiter=delimiter, header=header, comments=comments, fmt=fmt)
    except Exception as e:
        logger.error('Unable to write file "{}". Reason: {}'.format(path, str(e)))
        raise


def compress_and_savefig(fig, path, dpi, compress=True):

    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.error('Unable to remove file: "{}". Reason: {}'.format(path, str(e)))
            raise

    if compress:
        shm_dir = '/dev/shm'
        if os.path.isdir(shm_dir):
            logger.debug('Saving fig to /dev/shm')
            with tempfile.TemporaryDirectory(suffix='', prefix='tmp_pysquam', dir=shm_dir) as tmp_dir:
                logger.debug('tmp_path directory created')
                tmp_path = os.path.join(tmp_dir, 'tmp_{}'.format(os.path.basename(path)))
                logger.debug('Saving figure to tmp_path')
                fig.savefig(tmp_path, format='png', dpi=dpi)
                logger.debug('Opening uncompressed image from /dev/shm')
                im = Image.open(tmp_path)
                logger.debug('Converting image')
                im2 = im.convert('RGB')
                logger.debug('Saving image')
                im2.save(path, format='PNG', optimize=True)
                logger.debug('Done saving image')
        else:
            with io.BytesIO() as ram:
                logger.debug('Saving fig to ram')
                fig.savefig(ram, format='png', dpi=dpi)
                logger.debug('Seeking first position')
                ram.seek(0)
                logger.debug('Opening uncompressed image from ram')
                im = Image.open(ram)
                logger.debug('Converting image')
                im2 = im.convert('RGB')
                logger.debug('Saving image')
                im2.save(path, format='PNG', optimize=True)
                logger.debug('Done saving image')

        # with io.BytesIO() as ram:
        #     logger.debug('Saving fig to ram')
        #     fig.savefig(ram, format='png', dpi=dpi)
        #     logger.debug('Seeking first position')
        #     ram.seek(0)
        #     logger.debug('Opening compressed image from ram')
        #     im = Image.open(ram)
        #     logger.debug('Converting image')
        #     im2 = im.convert('RGB')
        #     logger.debug('Saving image')
        #     im2.save(path, format='PNG', optimize=True)
        #     logger.debug('Done saving image')

    else:
        fig.savefig(path, dpi=dpi, format='png')

    return


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def get_sst_cmap():

    violet = (93/255.0, 0/255.0, 96/255.0)
    blue = (29/255.0, 0/255.0, 255/255.0)
    cyan = (0/255.0, 255/255.0, 233/255.0)
    # green = (0/255.0, 255/255.0, 46/255.0)
    grey = (191/255.0, 191/255.0, 191/255.0)

    # yellow = (255/255.0, 248/255.0, 45/255.0)
    yellow = (246/255.0, 255/255.0, 0/255.0)

    orange = (255/255.0, 182/255.0, 0/255.0)
    red = (255/255.0, 0/255.0, 0/255.0)
    dark_red = (128.0/255.0, 0.0, 0.0)

    rvb = make_colormap([violet, blue, 0.15,
                         blue, cyan, 0.475,
                         grey, grey, 0.525,
                         yellow, orange, 0.60,
                         orange, red, 0.90,
                         red, dark_red, 1.00,
                         dark_red])

    return rvb


def get_sst_cmap_smooth():

    violet = (93/255.0, 0/255.0, 96/255.0)
    blue = (29/255.0, 0/255.0, 255/255.0)
    cyan = (0/255.0, 255/255.0, 233/255.0)
    # green = (0/255.0, 255/255.0, 46/255.0)
    grey = (191/255.0, 191/255.0, 191/255.0)
    # grey   = green

    # yellow = (255/255.0, 248/255.0, 45/255.0)
    yellow = (246/255.0, 255/255.0, 0/255.0)

    orange = (255/255.0, 182/255.0, 0/255.0)
    red = (255/255.0, 0/255.0, 0/255.0)
    dark_red = (128.0/255.0, 0.0, 0.0)

    rvb = make_colormap([violet, blue, 0.150,
                         blue, cyan, 0.450,
                         cyan, grey, 0.485,
                         grey, grey, 0.515,
                         grey, yellow, 0.550,
                         yellow, orange, 0.600,
                         orange, red, 0.900,
                         red, dark_red, 1.000,
                         dark_red])

    return rvb


def get_sst_sdev_cmap():

    violet = (93/255.0, 0/255.0, 96/255.0)
    blue = (29/255.0, 0/255.0, 255/255.0)
    cyan = (0/255.0, 255/255.0, 233/255.0)
    # grey = (191/255.0, 191/255.0, 191/255.0)
    green = (0.0/255.0, 255.0/255.0, 20.0/255.0)
    yellow = (246/255.0, 255/255.0, 0/255.0)
    orange = (255/255.0, 182/255.0, 0/255.0)
    red = (255/255.0, 0/255.0, 0/255.0)
    dark_red = (128.0/255.0, 0.0, 0.0)

    rvb = make_colormap([violet, blue, 0.15,
                         blue, cyan, 0.40,
                         cyan, green, 0.50,
                         green, yellow, 0.60,
                         yellow, orange, 0.75,
                         orange, red, 0.90,
                         red, dark_red, 1.00,
                         dark_red])

    return rvb


def get_sst_nobs_cmap():

    violet = (93/255.0, 0/255.0, 96/255.0)
    blue = (29/255.0, 0/255.0, 255/255.0)
    cyan = (0/255.0, 255/255.0, 233/255.0)
    grey = (191/255.0, 191/255.0, 191/255.0)
    green = (0.0/255.0, 255.0/255.0, 20.0/255.0)
    yellow = (246/255.0, 255/255.0, 0/255.0)
    orange = (255/255.0, 182/255.0, 0/255.0)
    red = (255/255.0, 0/255.0, 0/255.0)
    dark_red = (128.0/255.0, 0.0, 0.0)

    rvb = make_colormap([grey, grey, 0.005,
                         violet, blue, 0.15,
                         blue, cyan, 0.40,
                         cyan, green, 0.50,
                         green, yellow, 0.60,
                         yellow, orange, 0.75,
                         orange, red, 0.90,
                         red, dark_red, 1.00,
                         dark_red])

    return rvb


def make_sure_path_exists(path):

    os.umask(0o002)
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def chunk_list(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


# copies file in_path to out_path by creating a temporary directory in the directory
# containing out_path and then moving the file from the temporary path to out_path
def safe_copy(in_path, out_path):

    os.umask(0o002)

    out_dir = os.path.dirname(out_path)
    out_name = os.path.basename(out_path)

    if not os.path.isfile(in_path):
        raise RuntimeError('ERROR: in_path = {} is not a file'.format(in_path))

    if not os.path.isdir(out_dir):
        raise RuntimeError('ERROR: out_dir = {} is not a directory'.format(out_dir))

    if not os.access(out_dir, os.W_OK):
        raise RuntimeError('ERROR: out_dir = {} is not writable'.format(out_dir))

    with tempfile.TemporaryDirectory(dir=out_dir, prefix='tmp_') as temp_dir:
        # change group permission to make it easier to delte directory in case of a crash
        perm = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_ISGID | stat.S_IXGRP
        os.chmod(temp_dir, perm)
        temp_path = os.path.join(temp_dir, out_name)
        shutil.copy2(in_path, temp_path)
        os.chmod(temp_path, 0o664)
        os.rename(temp_path, out_path)

    return


def get_md5(path):

    if not os.path.isfile(path):
        raise RuntimeError('ERROR: path = {} is not a file'.format(path))

    with open(path, 'rb') as f:
        file_as_bytes = f.read()

    return hashlib.md5(file_as_bytes).hexdigest()


def get_dt_list(start_dt, end_dt, aggr):

    if aggr == 'daily':
        day_count = (end_dt-start_dt).days + 1
        dt_list = [start_dt + timedelta(days=n) for n in range(0, day_count)]
    elif aggr == 'monthly':
        trunc_start = datetime(start_dt.year, start_dt.month, 1)
        trunc_end = datetime(end_dt.year, end_dt.month, 1)
        dt_diff = relativedelta(trunc_end, trunc_start)
        month_count = dt_diff.years*12 + dt_diff.months + 1
        dt_list = [trunc_start + relativedelta(months=n) for n in range(0, month_count)]
    elif aggr == 'yearly':
        trunc_start = datetime(start_dt.year, 1, 1)
        trunc_end = datetime(end_dt.year, 1, 1)
        dt_diff = relativedelta(trunc_end, trunc_start)
        year_count = dt_diff.years+1
        dt_list = [trunc_start + relativedelta(years=n) for n in range(0, year_count)]
    else:
        raise ValueError('Invalid aggregation : "{}"'.format(aggr))

    return dt_list


def test_MISA_file_lite(path, atts_to_check=()):

    try:
        ncf = netCDF4.Dataset(path)
        for attname in atts_to_check:
            ncf.getncattr(attname)
        return True
    except Exception as err:
        logger.error(err)
        return False


def test_h5_file(path, max_tries=1, time_per_try=5):

    if max_tries < 0:
        raise RuntimeError('Error in test_h5_file. max_tries must be bigger than zero')

    if max_tries > 10:
        raise RuntimeError('Error in test_h5_file. max_tries may not be bigger than 10')

    if time_per_try < 0:
        raise RuntimeError('Error in test_h5_file. time_per_try must be bigger than zero')

    if time_per_try > 60:
        raise RuntimeError('Error in test_h5_file. time_per_try may not be bigger than 60')

    # h5check fails on the SST MISA files for some reason. I believe it is because they use netCDF4 1.10 features
    # while h5check only supports 1.8
    if re.search(r'MISA.*SST_\w\w\w.nc', os.path.basename(path)):
        return test_MISA_file_lite(path, atts_to_check=['SATELLITE', 'SENSOR', 'date_created'])

    if db_setup.h5check_path == '':
        logger.warning('''h5check path = {:s} is invalid. Unable to check file for corruptness.
                           Will only check file header by opening and closing file''')
        return test_MISA_file_lite(path)

    for try_num in range(0, max_tries):
        try:
            subprocess.check_output([db_setup.h5check_path, path])
            return True
        except subprocess.CalledProcessError as h5check_exc:
            logger.error('Detected invalid h5 file for try_num = {}. h5check return code = {:d}. Error message: {:s}'.format(
                          try_num+1, h5check_exc.returncode, h5check_exc.output.decode()))
            if try_num < max_tries - 1:
                time.sleep(time_per_try)
        
    return False
