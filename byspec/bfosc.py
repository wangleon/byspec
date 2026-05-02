import os
import re
import datetime
import dateutil.parser

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import scipy.signal as sg
from scipy.ndimage.filters import median_filter
from astropy.table import Table, Row
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from .utils import get_file
from .imageproc import combine_images, savgol_filter_2d
from .onedarray import (iterative_savgol_filter, get_simple_ccf,
                        gaussian, gengaussian,
                        consecutive, find_shift_ccf, get_clip_mean,
                        get_local_minima)
from .visual import plot_image_with_hist

from .common import (FOSCReducer, find_longslit_wavelength,
                     find_distortion, find_echelle_wavelength)

def print_wrapper(string, item):
    """A wrapper for obslog printing

    """
    datatype = item['datatype']
    objname  = item['object']

    if datatype=='BIAS':
        # bias, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif datatype in ['SLITTARGET', 'SPECLTARGET']:
        # science targets, use nighlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif datatype=='SPECLLAMP':
        # lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'

    else:
        return string

def make_obslog(path, display=True):
    obstable = Table(dtype=[
            ('frameid',     int),
            ('fileid',      int),
            ('datatype',    str),
            ('object',      str),
            ('exptime',     float),
            ('dateobs',     str),
            ('RAJ2000',     str),
            ('DEJ2000',     str),
            ('mode',        str),
            ('config',      str),
            ('slit',        str),
            ('filter',      str),
            ('binning',     str),
            ('gain',        float),
            ('rdnoise',     float),
            ('q99',         int),
            ('observer',    str),
        ], masked=True)


    fmt_str = ('  - {:7s} {:12s} {:>12s} {:>16s} {:>7} {:23s}'
               '{:8s} {:6s} {:8s} {:8s} {:7s} {:7s} {:7s} {:5s} {:15s}')
    head_str = fmt_str.format('frameid', 'fileid', 'datatype', 'object',
                    'exptime', 'dateobs', 'mode', 'config',
                    'slit', 'filter', 'binning', 'gain', 'rdnoise', 'q99',
                    'observer')
    if display:
        print(head_str)

    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.fit'):
            continue
        mobj = re.match('(\d{12})_([A-Z]+)_(\S*)_(\S*)_(\S*)_(\S*)\.fit', fname)
        if not mobj:
            continue
        fileid = int(mobj.group(1))
        frameid = int(str(fileid)[8:])
        datatype = mobj.group(2)
        objname2 = mobj.group(3)
        key1     = mobj.group(4)
        key2     = mobj.group(5)
        key3     = mobj.group(6)

        filename = os.path.join(path, fname)
        data, header = fits.getdata(filename, header=True)
        objname    = header['OBJECT']
        exptime    = header['EXPTIME']
        dateobs    = header['DATE-OBS']
        ra         = header['RA']
        dec        = header['DEC']
        _filter    = header['FILTER']
        xbinning   = header['XBINNING']
        ybinning   = header['YBINNING']
        gain       = header['GAIN']
        rdnoise    = header['RDNOISE']
        observer   = header['OBSERVER']
        q99        = int(np.percentile(data, 99))

        binning = '{}x{}'.format(xbinning, ybinning)

        if re.match('G\d+', key3):
            mode       = 'longslit'
            config     = key3
            filtername = key2
        elif re.match('G\d+', key2) and re.match('E\d+', key3):
            mode       = 'echelle'
            config     = '{}+{}'.format(key3, key2)
            filtername = ''
        else:
            mode       = ''
            config     = ''
            filtername = key2

        if filtername == 'Free':
            filtername = ''

        # find slit width
        mobj = re.match('slit(\d+)s?', key1)
        if mobj:
            slit = str(int(mobj.group(1))/10) + '"'
        else:
            slit = ''

        if datatype in ['BIAS', 'SPECLFLAT', 'SPECLLAMP',
                                'SPECSFLAT', 'SPECSLAMP',]:
            ra, dec = '', ''

        obstable.add_row((frameid, fileid, datatype, objname, exptime, dateobs,
                        ra, dec, mode, config, slit, filtername,
                        binning, float(gain), float(rdnoise), q99, observer))

        if display:
            logitem = obstable[-1]

            string = fmt_str.format(
                    '[{:d}]'.format(frameid), str(fileid),
                    '{:12s}'.format(datatype),
                    objname, exptime, dateobs[0:23], mode, config, 
                    slit, filtername, binning, gain, rdnoise,
                    '{:5d}'.format(q99), observer,
                    )
            print(print_wrapper(string, logitem))

    obstable.sort('fileid')

    return obstable

def get_mosaic_fileid(obsdate, dateobs):
    date = dateutil.parser.parse(obsdate)
    t0 = datetime.datetime.combine(date, datetime.time(0, 0, 0))
    t1 = dateutil.parser.parse(dateobs)
    delta_t = t1 - t0
    delta_minutes = int(delta_t.total_seconds()/60)
    newid = '{:4d}{:02d}{:02d}c{:4d}'.format(date.year, date.month, date.day,
                                           delta_minutes)
    return newid

def select_calib_from_database(index_file, lamp, mode, config, dateobs):
    calibtable = Table.read(index_file, format='ascii.fixed_width_two_line')

    # select the corresponding arclamp, mode, and config
    mask = (calibtable['object'] == lamp) * \
           (calibtable['mode']   == mode) * \
           (calibtable['config'] == config)

    calibtable = calibtable[mask]

    input_date = dateutil.parser.parse(dateobs)

    # select the closest ThAr
    timediff = [(dateutil.parser.parse(t)-input_date).total_seconds()
                for t in calibtable['dateobs']]
    irow = np.abs(timediff).argmin()
    row = calibtable[irow]
    fileid = row['fileid']  # selected fileid
    md5    = row['md5']

    message = 'Select {} from database index as {} reference'.format(fileid, lamp)
    print(message)

    filepath = os.path.join('calib/bfosc/', 'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    linelist = Table(hdu_lst[2].data)
    hdu_lst.close()

    return spec, linelist

def find_echelle_apertures(data, align_deg, scan_step):
    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    logdata = np.log10(np.maximum(data, 1))

    x0 = nx//2
    x_lst = {-1:[], 1:[]}
    x1 = x0
    direction = -1
    icol = 0

    csec_i1 = -ny//2
    csec_i2 = ny + ny//2
    csec_lst  = np.zeros(csec_i2 - csec_i1)
    csec_nlst = np.zeros(csec_i2 - csec_i1, dtype=np.int32)

    param_lst = {-1:[], 1:[]}
    nodes_lst = {}

    def forward(x, p):
        deg = len(p) - 1 # determine the polynomial degree
        res = p[0]
        for i in range(deg):
            res = res*x + p[i+1]
        return res
    def forward_der(x, p):
        deg = len(p)-1  # determine the polynomial degree
        p_der = [(deg-i)*p[i] for i in range(deg)]
        return forward(x, p_der)
    def backward(y, p):
        x = y
        for ite in range(20):
            dy    = forward(x, p) - y
            y_der = forward_der(x, p)
            dx = dy/y_der
            x = x - dx
            if (abs(dx) < 1e-7).all():
                break
        return x
    def fitfunc(p, interfunc, n):
        return interfunc(forward(np.arange(n), p[0:-1])) + p[-1]
    def resfunc(p, interfunc, flux0, mask=None):
        res_lst = flux0 - fitfunc(p, interfunc, flux0.size)
        if mask is None:
            mask = np.ones_like(flux0, dtype=bool)
        return res_lst[mask]
    def find_shift(flux0, flux1, deg):
        #p0 = [1.0, 0.0, 0.0]
        #p0 = [0.0, 1.0, 0.0, 0.0]
        #p0 = [0.0, 0.0, 1.0, 0.0, 0.0]

        p0 = [0.0 for i in range(deg+1)]
        p0[-3] = 1.0

        interfunc = intp.InterpolatedUnivariateSpline(
                    np.arange(flux1.size), flux1, k=3, ext=3)
        mask = np.ones_like(flux0, dtype=bool)
        clipping = 5.
        for i in range(10):
            p, _ = opt.leastsq(resfunc, p0, args=(interfunc, flux0, mask))
            res_lst = resfunc(p, interfunc, flux0)
            std  = res_lst.std()
            mask1 = res_lst <  clipping*std
            mask2 = res_lst > -clipping*std
            new_mask = mask1*mask2
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask
        return p, mask

    while(True):
        nodes_lst[x1] = []

        flux1 = logdata[:,x1]
        linflux1 = np.median(data[:,x1-2:x1+3], axis=1)

        if icol == 0:
            flux1_center = flux1

            # the middle column
            i1 = 0 - csec_i1
            i2 = ny - csec_i1
            # stack the linear flux to the stacked cross-section
            csec_lst[i1:i2] += linflux1
            csec_nlst[i1:i2] += 1

        else:
            param, _ = find_shift(flux0, flux1, deg=align_deg)
            param_lst[direction].append(param[0:-1])
            ysta, yend = 0., ny-1.
            for param in param_lst[direction][::-1]:
                ysta = backward(ysta, param)
                yend = backward(yend, param)
            # interpolate the new crosssection
            ynew = np.linspace(ysta, yend, ny)
            interfunc = intp.InterpolatedUnivariateSpline(ynew, linflux1, k=3)
            #
            ysta_int = int(round(ysta))
            yend_int = int(round(yend))
            fnew = interfunc(np.arange(ysta_int, yend_int+1))
            i1 = ysta_int - csec_i1
            i2 = yend_int + 1 - csec_i1
            csec_lst[i1:i2] += fnew
            csec_nlst[i1:i2] += 1

        x1 += direction*scan_step
        if x1 <= 10:
            direction = +1
            x1 = x0 + direction*scan_step
            x_lst[direction].append(x1)
            flux0 = flux1_center
            icol += 1
            continue
        elif x1 >= nx - 10:
            # scan ends
            break
        else:
            x_lst[direction].append(x1)
            flux0 = flux1
            icol += 1
            continue
    #
    i_nonzero = np.nonzero(csec_nlst)[0]
    istart, iend = i_nonzero[0], i_nonzero[-1]
    csec_ylst = np.arange(csec_lst.size) + csec_i1

    x = csec_ylst[istart:iend]
    y = csec_lst[istart:iend]
    x = x[100:-30]
    y = y[100:-30]
    n = y.size

    #########################
    # cross-section stacking
    #fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    #for x0 in np.arange(nx//2, nx, 100):
    #    ax1.plot(data[:,x0], lw=0.5)
    #ax1.set_yscale('log')
    #ax2.plot(x, y, lw=0.5)
    #ax2.set_yscale('log')

    ############################

    winmask = np.zeros_like(y, dtype=bool)
    xnodes = [100, 1100]
    wnodes = [30, 240]
    snodes = [20, 220]
    c1 = np.polyfit(xnodes, wnodes, deg=len(xnodes)-1)
    c2 = np.polyfit(xnodes, snodes, deg=len(xnodes)-1) 
    get_winlen = lambda x: np.polyval(c1, x)
    get_gaplen = lambda x: np.polyval(c2, x)
    for i1 in np.arange(0, n):
        winlen = get_winlen(i1)
        gaplen = get_gaplen(i1)
        gaplen = max(gaplen, 5)
        percent = gaplen/winlen*100
        i2 = i1 + int(winlen)
        if i2 >= n-1:
            break
        v = np.percentile(y[i1:i2], percent)
        pick = y[i1:i2]>v
        if (~pick).sum()==0:
            pick[pick.argmin()] = False
        idx = np.nonzero(pick)[0]
        winmask[idx+i1] = True

    bkgmask = ~winmask
    maxiter = 10
    for ite in range(maxiter):
        c = np.polyfit(x[bkgmask], np.log(y[bkgmask]), deg=15)
        newy = np.polyval(c, x)
        resy = np.log(y) - newy
        std = resy[bkgmask].std()
        newbkgmask = resy < 2*std
        if newbkgmask.sum() == bkgmask.sum():
            break
        bkgmask = newbkgmask

    aper_mask = y > np.exp(newy + 3*std)
    aper_idx = np.nonzero(aper_mask)[0]

    gap_mask = ~aper_mask
    gap_idx = np.nonzero(gap_mask)[0]

    max_order_width = 120
    min_order_width = 3

    order_index_lst = []
    for group in np.split(aper_idx, np.where(np.diff(aper_idx)>3)[0]+1):
        i1 = group[0]
        i2 = group[-1]
        if i2-i1 > max_order_width or i2-i1<min_order_width:
            continue
        chunk = y[i1:i2]
        m = chunk > (chunk.max()*0.3 + chunk.min()*0.7)
        i11 = np.nonzero(m)[0][0] + i1
        i22 = np.nonzero(m)[0][-1] + i1
        order_index_lst.append((i11, i22))

    norder = len(order_index_lst)
    order_lst = np.arange(norder)
    order_cen_lst = np.array([(i1+i2)/2 for i1, i2 in order_index_lst])
    goodmask = np.zeros(norder, dtype=bool)
    goodmask[0:10] = True

    ### find good orders
    #fig3 = plt.figure()
    #ax31 = fig3.add_subplot(211)
    #ax32 = fig3.add_subplot(212)
    #ax31.plot(order_lst[goodmask], order_cen_lst[goodmask], 'o', c='C0')
    #ax31.plot(order_lst[~goodmask], order_cen_lst[~goodmask], 'o', c='none', mec='C0')
    for i in range((~goodmask).sum()):
        fintp = intp.InterpolatedUnivariateSpline(
                np.arange(goodmask.sum()), order_cen_lst[goodmask], k=3)
        newcen = fintp(goodmask.sum())
        #ax31.axhline(newcen, color='k', ls='--')
        min_order = None
        min_dist = 9999
        for iorder, cen in enumerate(order_cen_lst):
            if abs(cen - newcen) < min_dist:
                min_dist = abs(cen - newcen)
                min_order = iorder
        goodmask[min_order] = True
        if newcen > order_cen_lst[-1]:
            break
    #ax31.plot(order_lst[goodmask], order_cen_lst[goodmask], 'o', c='C0', ms=2)

    # plot co-added cross-order profiles
    fig2 = plt.figure(dpi=200, figsize=(6, 3.7))
    ax2 = fig2.add_axes([0.12, 0.13, 0.85, 0.83])
    ax2.plot(x, y, lw=0.8)
    #ax2.plot(x[aper_idx], y[aper_idx], 'o', ms=1)
    ax2.plot(x, np.exp(newy), '-')
    ax2.set_yscale('log')
    _y1, _y2 = ax2.get_ylim()
    for iorder, (i1, i2) in enumerate(order_index_lst):
        if goodmask[iorder]:
            color = 'C0'
        else:
            color = 'C1'
        ax2.fill_betweenx([_y1, _y2], x[i1], x[i2], color=color, alpha=0.2, lw=0)
    ax2.plot(x, np.exp(newy+3*std), '--')
    ax2.set_ylim(_y1,_y2)
    ax2.set_xlabel('Y (pixel)')
    ax2.set_ylabel('Flux')

    # plot all detected orders
    fig = plt.figure(dpi=200, figsize=(6, 3.7))
    ax = fig.add_axes([0.12, 0.13, 0.85, 0.83])
    ax.imshow(np.log10(data), cmap='gray')

    coeff_lst = []
    for iorder, (i1, i2) in enumerate(order_index_lst):
        cen = (x[i1] + x[i2])/2
        xnode_lst = [x0]
        ynode_lst = [cen]
        for direction in [-1, 1]:
            cen1 = cen
            for icol, param in enumerate(param_lst[direction]):
                cen1 = forward(cen1, param)
                xcol = x_lst[direction][icol]
                xnode_lst.append(xcol)
                ynode_lst.append(cen1)
        xnode_lst = np.array(xnode_lst)
        ynode_lst = np.array(ynode_lst)
        # resort
        args = xnode_lst.argsort()
        xnode_lst = xnode_lst[args]
        ynode_lst = ynode_lst[args]
        # fit polynomial with 3rd degree
        c = np.polyfit(xnode_lst, ynode_lst, deg=3)
        coeff_lst.append(c)

        if goodmask[iorder]:
            color = 'C0'
            ls = '-'
        else:
            color = 'C1'
            ls = '--'
        # plot node
        #ax.plot(xnode_lst, ynode_lst, ls=ls, c=color, lw=0.5)
        # plot positions
        ax.plot(allx, np.polyval(c, allx), ls=ls, c=color, lw=0.5)

        #fig0 = plt.figure()
        #ax01 = fig0.add_subplot(211)
        #ax02 = fig0.add_subplot(212)
        #ax01.plot(xnode_lst, ynode_lst, 'o', ms=3)
        #ax01.plot(allx, np.polyval(c, allx), '-')
        #ax02.plot(xnode_lst, ynode_lst - np.polyval(c, xnode_lst), 'o', ms=3)
        #fig0.savefig('order_{}.png'.format(iorder))
        #plt.close(fig0)
        
    ax.set_xlim(0, nx-1)
    ax.set_ylim(0, ny-1)
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')

    return coeff_lst, goodmask, fig, fig2


def get_longslit_sensmap(data):
    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)
    sensmap = np.ones_like(data, dtype=np.float64)

    #for y in np.arange(ny):
    #    #fluxt1d = self.flat_data[y, 20:int(nx) - 20]
    #    flux1d = data[y, :]
    #    flux1d_sm, _, mask, std = iterative_savgol_filter(
    #                                flux1d,
    #                                winlen=51, order=3, mode='interp',
    #                                upper_clip=6, lower_clip=6, maxiter=10)
    #    sensmap[y, :] = flux1d/flux1d_sm

    newdata = data[3:, :]
    smoothed = savgol_filter_2d(newdata, window_length=151, order=3)
    sensmap[3:, :] = newdata/smoothed

    return sensmap


class BFOSC(FOSCReducer):

    # Geolocation of Xinglong 2.16m telescope
    longitude = 117.57454167
    latitude  = 40.395933
    altitude  = 950.0
    
    def __init__(self, **kwargs):
        super(BFOSC, self).__init__(**kwargs)

    def make_obslog(self, filename=None):
        """Scan the raw data path and generate an observing log.
        """
        logtable = make_obslog(self.rawdata_path, display=True)

        # find obsdate
        self.obsdate = logtable[0]['dateobs'][0:10]

        if filename is None:
            filename = 'BFOSC.{}.txt'.format(self.obsdate)
        filename = os.path.join(self.reduction_path, filename)

        logtable.write(filename, format='ascii.fixed_width_two_line',
                        overwrite=True)
        self.logtable = logtable

    def fileid_to_filename(self, fileid):
        for fname in os.listdir(self.rawdata_path):
            if fname.startswith(str(fileid)):
                return os.path.join(self.rawdata_path, fname)
        raise ValueError

    def get_bias(self):

        self.get_all_ccdconf()
        self.bias = {}

        for ccdconf in self.ccdconf_lst:
            ccdconf_string = self.get_ccdconf_string(ccdconf)
            bias_file = 'bias_{}.fits'.format(ccdconf_string)
            bias_filename = os.path.join(self.reduction_path, bias_file)
            if os.path.exists(bias_filename):
                hdulst = fits.open(bias_filename)
                bias_data = hdulst[1].data
                bias_img  = hdulst[2].data
                hdulst.close()
            else:
                print('Combine bias')
                selectfunc = lambda item: item['datatype']=='BIAS' and \
                                self.get_ccdconf(item)==ccdconf

                item_lst = list(filter(selectfunc, self.logtable))
                if len(item_lst)==0:
                    continue
                
                data_lst = []
                for logitem in item_lst:
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    data_lst.append(data)
                data_lst = np.array(data_lst)
                
                bias_data = combine_images(data_lst, mode='mean',
                                upper_clip=5, maxiter=10, maskmode='max')

                # get the mean cross-section of coadded bias image
                section = bias_data.mean(axis=0)
                # get the smoothed cross-section
                smsection = sg.savgol_filter(section,
                            window_length=151, polyorder=3)
                # broadcast the smoothed section to the entire image
                ny, nx = bias_data.shape
                #bias_img = np.repeat(smsection, nx).reshape(-1, nx)
                bias_img = np.tile(smsection, ny).reshape(ny, -1)

                # save to fits file
                hdulst = fits.HDUList([fits.PrimaryHDU(),
                                       fits.ImageHDU(data=bias_data),
                                       fits.ImageHDU(data=bias_img),
                                       ])
                hdulst.writeto(bias_filename, overwrite=True)

            self.bias[ccdconf] = bias_img

    def combine_flat(self):
        """Combine flat images
        """

        self.get_all_conf()
        self.flat = {}

        for conf in self.conf_lst:
            conf_string = self.get_conf_string(conf)
            flat_file = 'flat_{}.fits'.format(conf_string)
            flat_filename = os.path.join(self.reduction_path, flat_file)
            if os.path.exists(flat_filename):
                flat_data = fits.getdata(flat_filename)
            else:
                selectfunc = lambda item: item['object']=='FLAT' and \
                                self.get_conf(item)==conf

                item_lst = list(filter(selectfunc, self.logtable))

                if len(item_lst)==0:
                    continue

                print('Combine Flat for {}'.format(conf_string))

                data_lst = []
                for logitem in item_lst:
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    ccdconf = self.get_ccdconf(logitem)
                    data = data - self.bias[ccdconf]
                    data_lst.append(data)
                data_lst = np.array(data_lst)

                flat_data = combine_images(data_lst, mode='mean',
                                    upper_clip=5, maxiter=10, maskmode='max')
                fits.writeto(flat_filename, flat_data, overwrite=True)
            self.flat[conf] = flat_data

    def plot_bias(self, show=True):
        figfilename = os.path.join(self.figpath, 'bias.png')
        title = 'Bias ({})'.format(os.path.basename(self.bias_file))
        plot_image_with_hist(self.bias_data,
                        show        = show,
                        figfilename = figfilename,
                        title       = title,
                        )

    def plot_flat(self, show=True):
        figfilename = os.path.join(self.figpath, 'flat.png')
        title = 'Flat ({})'.format(os.path.basename(self.flat_file))
        plot_image_with_hist(self.flat_data,
                        show        = show,
                        figfilename = figfilename,
                        title       = title,
                        )

    def plot_sens(self, show=True):
        figfilename = os.path.join(self.figpath, 'sens.png')
        title = 'Sensitivity ({})'.format(os.path.basename(self.sens_file))
        plot_image_with_hist(self.sens_data,
                        show        = show,
                        figfilename = figfilename,
                        title       = title,
                        )

    def get_conf(self, logitem):
        return (logitem['mode'], logitem['config'],
                logitem['binning'], logitem['gain'], logitem['rdnoise'])
    
    def get_conf_string(self, conf):
        mode, config, binning, gain, rdspeed = conf
        return '{}_{}_{}_gian{:3.1f}.ron{:.1f}'.format(mode, config, binning,
                                                       gain, rdspeed)

    def get_ccdconf(self, logitem):
        return (logitem['binning'], logitem['gain'], logitem['rdnoise'])

    def get_ccdconf_string(self, ccdconf):
        binning, gain, rdnoise = ccdconf
        return '{}_gain{:.1f}_ron{}'.format(binning, gain, rdnoise)

    def find_echelle_orders(self):

        self.echelle_coeff_lst = {}
        self.echelle_goodmask = {}
        for conf, flat_data in self.flat.items():
            if conf[0]!='echelle':
                continue

            # trim the image
            data = flat_data[800:, :]

            # find apertures
            result = find_echelle_apertures(data,
                                scan_step=50, align_deg=3)
            coeff_lst = result[0]
            goodmask  = result[1]
            fig_orders = result[2]
            fig_section = result[3]

            conf_string = self.get_conf_string(conf)

            figfilename = 'echelle_orders_{}.png'.format(conf_string)
            fig_orders.savefig(figfilename)
            plt.close(fig_orders)

            figfilename = 'echelle_sections_{}.png'.format(conf_string)
            fig_section.savefig(figfilename)
            plt.close(fig_section)

            self.echelle_coeff_lst[conf] = coeff_lst
            self.echelle_goodmask[conf] = goodmask

    def get_echelle_sens(self):

        self.echelle_sens = {}
        for conf, flat_data in self.flat.items():
            if conf[0] != 'echelle':
                continue

            conf_string = self.get_conf_string(conf)

            echelle_coeff_lst = self.echelle_coeff_lst[conf]
            echelle_goodmask  = self.echelle_goodmask[conf]

            # trim the image
            data = flat_data[800:, :]
            ny, nx = data.shape
            allx = np.arange(nx)
            ally = np.arange(ny)

            # fix flat data
            #m = np.ones(ny, dtype=bool)
            #m[1537-800:1538+1-800] = False
            #fixdata = data.copy()
            #fig = plt.figure()
            #ax = fig.gca()
            #for x in np.arange(nx):
            #    flux = np.log(data[:, x])
            #    fixfunc = intp.InterpolatedUnivariateSpline(ally[m], flux[m], k=2)
            #    fixdata[:, x][~m] = np.exp(fixfunc(ally[~m]))
            #    if 320 < x < 330:
            #        ax.plot(ally, flux, '-')
            #        ax.plot(ally[m], flux[m], '--')
            #        ax.plot(ally[~m], fixdata[:,x][~m], 'o')
            #plt.show()
            #data = fixdata

            yy, xx = np.mgrid[:ny:, :nx:]
            sensmap = np.ones_like(data, dtype=np.float32)

            #fig = plt.figure()
            #ax = fig.gca()
            #ax.imshow(np.log10(data))
            #fig2 = plt.figure()
            #ax2 = fig2.gca()

            allmask = np.zeros_like(data, dtype=bool)
            for iorder, coeff in enumerate(echelle_coeff_lst):
                win = 5
            
                cen_lst = np.polyval(coeff, allx)
                #ax.plot(allx, cen_lst, c='r', ls='-', lw=0.5)
            
                mask = (yy<cen_lst+win)*(yy>cen_lst-win)
                allmask += (yy<cen_lst+win+5)*(yy>cen_lst-win-5)
            
                # skip ghost orders
                if not echelle_goodmask[iorder]:
                    continue
            
                spec = (data*mask).sum(axis=0)/10
                #ax2.plot(spec, lw=0.5)
               
                #fig0 =plt.figure(dpi=200)
                #ax0 = fig0.gca()
                #ax0.plot(spec, lw=0.5)
                ## smooth
            
                #spec_sm, _, m, std = iterative_savgol_filter(spec,
                #                    winlen=21, order=3,
                #                    upper_clip=5, lower_clip=5, maxiter=5)
               
            
                #core = sg.gaussian(15, std=3)
                #core /= core.sum()
                #spec_sm = np.convolve(spec, core, mode='same')
            
                m = np.ones_like(spec, dtype=bool)
                m2 = (cen_lst>0)*(cen_lst<ny)
                for i in range(3):
                    m = m*m2
                    coeff = np.polyfit(allx[m]/nx, np.log(spec[m]), deg=7)
                    res_lst = np.log(spec) - np.polyval(coeff, allx/nx)
                    std = res_lst[m].std()
                    newm = (res_lst > -3*std) * (res_lst < 3*std)
                    if newm.sum()==m.sum():
                        break
                    m = newm
                spec_sm = np.zeros(nx)
                spec_sm[m2] = np.exp(np.polyval(coeff, allx/nx)[m2])
            
                #ax0.plot(allx, spec_sm, lw=0.5)
                minidx, minvalue = get_local_minima(spec_sm, window=31)
                #ax0.plot(minidx, minvalue, 'o', ms=2, alpha=0.6)
                #fig0.savefig('smooth_{:02d}.png'.format(iorder))
                #plt.close(fig0)
            
                smooth_2d = np.tile(spec_sm, ny).reshape(ny, -1)
                sensmap[mask] = (data/smooth_2d)[mask]
                #sensmap[:,m2][mask[:,m2]] = (data[:, m2]/smooth_2d)
            
        
            ### tmp
            #hdulst = fits.HDUList([fits.PrimaryHDU(data=data),
            #                       fits.ImageHDU(data=np.int16(~allmask)),
            #                       ])
            #hdulst.writeto('flat_background.fits', overwrite=True)
            ### tmp
            
            #fig3 = plt.figure()
            #ax3 = fig3.gca()
            #ax3.imshow(np.log10(data*(~allmask)))
            
            #fig4 = plt.figure()
            #ax4 = fig4.gca()
            #ax4.imshow(sensmap)
            
            fits.writeto('sens_{}.fits'.format(conf_string),
                         sensmap, overwrite=True)
            
            #plt.show()
            
            self.echelle_sens[conf] = sensmap

    def get_longslit_sens(self):
        self.sensmap = {}
        for conf, flat_data in self.flat.items():
            mode = conf[0]
            conf_string = self.get_conf_string(conf)
            if mode == 'longslit':
                sensmap = get_longslit_sensmap(flat_data)
                sens_file = 'sens_{}.fits'.format(conf_string)
                sens_filename = os.path.join(self.reduction_path, sens_file)
                fits.writeto(sens_filename, sensmap, overwrite=True)
                self.sensmap[conf] = sensmap

    def extract_echelle(self, logitem):

        ccdconf = self.get_ccdconf(logitem)
        conf = self.get_conf(logitem)

        fileid = logitem['fileid']
        filename = self.fileid_to_filename(fileid)
        data, header = fits.getdata(filename, header=True)

        data = data - self.bias[ccdconf]
        # trim image
        data = data[800:, :]
        data = data / self.echelle_sens[conf]

        ny, nx = data.shape
        allx = np.arange(nx)
        ally = np.arange(ny)
        yy, xx = np.mgrid[:ny:, :nx:]

        echelle_coeff_lst = self.echelle_coeff_lst[conf]
        echelle_goodmask  = self.echelle_goodmask[conf]

        order_mask = np.zeros_like(data, dtype=bool)

        win = 8
        for coeff in echelle_coeff_lst:
            cen_lst = np.polyval(coeff, allx)
            mask = (yy<cen_lst+win) * (yy>cen_lst-win)
            order_mask += mask

        background = np.zeros_like(data, dtype=np.float32)
        fig = plt.figure()
        ax = fig.gca()
        for x in allx:
            section = data[:, x]
            m = order_mask[:, x]

            xdata = ally[~m]
            ydata = section[~m]

            mask = np.ones_like(xdata, dtype=bool)
            for i in range(5):
                c = np.polyfit(xdata[mask], ydata[mask], deg=3)
                yfit = np.polyval(c, xdata)
                res_lst = ydata - yfit
                std = res_lst[mask].std()
                newmask = (res_lst > -3*std) * (res_lst < 3*std)
                if newmask.sum()==mask.sum():
                    break
                mask = newmask
            background[:, x] = np.polyval(c, ally)

            if x%100==0:
                color = 'C{}'.format(x%99%10)
                ax.plot(ally[~m], section[~m]+x/5, '-',
                        lw=0.5, alpha=0.2, c=color)
                ax.plot(ally[~m][mask], section[~m][mask]+x/5, '-',
                        lw=0.5, alpha=1, c=color)
                ax.plot(ally, np.polyval(c, ally)+x/5, '--',
                        lw=0.5, c=color)

        background = median_filter(background, size=(9,5), mode='nearest')
        background = savgol_filter_2d(background, window_length=(21, 51),
                        order=3, mode='nearest')

        bkgdata = data*(~order_mask)
        

        # generate background figure
        fig1 = plt.figure(figsize=(15, 6), dpi=200)
        ax1  = fig1.add_axes([0.06, 0.13, 0.44, 0.87])
        ax2  = fig1.add_axes([0.53, 0.13, 0.44, 0.87])
        ax1c = fig1.add_axes([0.06, 0.1, 0.44, 0.02])
        ax2c = fig1.add_axes([0.53, 0.1, 0.44, 0.02])
        #ax1.imshow(bkgdata, vmin=np.percentile(bkgdata, 5),
        #                    vmax=np.percentile(bkgdata, 95))
        cax1 = ax1.imshow(data, cmap='gray', origin='lower',
                        vmin=np.percentile(data, 1),
                        vmax=np.percentile(data, 95))
        bbox = ax1.get_position()
        #ax1c.set_position([0.92, bbox.y0, 0.02, bbox.height])
        fig1.colorbar(cax1, cax=ax1c, orientation='horizontal')
        for coeff in echelle_coeff_lst:
            cen_lst = np.polyval(coeff, allx)
            upper = cen_lst + win
            m = (upper>0)*(upper<ny)
            ax1.plot(allx[m], upper[m], '-', lw=0.4, c='C0')
            lower = cen_lst - win
            m = (lower>0)*(lower<ny)
            ax1.plot(allx[m], lower[m], '-', lw=0.4, c='C0')
        ax1.set_xlim(0, nx-1)
        ax1.set_ylim(0, ny-1)
        cax2 = ax2.imshow(background, cmap='gray', origin='lower')
        bbox = ax2.get_position()
        #ax2c.set_position([0.92, bbox.y0, 0.02, bbox.height])
        cax2 = fig1.colorbar(cax2, cax=ax2c, orientation='horizontal')
        ax1.set_xlabel('X (pixel)')
        ax1.set_ylabel('Y (pixel)')
        ax2.set_xlabel('X (pixel)')
        #ax2.set_ylabel('Y (pixel)')
        ax2.set_yticklabels([])
        fig1.suptitle('Background for {} ({})'.format(fileid, logitem['object']))
        figname = 'background_{}.png'.format(fileid)
        fig1.savefig(figname)
        plt.close(fig1)

        # correct background
        data = data - background

    def extract_echelle_targets(self):
        func = lambda item: item['datatype']=='SPECSTARGET'
        logitem_lst = list(filter(func, self.logtable))
        for logitem in logitem_lst:
            self.extract_echelle(logitem)


    def extract_echelle_lamp(self):

        self.echelle_arclamp = {}

        nx = list(self.echelle_sens.values())[0].shape[1]
        # define dtype of 1-d spectra
        types = [
                ('aperture',   np.int16),
                ('order',      np.int16),
                ('points',     np.int16),
                ('wavelength', (np.float64, nx)),
                ('flux',       (np.float32, nx)),
                ('mask',       (np.int32,   nx)),
                ]
        names, formats = list(zip(*types))
        wlcalib_spectype = np.dtype({'names': names, 'formats': formats})

        for logitem in self.logtable:

            if logitem['mode'] != 'echelle':
                continue

            if logitem['datatype'] != 'SPECSLAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            fileid = logitem['fileid']
            filename = self.fileid_to_filename(fileid)
            data, header = fits.getdata(filename, header=True)

            data = data - self.bias[ccdconf]
            # trim image
            data = data[800:, :]
            data = data / self.echelle_sens[conf]

            ny, nx = data.shape
            allx = np.arange(nx)
            yy, xx = np.mgrid[:ny:, :nx:]

            echelle_coeff_lst = self.echelle_coeff_lst[conf]
            echelle_goodmask  = self.echelle_goodmask[conf]

            spec_lst = []
            aper = 0
            for iorder, coeff in enumerate(echelle_coeff_lst):
                win = 5
                if not echelle_goodmask[iorder]:
                    continue
                cen_lst = np.polyval(coeff, allx)
                mask = (yy<cen_lst+win)*(yy>cen_lst-win)
                spec = (data*mask).sum(axis=0)

                # pack to table
                row = (aper, 0, spec.size,
                        np.zeros(nx, dtype=np.float64), # wavelength
                        spec,                           # flux
                        np.zeros(nx, dtype=np.int16),   # mask
                       )
                spec_lst.append(row)
                aper += 1

            spec_lst = np.array(spec_lst, dtype=wlcalib_spectype)
            
            hdulst = fits.HDUList([fits.PrimaryHDU(header=header),
                                   fits.BinTableHDU(data=spec_lst),
                                   ])
            hdulst.writeto('spec_{}.fits'.format(fileid), overwrite=True)

            self.echelle_arclamp[fileid] = spec_lst

    def ident_echelle_wavelength(self):
        self.echelle_wave = {}
        #self.echelle_ident = {}

        index_file = os.path.join(os.path.dirname(__file__),
                                  'data/calib/wlcalib_bfosc.dat')


        wlcalib_result_lst = {}

        for logitem in self.logtable:
            if logitem['mode']!='echelle' or logitem['datatype']!='SPECSLAMP':
                continue
            if logitem['fileid'] not in self.echelle_arclamp:
                continue

            fileid = logitem['fileid']
            lamp   = logitem['object']

            spec = self.echelle_arclamp[fileid]
            
            ref_spec, linelist = select_calib_from_database(index_file,
                                lamp = lamp,
                                mode = logitem['mode'],
                                config = logitem['config'],
                                dateobs = logitem['dateobs'],
                                )

            linelist = linelist['aperture', 'order', 'wavelength', 'element', 'ion']

            result = find_echelle_wavelength(spec, ref_spec, (-50,50), linelist,
                    window=15, xdeg=3, ydeg=3, clipping=3, q_threshold=10)


            fig_lbl_lst = result['fig_fitlbl']
            for ifig, fig_lbl in enumerate(fig_lbl_lst):
                figname = 'linefit_lbl_{}_{:02d}.png'.format(
                            fileid, ifig+1)
                fig_lbl.savefig(figname)
                plt.close(fig_lbl)

            fig_sol = result['fig_solution']
            title = '{} ({})'.format(fileid, lamp)
            fig_sol.suptitle(title)
            figname = 'wlcalib_{}.png'.format(fileid)
            figfilename = os.path.join('./', figname)
            fig_sol.savefig(figfilename)
            plt.close(fig_sol)

            wlcalib_result_lst[fileid] = result
            
        ## pick up the best result
        most_lines, most_orders, least_std = 0, 0, 999
        for fileid, result in sorted(wlcalib_result_lst.items()):
            if result['nlines'] > most_lines:
                most_lines = result['nlines']
                most_lines_id = fileid
            if result['norders'] > most_orders:
                most_orders = result['norders']
                most_orders_id = fileid
            if result['std'] < least_std:
                least_std = result['std']
                least_std_id = fileid
        if most_lines_id == most_orders_id and most_orders_id == least_std_id:
            adopted_fileid = most_lines_id
        else:
            adopted_fileid = least_std_id

        self.echelle_wave = wlcalib_result_lst[adopted_fileid]['wavelength']


    def extract_lamp(self):
        lamp_item_lst = filter(lambda item: item['datatype']=='SPECLLAMP',
                               self.logtable)

        hwidth = 5

        spec_lst = {} # use to save the extracted 1d spectra of calib lamp

        for logitem in lamp_item_lst:
            fileid = logitem['fileid']
            filename = self.fileid_to_filename(fileid)
            data = fits.getdata(filename)
            data = data - self.bias_data
            data = data / self.sens_data

            ny, nx = data.shape

            # extract 1d spectra of wavelength calibration lamp
            spec = data[ny//2-hwidth: ny//2+hwidth+1, :].sum(axis=0)
            spec_lst[fileid] = {'wavelength': None, 'flux': spec}

        self.lamp_spec_lst = spec_lst

    def ident_longslit_wavelength(self):
        self.wave = {}
        self.ident = {}

        index_file = os.path.join(os.path.dirname(__file__),
                                  'data/calib/wlcalib_bfosc.dat')

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='SPECLLAMP':
                continue
            if logitem['fileid'] not in self.arclamp:
                continue

            fileid = logitem['fileid']
            lamp   = logitem['object']

            spec = self.arclamp[fileid]

            ref_data, linelist = select_calib_from_database(index_file,
                                lamp = lamp,
                                mode = logitem['mode'],
                                config = logitem['config'],
                                dateobs =logitem['dateobs'],
                                )
            ref_wave = ref_data['wavelength']
            ref_flux = ref_data['flux']

            #filename = os.path.join(os.path.dirname(__file__),
            #                        'data/linelist/{}_l.dat'.format(lamp))
            #linelist = Table.read(filename, format='ascii.fixed_width_two_line')
            #linelist = linelist[np.where(linelist['intlev']<=3)]

            linelist = linelist['wave_air', 'element', 'ion', 'source']

            window = 17
            deg = 5
            clipping = 3
            q_threshold = 5

            result = find_longslit_wavelength(
                    spec, ref_wave, ref_flux, (-50, 50),
                    linelist    = linelist,
                    window      = window,
                    deg         = deg,
                    q_threshold = q_threshold,
                    clipping    = clipping,
                    )

            allwave  = result['wavelength']
            linelist = result['linelist']
            stdwave  = result['std']
            fig_sol  = result['fig_solution']

            # save line-by-line fit figure
            fig_lbl_lst  = result['fig_fitlbl']
            if len(fig_lbl_lst)==1:
                fig_lbl = fig_lbl_lst[0]
                title = 'Line-by-line fit of {} ({})'.format(fileid, lamp)
                fig_lbl.suptitle(title)
                figname = 'linefit_lbl_{}.png'.format(fileid)
                figfilename = os.path.join(self.reduction_path, figname)
                fig_lbl.savefig(figfilename)
                plt.close(fig_lbl)
            else:
                for ifig, fig_lbl in enumerate(fig_lbl_lst):
                    # add figure title
                    title = 'Line-by-line fit of {} ({}, {} of {})'.format(
                            fileid, lamp, ifig+1, len(fig_lbl_lst))
                    fig_lbl.suptitle(title)
                    figname = 'linefit_lbl_{}_{:02d}.png'.format(
                            fileid, ifig+1)
                    figfilename = os.path.join(self.reduction_path, figname)
                    fig_lbl.savefig(figfilename)
                    plt.close(fig_lbl)


            # save wavelength solution figure
            # add title
            title = 'Wavelength Solution for {} ({})'.format(fileid, lamp)
            fig_sol.suptitle(title)
            figname = 'wlcalib_{}.png'.format(fileid)
            figfilename = os.path.join(self.reduction_path, figname)
            fig_sol.savefig(figfilename)
            plt.close(fig_sol)

            # prepare the wavelength calibrated arc lamp spectra
            newtable = Table([allwave, spec], names=['wavelength', 'flux'])

            ntotal = len(linelist)
            nused = sum(list(linelist['use']))

            # prepare the FITS header
            head = fits.Header()
            head['OBSERVAT'] = 'Xinglong'
            head['TELESCOP'] = 'Xinglong 2.16m'
            head['INSTRUME'] = 'BFOSC'
            head['FILEID']   = fileid
            head['OBJECT']   = lamp
            head['EXPTIME']  = logitem['exptime']
            head['DATEOBS']  = logitem['dateobs']
            head['MODE']     = logitem['mode']
            head['CONFIG']   = logitem['config']
            head['SLIT']     = logitem['slit']
            head['BINNING']  = logitem['binning']
            head['GAIN']     = logitem['gain']
            head['RDNOISE']  = logitem['rdnoise']
            head['FITFUNC']  = 'GENGAUSSIAN'
            head['WINDOW']   = window
            head['FITDEG']   = deg
            head['CLIPPING'] = clipping
            head['QTHOLD']   = q_threshold
            head['WAVERMS']  = stdwave
            head['NTOTAL']   = ntotal
            head['NUSED']    = nused
            hdulst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(data=newtable),
                        fits.BinTableHDU(data=linelist),
                ])
            hdulst.writeto('wlcalib_{}.fits'.format(fileid),
                           overwrite=True)

            self.wave[fileid] = allwave
            self.ident[fileid] = linelist

    def plot_wlcalib(self):
        fig = plt.figure(figsize=(12,6), dpi=150)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(self.wavelength, self.calibflux, lw=0.5)
        ax1.set_xlabel(u'Wavelength (\xc5)')
        plt.show()
    
    def extract_longslit_arclamp(self):
        self.arclamp = {}
        for logitem in self.logtable:
            if logitem['mode'] != 'longslit':
                continue

            if logitem['datatype'] != 'SPECLLAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            fileid = logitem['fileid']

            filename = self.fileid_to_filename(fileid)
            data = fits.getdata(filename)
            data = data - self.bias[ccdconf]
            data = data / self.sensmap[conf]

            ny, nx = data.shape
            winlen = 10
            hwidth = int(winlen/2)
            spec = data[ny//2-hwidth:ny//2+hwidth, :].mean(axis=0)

            self.arclamp[fileid] = spec

    def find_longslit_distortion(self):

        self.distortion = {}

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='SPECLLAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            fileid = logitem['fileid']
            filename = self.fileid_to_filename(fileid)
            data = fits.getdata(filename)
            data = data - self.bias[ccdconf]
            data = data / self.sensmap[conf]

            allwave = self.wave[fileid]
            linelist = self.ident[fileid]

            distortion, fig1, fig3d = find_distortion(data,
                                        hwidth=5, disp_axis='x',
                                        linelist=linelist,
                                        deg=5, xorder=3, yorder=3)
            fig1.suptitle('Distortion of {}'.format(fileid))
            fig1.savefig('distortion_{}.pdf'.format(fileid))
            fig1.savefig('distortion_{}.png'.format(fileid))
            plt.close(fig1)
            #fig3d.suptitle('Distortion of {}'.format(fileid))
            fig3d.suptitle('BFOSC')
            fig3d.savefig('distortion3d_{}.png'.format(fileid))
            plt.close(fig3d)

            # plot distortion map
            fig = distortion.plot(times=10)
            fig.suptitle('BFOSC', fontsize=13)
            fig.savefig('distortion_bfosc.png')
            plt.close(fig)

            newdata = distortion.correct_image(data)
            fits.writeto('distcorr_{}.fits'.format(fileid), newdata, overwrite=True)

            #fig = plt.figure()
            #ax1 = fig.add_subplot(121)
            #ax2 = fig.add_subplot(122)
            #ax1.imshow(data)
            #ax2.imshow(newdata)

            #fig2 = plt.figure()
            #ax1 = fig2.add_subplot(121)
            #ax2 = fig2.add_subplot(122)
            #for y in np.arange(0, newdata.shape[0], 100):
            #    ax1.plot(data[y, :]+y*10, lw=0.5)
            #    ax2.plot(newdata[y, :]+y*10, lw=0.5)
            #plt.show()
            self.distortion[conf] = distortion

    def extract_longslit_targets(self, search_region=(0.2, 0.8)):
        func = lambda item: item['datatype']=='SPECLTARGET'
        logitem_lst = list(filter(func, self.logtable))
        for logitem in logitem_lst:
            self.extract(logitem, search_region=search_region)

    def extract(self, logitem, search_region):

        # column-by-column figure of optimal extraction
        plot_opt_columns = False

        ccdconf = self.get_ccdconf(logitem)
        conf = self.get_conf(logitem)

        fileid  = logitem['fileid']
        ccd_gain = logitem['gain']
        ccd_ron  = logitem['rdnoise']
        filename = self.fileid_to_filename(fileid)
        data, head = fits.getdata(filename, header=True)
        data = data - self.bias[ccdconf]
        data = data / self.sensmap[conf]

        distortion = self.distortion[conf]
        data = distortion.correct_image(data)

        print('* FileID: {} - 1d spectra extraction'.format(fileid))

        ny, nx = data.shape
        allx = np.arange(nx)
        ally = np.arange(ny)

        y1 = int(ny * search_region[0])
        y2 = int(ny * search_region[1])

        # search the target from Y=(y1, y2)
        ymax = data[y1:y2, 30:250].mean(axis=1).argmax() + y1
        result = trace_target(data, ymax, xstep=50, polyorder=3)
        coeff_loc, fwhm_mean, profile_func, tracefig = result[:]

        # set and save figures
        figname = 'trace_{}.png'.format(fileid)
        figfilename = os.path.join(self.figpath, figname)
        title = 'Trace for {} ({})'.format(fileid, logitem['object'])
        #tracefig.suptitle(title)
        tracefig.savefig(figfilename)
        plt.close(tracefig)

        # find closet wavelength calibration 
        mid_time = dateutil.parser.parse(logitem['dateobs']) + \
                datetime.timedelta(seconds=logitem['exptime']/2)


        timediff_lst = []
        calibfileid_lst = []
        for _fileid in self.wave.keys():
            _logitem = self.logtable[self.logtable['fileid']==_fileid][0]
            calib_midtime = dateutil.parser.parse(_logitem['dateobs']) + \
                    datetime.timedelta(seconds=_logitem['exptime']/2)
            timediff = mid_time - calib_midtime
            timediff_lst.append(timediff)
            calibfileid_lst.append(_fileid)
        argmin = np.argmin(timediff_lst)
        calib_fileid = calibfileid_lst[argmin]

        wave = self.wave[calib_fileid].copy()

        # extract 1d sepectra
        ycen = np.polyval(coeff_loc, allx) 
        # summ extraction
        yy, xx = np.mgrid[:ny, :nx]
        upper_line = ycen + fwhm_mean
        lower_line = ycen - fwhm_mean
        upper_ints = np.int32(np.round(upper_line))
        lower_ints = np.int32(np.round(lower_line))
        extmask = (yy > lower_ints) * (yy < upper_ints)
        mask = np.float32(extmask)
        # determine the weights in the boundary
        mask[upper_ints, allx] = (upper_line + 0.5) % 1
        mask[lower_ints, allx] = 1 - (lower_line + 0.5) % 1

        # extract
        spec_sum = (data * mask).sum(axis=0)
        nslit = mask.sum(axis=0)

        # initialize background mask
        bkgmask = np.zeros_like(data, dtype=bool)
        # index of rows to exctract background
        background_rows = [(520, 800), (1000, 1250)]
        for r1, r2 in background_rows:
            bkgmask[r1:r2, :] = True

        # remove cosmic rays in the background region
        # ori_bkgspec= (cdata*bkgmask).sum(axis=0)/(bkgmask.sum(axis=0))
     
        # method 1
        # for r1, r2 in background_rows:
        #    cutdata = cdata[r1:r2, :]
        #    fildata = median_filter(cutdata, (1, 5), mode='nearest')
        #    resdata = cutdata - fildata
        #    std = resdata.std()
        #    mask = (resdata < 3*std)*(resdata > -3*std)
        #    bkgmask[r1:r2, :] = mask
     
        # method 2
        for r1, r2 in background_rows:
            cutdata = data[r1:r2, :]
            mean = cutdata.mean(axis=0)
            std = cutdata.std()
            mask = (cutdata < mean + 3 * std) * (cutdata > mean - 3 * std)
            bkgmask[r1:r2, :] = mask

        # plot the bkg and bkg mask
        # fig0 = plt.figure()
        # ax01 = fig0.add_subplot(121)
        # ax02 = fig0.add_subplot(122)
        # for r1, r2 in background_rows:
        #    for y in np.arange(r1, r2):
        #        ax01.plot(cdata[y, :]+y*15, lw=0.5)
        #        m = ~bkgmask[y, :]
        #        ax01.plot(allx[m], cdata[y, :][m]+y*15, 'o', color='C0')
        # bkgspec = (cdata*bkgmask).sum(axis=0)/(bkgmask.sum(axis=0))
        # ax02.plot(ori_bkgspec)
        # ax02.plot(bkgspec)
        # plt.show()
     
        # remove the peaks in the spatial direction
        # sum of background mask along y
        bkgmasksum = bkgmask.sum(axis=1)
        # find positive positions
        posmask = np.nonzero(bkgmasksum)[0]
        # initialize crossspec
        crossspec = np.zeros(ny)
        crossspec[posmask] = (data * bkgmask).sum(axis=1)[posmask] / bkgmasksum[posmask]
        fitx = ally[posmask]
        fity = crossspec[posmask]
        fitmask = np.ones_like(posmask, dtype=bool)
        maxiter = 3
        for i in range(maxiter):
            c = np.polyfit(fitx[fitmask], fity[fitmask], deg=2)
            res_lst = fity - np.polyval(c, fitx)
            std = res_lst[fitmask].std()
            new_fitmask = (res_lst > -2 * std) * (res_lst < 2 * std)
            if new_fitmask.sum() == fitmask.sum():
                break
            fitmask = new_fitmask
     
        # block these pixels in bkgmask
        for y in ally[posmask][~fitmask]:
            bkgmask[y, :] = False
     
        # plot the cross-section of background regions
        figbkg = plt.figure(figsize=(9, 6), dpi=200)
        ax1 = figbkg.add_axes([0.07, 0.54, 0.87, 0.36])
        ax2 = figbkg.add_axes([0.07, 0.12, 0.87, 0.36])
        newy = np.polyval(c, ally)
        for ax in figbkg.get_axes():
            ax.plot(ally, data.mean(axis=1), alpha=0.3, color='C0', lw=0.7)
        y1, y2 = ax1.get_ylim()
     
        ylst = ally[posmask][fitmask]
        for idxlst in np.split(ylst, np.where(np.diff(ylst) != 1)[0] + 1):
            for ax in figbkg.get_axes():
                ax.plot(ally[idxlst], crossspec[idxlst], color='C0', lw=0.7)
                ax.fill_betweenx([y1, y2], idxlst[0], idxlst[-1],
                                 facecolor='C2', alpha=0.15)
     
        for ax in figbkg.get_axes():
            ax.plot(ally, newy, color='C1', ls='-', lw=0.5)
     
        ax2.plot(ally, newy + std, color='C1', ls='--', lw=0.5)
        ax2.plot(ally, newy - std, color='C1', ls='--', lw=0.5)
        for ax in figbkg.get_axes():
            ax.set_xlim(0, ny - 1)
            ax.grid(True, ls='--', lw=0.5)
            ax.set_axisbelow(True)
        ax1.set_ylim(y1, y2)
        ax2.set_ylim(newy.min() - 6 * std, newy.max() + 6 * std)
        ax2.set_xlabel('Y (pixel)')
        title = '{} ({})'.format(fileid, logitem['object'])
        figbkg.suptitle(title)
        figname = 'bkg_cross_{}.png'.format(fileid)
        figfilename = os.path.join(self.figpath, figname)
        figbkg.savefig(figfilename)
        plt.close(figbkg)
     
        # plot a 2d image of distortion corrected image
        # and background region
        fig3 = plt.figure(dpi=200, figsize=(10, 5))
        ax31 = fig3.add_axes([0.07, 0.1, 0.42, 0.84])
        ax32 = fig3.add_axes([0.57, 0.1, 0.42, 0.84])
        vmin = np.percentile(data, 10)
        vmax = np.percentile(data, 99)
        ax31.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
        bkgdata = np.zeros_like(data, dtype=data.dtype)
        bkgdata[bkgmask] = data[bkgmask]
        bkgdata[~bkgmask] = (vmin + vmax) / 2
        ax32.imshow(bkgdata, origin='lower', vmin=vmin, vmax=vmax)
        for ax in fig3.get_axes():
            ax.set_xlim(0, nx - 1)
            ax.set_ylim(0, ny - 1)
            ax.set_xlabel('X (pixel)')
            ax.set_ylabel('Y (pixel)')
        title = '{} ({})'.format(fileid, logitem['object'])
        #fig3.suptitle(title)
        figname = 'bkg_region_{}.png'.format(fileid)
        figfilename = os.path.join(self.figpath, figname)
        fig3.savefig(figfilename)
        plt.close(fig3)
     
        # background spectra per pixel along spatial direction
        bkgspec = (data * bkgmask).sum(axis=0) / (bkgmask.sum(axis=0))
        # background spectra in the spectrum aperture
        background_sum = bkgspec * nslit
     
        spec_sum_dbkg = spec_sum - background_sum

        ####### optimal extraction ##########
        debkg_data = data - np.repeat([bkgspec], ny, axis=0)

        fitprof_func = lambda p, x: p[0] * profile_func(x) + p[1]
        f_opt_lst = []
        b_opt_lst = []
        for x in np.arange(nx):
            ycenint = np.int32(np.round(ycen[x]))
            y1 = ycenint - 18
            y2 = ycenint + 19
            fitx = ally[y1:y2] - ycenint
            flux = data[y1:y2, x]
            debkg_flux = debkg_data[y1:y2, x]
            mask = np.ones(y2 - y1, dtype=bool)

            # b0 = (flux[0]+flux[-1])/2
            b0 = bkgspec[x]
            p0 = [flux.max() - b0, b0]
            maxiter = 6
            for ite in range(maxiter):
                fitres = opt.least_squares(errfunc, p0,
                                args=(fitx[mask], flux[mask], fitprof_func))
                p = fitres['x']
                res_lst = errfunc(p, fitx, flux, fitprof_func)
                std = res_lst[mask].std()
                new_mask = res_lst < 3 * std
                if new_mask.sum() == mask.sum():
                    break
                mask = new_mask

            # plot the column-by-column fitting figure
            if plot_opt_columns:
                nrow = 5
                ncol = 7
                if x % (nrow * ncol) == 0:
                    fig = plt.figure(figsize=(14, 8), dpi=200)
                iax = x % (nrow * ncol)
                icol = iax % ncol
                irow = int(iax / ncol)
                w1 = 0.95 / ncol
                w2 = w1 - 0.025
                h1 = 0.96 / nrow
                h2 = h1 - 0.025
                ax = fig.add_axes([0.05 + icol * w1, 0.05 + (nrow - irow - 1) * h1, w2, h2])
                ax.scatter(fitx, flux, c='w', edgecolor='C0', s=15)
                ax.scatter(fitx[mask], flux[mask], c='C0', s=15)
                newx = np.arange(y1, y2 + 1e-3, 0.1) - ycenint
                newy = fitprof_func(p, newx)
                ax.plot(newx, newy, ls='-', color='C1')
                ax.plot(newx, newy + std, ls='--', color='C1')
                ax.plot(newx, newy - std, ls='--', color='C1')
                ylim1, ylim2 = ax.get_ylim()
                ax.text(0.95 * fitx[0] + 0.05 * fitx[-1], 0.1 * ylim1 + 0.9 * ylim2,
                        'X = {:4d}'.format(x))
                ax.axvline(x=0, c='k', ls='--', lw=0.5)
                ax.set_ylim(ylim1, ylim2)
                ax.set_xlim(fitx[0], fitx[-1])
                if iax == (nrow * ncol - 1) or x == nx - 1:
                    figname = 'fit_{}_{:04d}.png'.format(fileid, x)
                    if not os.path.exists('debug'):
                        os.mkdir('debug')
                    figfilename = os.path.join('debug', figname)
                    fig.savefig(figfilename)
                    plt.close(fig)

            # variance array
            s_lst = 1 / (np.maximum(flux * ccd_gain, 0) + ccd_ron ** 2)
            profile = profile_func(fitx)
            normpro = profile / profile.sum()
            fopt = ((s_lst * normpro * debkg_flux)[mask].sum()) / \
                   ((s_lst * normpro ** 2)[mask].sum())
  
            bkg_flux = np.repeat(bkgspec[x], y2 - y1)
            bopt = ((s_lst * normpro * bkg_flux)[mask].sum()) / \
                   ((s_lst * normpro ** 2)[mask].sum())
            f_opt_lst.append(fopt)
            b_opt_lst.append(bopt)
        f_opt_lst = np.array(f_opt_lst)
        b_opt_lst = np.array(b_opt_lst)
  
        spec_opt_dbkg = f_opt_lst
        background_opt = b_opt_lst
        spec_opt = spec_opt_dbkg + background_opt
  
        # now:
        #                      |      sum       |    optimal
        # ---------------------+----------------+----------------
        # backgroud:           | background_sum | background_opt
        # target + background: | spec_sum       | spec_opt
        # target:              | spec_sum_dbkg  | spec_opt_dbkg

        # save 1d spectra to ascii files

        if wave[0] > wave[-1]:
            # reverse spectrum
            wave = wave[::-1]
            spec_sum       = spec_sum[::-1]
            spec_opt       = spec_opt[::-1]
            spec_opt_dbkg  = spec_opt_dbkg[::-1]
            spec_sum_dbkg  = spec_sum_dbkg[::-1]
            background_opt = background_opt[::-1]
            background_sum = background_sum[::-1]

        types = [
                ('wavelength',      np.float64),
                ('flux_opt',        np.float32),
                ('background_opt',  np.float32),
                ('flux_sum',        np.float32),
                ('background_sum',  np.float32),
                ]
        names, formats=list(zip(*types))
        spectype = np.dtype({'names': names, 'formats': formats})

        data = []
        for w, f1, b1, f2, b2 in zip(wave, spec_opt_dbkg, background_opt,
                                           spec_sum_dbkg, background_sum):
            data.append((w, f1, b1, f2, b2))
        data = np.array(data, dtype=spectype)


        # prepare header
        mid_time_utc = mid_time - datetime.timedelta(hours=8)
        head['HIERARCH BYSPEC MIDTIME'] = mid_time_utc.isoformat()
        # calculate barycentric velocity correction
        barycorr = self.get_barycorr(logitem['RAJ2000'], logitem['DEJ2000'],
                                     mid_time_utc)
        head['HIERARCH BYSPEC BARYCORR'] = barycorr

        fname = 'spec_{}.fits'.format(fileid)
        oned_path = 'onedspec'
        if not os.path.exists(oned_path):
            os.mkdir(oned_path)
        filename = os.path.join(oned_path, fname)
        hdulst = fits.HDUList([fits.PrimaryHDU(header=head),
                               fits.BinTableHDU(data=data),
                               ])
        hdulst.writeto(filename, overwrite=True)

def errfunc(p, x, y, fitfunc):
    return y - fitfunc(p, x)

def trace_target(data, ypos, xstep, polyorder):


    def fitfunc(p, x):
        return gaussian(p[0], p[1], p[2], x)+p[3]


    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    xscan_lst, ycen_lst, fwhm_lst = [], [], []
    xnode_lst, ynode_lst = [], []

    # make a plot
    fig1 = plt.figure(figsize=(10, 6.7), dpi=200)
    w1 = 0.39
    ax1 = fig1.add_axes([0.07, 0.45, 0.39, 0.50])  # upper left
    ax2 = fig1.add_axes([0.56, 0.07, w1, 0.22])    # lower left
    ax3 = fig1.add_axes([0.07, 0.07, 0.39, 0.30])   # lower left
    ax4 = fig1.add_axes([0.56, 0.37, w1, w1/2*3])   # upper right
    offset_mark = 0
    yc = ypos
    for ix, x in enumerate(np.arange(30, nx-200, xstep)):
        xdata = ally
        # ydata = data[:,x]
        ydata = data[:, x-20:x+21].mean(axis=1)
        y1 = yc - 20
        y2 = yc + 20
        yc = ydata[y1:y2].argmax() + y1
        succ = True

        for i in range(2):
            y1 = yc - 20
            y2 = yc + 20
            xfitdata = ally[y1:y2]
            yfitdata = ydata[y1:y2]
            p0 = [yfitdata.max() - yfitdata.min(), 6.0, yc, yfitdata.min()]

            mask = np.ones_like(xfitdata, dtype=bool)
            for j in range(2):
                fitres = opt.least_squares(errfunc, p0,
                            bounds=([0,      3,  -np.inf, -np.inf],
                                    [np.inf, 50, np.inf,  np.inf]),
                            args=(xfitdata[mask], yfitdata[mask], fitfunc))
                p = fitres['x']
                res_lst = errfunc(p, xfitdata, yfitdata, fitfunc)
                std = res_lst[mask].std()
                new_mask = (res_lst > -3 * std) * (res_lst < 3 * std)
                mask = new_mask

            A, fwhm, center, bkg = p
            if A < 0 or fwhm > 100 or fwhm < 1:
                succ = False
                break
            yc = int(round(center))

        if not succ:
            continue

        # pack results
        xscan_lst.append(x)
        ycen_lst.append(center)
        fwhm_lst.append(fwhm)

        newx = np.arange(xfitdata[0], xfitdata[-1], 0.1)
        newy = fitfunc(p, newx)

        # plot fitting in ax1
        # determine the color

        if offset_mark == 0:
            offset_rate = A * 0.002
            offset_mark = 1
        offset = x * offset_rate

        color = 'C{:d}'.format(ix%10)
        ax1.scatter(xfitdata, yfitdata-bkg+offset,
                    alpha=0.5, s=4, c=color)
        ax1.plot(newx, newy-bkg+offset,
                 alpha=0.5, lw=0.5, color=color)
        ax1.vlines(center, offset, A+offset,
                   color=color, lw=0.5, alpha=0.5)
        ax1.hlines(offset, center-fwhm, center+fwhm,
                   color=color, lw=0.5, alpha=0.5)

        # plot stacked profiles in ax3
        xprofile = xfitdata - center
        yprofile = (yfitdata - bkg) / A
        ax3.plot(xprofile, yprofile, color=color, alpha=0.5, lw=0.5)

        for vx, vy in zip(xprofile, yprofile):
            xnode_lst.append(vx)
            ynode_lst.append(vy)

    xscan_lst = np.array(xscan_lst)
    ycen_lst = np.array(ycen_lst)
    fwhm_lst = np.array(fwhm_lst)

    # fit the order position with iterative polynomial
    mask = np.ones_like(xscan_lst, dtype=bool)
    for i in range(3):
        coeff_loc = np.polyfit(xscan_lst[mask], ycen_lst[mask], deg=polyorder)
        res_lst = ycen_lst - np.polyval(coeff_loc, xscan_lst)
        std = res_lst[mask].std()
        new_mask = (res_lst < 3 * std) * (res_lst > -3 * std)
        if new_mask.sum()==mask.sum():
            break
        mask = new_mask

    # final position
    ycen = np.polyval(coeff_loc, allx)

    # calculate averaged FWHM
    fwhm_mean, _, _ = get_clip_mean(fwhm_lst)

    # adjust fitting in ax1
    ax1.set_xlim(ycen_lst.min() - 20, ycen_lst.max() + 20)
    ax1.set_ylim(-offset_rate * 200, offset_rate * 2500)
    ax1.set_xlabel('Y (pixel)')
    ax1.set_ylabel('Flux (with offset)')

    # plot fitted positon in ax2
    ax2.scatter(xscan_lst, ycen_lst,
                alpha=0.8, s=20, edgecolor='C0', c='w')
    ax2.scatter(xscan_lst[mask], ycen_lst[mask],
                alpha=0.8, s=20, c='C0')
    ax2.errorbar(xscan_lst, ycen_lst, yerr=fwhm_lst,
                 fmt='o', capsize=0, ms=0, zorder=-1)
    ax2.plot(allx, ycen, ls='-', color='C3', lw=1, alpha=0.6)
    ax2.plot(allx, ycen + std, ls='--', color='C3', lw=1, alpha=0.6)
    ax2.plot(allx, ycen - std, ls='--', color='C3', lw=1, alpha=0.6)
    ax2.grid(True, ls='--')
    ax2.set_axisbelow(True)
    x1, x2 = 0, nx - 1
    y1 = (ycen - fwhm_mean * 3).min()
    y2 = (ycen + fwhm_mean * 3).max()
    ax2.text(0.95 * x1 + 0.05 * x2, 0.15 * y1 + 0.85 * y2,
             'Mean FWHM = {:4.2f}'.format(fwhm_mean))
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_xlabel('X (pixel)')
    ax2.set_ylabel('Y (pixel)')

    # calculate average profile
    xnode_lst = np.array(xnode_lst)
    ynode_lst = np.array(ynode_lst)

    step = 1
    xprofile_node_lst = []
    yprofile_node_lst = []
    for x in np.arange(-18, 18, step):
        x1 = x - step / 2
        x2 = x + step / 2
        m = (xnode_lst > x1) * (xnode_lst < x2)
        ymean, _, mask = get_clip_mean(ynode_lst[m])
        xprofile_node_lst.append(xnode_lst[m][mask].mean())
        yprofile_node_lst.append(ymean)
    xprofile_node_lst = np.array(xprofile_node_lst)
    yprofile_node_lst = np.array(yprofile_node_lst)

    yprofile_node_lst = np.maximum(yprofile_node_lst, 0)
    interf = intp.InterpolatedUnivariateSpline(
        xprofile_node_lst, yprofile_node_lst, k=3, ext=1)

    # adjust ax3
    ax3.plot(xprofile_node_lst, yprofile_node_lst, 'o-',
             ms=3, color='k', lw=0.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.grid(True, ls='--', lw=0.5)
    ax3.axvline(0, ls='--', lw=1, color='k')
    ax3.axhline(0, ls='--', lw=1, color='k')
    ax3.set_axisbelow(True)
    ax3.set_xlabel('X (pixel)')

    # plot image data in ax4
    vmin = np.percentile(data, 10)
    vmax = np.percentile(data, 99)
    ax4.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    ax4.set_xlim(0, nx - 1)
    ax4.set_ylim(0, ny - 1)
    ax4.set_xlabel('X (pixel)')
    ax4.set_ylabel('Y (pixel)')
    ax4.plot(allx, ycen, ls='-', color='C3', lw=0.5, alpha=1)
    ax4.plot(allx, ycen + fwhm_mean, ls='--', color='C3', lw=0.5, alpha=1)
    ax4.plot(allx, ycen - fwhm_mean, ls='--', color='C3', lw=0.5, alpha=1)

    return coeff_loc, fwhm_mean, interf, fig1
