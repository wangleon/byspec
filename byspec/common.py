import os

from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import scipy.optimize as opt
import scipy.interpolate as intp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from .imageproc import combine_images
from .onedarray import gengaussian, get_simple_ccf
from .regression import polyfit2d, polyval2d

class FOSCReducer(object):
    
    def __init__(self, **kwargs):

        rawdata_path = kwargs.pop('rawdata_path', None)
        if rawdata_path is not None and os.path.exists(rawdata_path):
            self.set_rawdata_path(rawdata_path)

        reduction_path = kwargs.pop('reduction_path', './')
        if reduction_path is not None:
            self.set_reduction_path(reduction_path)

        obslogfile = kwargs.pop('obslog', None)
        if obslogfile is not None and os.path.exists(obslogfile):
            self.obslogfile = obslogfile
            self.read_obslog(obslogfile)



    def set_rawdata_path(self, rawdata_path):
        """Set rawdata path.
        """
        if os.path.exists(rawdata_path):
            self.rawdata_path = rawdata_path

    def set_reduction_path(self, reduction_path):
        """Set data reduction path

        """
        if not os.path.exists(reduction_path):
            os.mkdir(reduction_path)
        self.reduction_path = reduction_path

        self.figpath = os.path.join(self.reduction_path, 'figures')
        if not os.path.exists(self.figpath):
            os.mkdir(self.figpath)

        self.odspath = os.path.join(self.reduction_path, 'onedspec')
        if not os.path.exists(self.odspath):
            os.mkdir(self.odspath)
        #self.bias_file = os.path.join(self.reduction_path, 'bias.fits')
        #self.flat_file = os.path.join(self.reduction_path, 'flat.fits')
        #self.sens_file = os.path.join(self.reduction_path, 'sens.fits')

    def read_obslog(self, filename):
        self.logtable = Table.read(filename,
                            format='ascii.fixed_width_two_line')
        print(self.logtable)

    def get_all_conf(self):
        conf_lst = []
        for logitem in self.logtable:
            conf = self.get_conf(logitem)
            if conf not in conf_lst:
                conf_lst.append(conf)
        self.conf_lst = conf_lst

    def get_all_ccdconf(self):
        ccdconf_lst = []
        for logitem in self.logtable:
            ccdconf = self.get_ccdconf(logitem)
            if ccdconf not in ccdconf_lst:
                ccdconf_lst.append(ccdconf)
        self.ccdconf_lst = ccdconf_lst

    def combine_flat(self):
        """Combine flat images
        """

        conf_lst = []
        # scan the entire logtable
        for logitem in self.logtable:
            if logitem['object']=='FLAT':
                conf = get_conf_string(logitem)
                if conf not in conf_lst:
                    conf_lst.append(conf)

        for conf in conf_lst:
            flat_file = 'flat.{}.fits'.format(conf)

            flat_filename = os.path.join(self.reduction_path, flat_file)

            if os.path.exists(flat_filename):
                self.flat_data[_conf] = fits.getdata(flat_filename)
            else:
                print('Combine Flat')
                data_lst = []
                for logitem in logtable:
                    _conf = self.get_conf_string(logitem)
                    if _conf != conf:
                        continue
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    # correct bias
                    data = data - self.bias_data
                    data_lst.append(data)
                data_lst = np.array(data_lst)
                flat_data = combine_images(data_lst, mode='mean',
                                    upper_clip=5, maxiter=10, maskmode='max')
                fits.writeto(flat_filename, flat_data, ovewrite=True)
                self.flat_data[_conf] = flat_data

    def find_logitem(self, arg):

        item_lst = [logitem for logitem in self.logtable if
                            logitem['frameid']==arg or \
                            logitem['fileid']==arg or \
                            logitem['object']==arg]
        return item_lst

    def has_echelle(self):
        for logitem in self.logtable:
            if logitem['mode']=='echelle':
                return True
        return False

    def has_longslit(self):
        for logitem in self.logtable:
            if logitem['mode']=='longslit':
                return True
        return False

    def get_barycorr(self, ra, dec, obstime):
        loc = EarthLocation.from_geodetic(
                lat     = self.latitude*u.deg,
                lon     = self.longitude*u.deg,
                height  = self.altitude*u.m,
                )
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        barycorr = coord.radial_velocity_correction(
                    kind = 'barycentric',
                    obstime=Time(obstime),
                    location=loc)
        return barycorr.to(u.km/u.s).to_value()

class Distortion(object):

    def __init__(self, coeff, nx, ny, axis):
        self.coeff = coeff
        self.nx = nx
        self.ny = ny
        if axis in ['x', 'y']:
            self.axis = axis
        else:
            raise ValueError
        
    def plot(self, nx=10, ny=10, times=1,
             linestyles=['--', '-'], linewidths=0.5, colors='k'):

        # set line styles
        if isinstance(linestyles, list) or isinstance(linestyles, tuple):
            ls1, ls2 = linestyles[0], linestyles[1]
        else:
            ls1, ls2 = linestyles, linestyles

        # set line widths
        if isinstance(linewidths, list) or isinstance(linewidths, tuple):
            lw1, lw2 = linewidths[0], linewidths[1]
        else:
            lw1, lw2 = linewidths, linewidths

        # set line colors
        if isinstance(colors, list) or isinstance(colors, tuple):
            c1, c2 = colors[0], colors[1]
        else:
            c1, c2 = colors, colors

        # plot the distortion curve
        fig = plt.figure(dpi=150, figsize=(5.5, 5))
        ax = fig.add_axes([0.12, 0.1, 0.82, 0.82], aspect=1)
        #ax.imshow(np.log10(data), cmap='gray')

        # plot the vertical lines
        for x in np.linspace(0, self.nx-1, nx):
            y_lst = np.linspace(0, self.ny-1, 100)
            x_lst = np.repeat(x, y_lst.size)
            dc_lst = polyval2d(self.coeff, x_lst/self.nx, y_lst/self.ny)

            ax.plot(x_lst, y_lst, ls=ls1, c=c1, lw=lw1)

            if self.axis=='x':
                x_lst += dc_lst*times
            else:
                y_lst += dc_lst*times

            ax.plot(x_lst, y_lst, ls=ls2, c=c2, lw=lw2)

        # plot the horizontal lines
        for y in np.linspace(0, self.ny-1, ny):
            x_lst = np.linspace(0, self.nx-1, 100)
            y_lst = np.repeat(y, x_lst.size)
            dc_lst = polyval2d(self.coeff, x_lst/self.nx, y_lst/self.ny)

            ax.plot(x_lst, y_lst, ls=ls1, c=c1, lw=lw1)

            if self.axis=='x':
                x_lst += dc_lst*times
            else:
                y_lst += dc_lst*times

            ax.plot(x_lst, y_lst, ls=ls2, c=c2, lw=lw2)

        xlabel = 'X (pixel)'
        ylabel = 'Y (pixel)'
        if self.axis=='x':
            xlabel += ' (Dispertion Direction)'
        else:
            ylabel += ' (Dispertion Direction)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig

    def correct_image(self, data):
        ny, nx = data.shape
        if nx != self.nx or ny != self.ny:
            print('Image shape and distortion data mismatch')
            return None
    
        newdata = np.zeros_like(data)
        allx = np.arange(nx)
        ally = np.arange(ny)
    
        yy, xx = np.mgrid[:ny:, :nx:]
        dc = polyval2d(self.coeff, xx/nx, yy/ny)

        if self.axis == 'x':
            for y in np.arange(ny):
                dc_lst = dc[y, :]
                flux = data[y, :]
                func = intp.InterpolatedUnivariateSpline(
                        allx-dc_lst, flux, k=3, ext=1)
                newdata[y, :] = func(allx)
        elif self.axis == 'y':
            for x in np.arange(nx):
                dc_lst = dc[:, x]
                flux = data[:, x]
                func = intp.InterpolatedUnivariateSpline(
                        ally-dc_lst, flux, k=3, ext=1)
                newdata[:, x] = func(ally)
        else:
            raise ValueError
    
        return newdata

def find_distortion(data, hwidth, disp_axis, linelist, deg,
                    xorder, yorder, verbose=False):

    if disp_axis == 'x':
        data = data.T
        transpose = True
    elif disp_axis == 'y':
        transpose = False
    else:
        raise ValueError

    # define the line fitting function and error function
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)
    def fitline(p, x):
        nline = int((len(p)-1)/4)
        y = np.ones_like(x, dtype=np.float64) + p[0]
        for i in range(nline):
            A, alpha, beta, center = p[i*4+1:i*4+5]
            y = y + gengaussian(A, alpha, beta, center, x)
        return y

    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)
    # now Y is the dispersion direction

    # get the central spectra as reference
    spec0 = data[:, nx//2-hwidth:nx//2+hwidth].mean(axis=1)

    all_x_lst, all_y_lst, all_dc_lst, all_w_lst = [], [], [], []

    # scan the slit direction every 200 columns
    for x in np.arange(100, nx-100, 200):

        y_lst, w_lst, dc_lst, m_lst = [], [], [], []

        # get the spectra
        spec1 = data[:, x-hwidth:x+hwidth].mean(axis=1)

        # fig0 is the line-by-line figure
        #fig0 = plt.figure(figsize=(16,9), dpi=200,tight_layout=True)

        #newlines = linelist[np.where(linelist['use']==1)]
        count_line = 0
        #for fitid in np.unique(list(newlines['fitid'])):
        for fitid in np.unique(list(linelist['fitid'])):
            lines = linelist[np.where(linelist['fitid']==fitid)]
            # get the original i1 and i2
            i1o = lines[0]['i1']
            i2o = lines[0]['i2']

            # calculte local shift
            ico = int((i1o+i2o)/2)
            j1 = max(i1o-100, 0)
            j2 = min(i1o+100, spec1.size)
            shift_lst = np.arange(-50, 50)
            ccf0_lst = get_simple_ccf(spec1[j1:j2], spec0[j1:j2],
                        shift_lst)
            s0 = shift_lst[ccf0_lst.argmax()]

            i1 = int(i1o + s0)
            i2 = int(i2o + s0)

            xdata = np.arange(i1, i2)
            ydata = spec1[i1:i2]

            if ydata.argmax()==0 or ydata.argmax()==i2-i1-1:
                continue
            p0 = [lines[0]['bkg']]
            lower_bounds = [-np.inf]
            upper_bounds = [np.inf]

            for line in lines:
                p0.append(line['A'])
                p0.append(line['alpha'])
                p0.append(line['beta'])
                p0.append(line['pixel'] + s0)
                lower_bounds.append(0)
                lower_bounds.append(0.5)
                lower_bounds.append(0.1)
                #lower_bounds.append(line['pixel']+s0-15)
                lower_bounds.append(i1)
                upper_bounds.append(np.inf)
                upper_bounds.append(20)
                upper_bounds.append(20)
                #upper_bounds.append(line['pixel']+s0+15)
                upper_bounds.append(i2)

            fitres = opt.least_squares(errfunc, p0,
                    bounds=(lower_bounds, upper_bounds),
                    args=(xdata, ydata, fitline),
                    )
            count_line += 1
            param = fitres['x']
            nline = int((len(param)-1)/4)
            bkg = param[0]
            std = np.sqrt(fitres['cost']*2/xdata.size)
            for i in range(nline):
                A, alpha, beta, center = param[i*4+1:i*4+5]
                dc = center - lines[i]['pixel']
                wave = lines[i]['wave_air']
                #print(x, shift0, center)
                if A/std < 5 or \
                    abs(alpha - upper_bounds[i*4+2])<1e-2 or \
                    abs(beta - upper_bounds[i*4+3])<1e-2:
                    mask = False
                else:
                    mask = True
                if verbose:
                    print(' - Col {:4d}'.format(x),
                          '{:4d}'.format(count_line),
                          '{:6.2f}'.format(center-i1),
                          '{:6.2f}'.format(i2-center),
                          ' alpha={:6.3f},'.format(alpha),
                          ' beta={:6.3f},'.format(beta),
                          ' c={:7.2f},'.format(center),
                          ' q={:6.2f}'.format(A/std),
                          int(mask),
                          )

                m_lst.append(mask)
                y_lst.append(center)
                w_lst.append(wave)
                dc_lst.append(dc)

            # draw the fitting line-by0line
            #ax0 = fig0.add_subplot(6,7, count_line)
            #ax0.plot(xdata, ydata*1e-3, 'o', ms=3)
            #newx = np.linspace(xdata[0], xdata[-1], 100)
            #newy = fitline(param, newx)
            #ax0.plot(newx, newy*1e-3)
        #fig0.savefig('align_x_{:04d}.png'.format(x))
        #plt.close(fig0)

        m_lst = np.array(m_lst)
        y_lst = np.array(y_lst)
        w_lst = np.array(w_lst)
        dc_lst = np.array(dc_lst)

        # fit the wavelength solution column by column
        # initialize the mask
        m = m_lst.copy()

        for i in range(10):
            coeff = np.polyfit(y_lst[m], w_lst[m], deg=deg)
            allw = np.polyval(coeff, ally)
            res_lst = w_lst - np.polyval(coeff, y_lst)
            std = res_lst[m].std()
            newm = (np.abs(res_lst)<3*std)*m_lst
            if newm.sum()==m.sum():
                break
            else:
                m = newm

        ## plot the wavelength solution column-by-column
        #figsol = plt.figure(dpi=200)
        #axs1 = figsol.add_subplot(211)
        #axs2 = figsol.add_subplot(212)
        #axs1.plot(y_lst[m], w_lst[m], 'o', c='C0', alpha=0.8)
        #axs1.plot(y_lst[~m], w_lst[~m], 'o', c='none', mec='C0')
        #axs1.plot(ally, np.polyval(coeff, ally), '-', lw=0.5)
        #axs2.plot(y_lst[m], res_lst[m], 'o', alpha=0.8)
        #axs2.plot(y_lst[~m], res_lst[~m], 'o', c='none', mec='C0')
        #axs1.set_xlim(0, ny-1)
        #axs2.set_xlim(0, ny-1)
        #axs2.axhline(0, c='k', ls='-', lw=0.5)
        #axs2.axhline(std, c='k', ls='--', lw=0.5)
        #axs2.axhline(-std, c='k', ls='--', lw=0.5)
        #axs2.set_ylim(-6*std, 6*std)
        #figsol.savefig('sol_{:04d}.png'.format(x))
        #plt.close(figsol)

        for y, w, dc in zip(y_lst[m], w_lst[m], dc_lst[m]):
            all_x_lst.append(x)
            all_y_lst.append(y)
            all_w_lst.append(w)
            all_dc_lst.append(dc)

    all_x_lst = np.array(all_x_lst)
    all_y_lst = np.array(all_y_lst)
    all_w_lst = np.array(all_w_lst)
    all_dc_lst = np.array(all_dc_lst)

    # fit the distortion coefficients
    m = np.ones_like(all_x_lst, dtype=bool)
    for i in range(5):
        coeff = polyfit2d(all_x_lst[m]/nx, all_y_lst[m]/ny, all_dc_lst[m],
                      xorder=xorder, yorder=yorder)
        res_lst = all_dc_lst - polyval2d(coeff, all_x_lst/nx, all_y_lst/ny)
        std = res_lst[m].std()
        newm = np.abs(res_lst)<3*std
        if newm.sum()==m.sum():
            break
        else:
            m = newm

    if transpose:
        # exchange the x and y axies
        coeff = coeff.T
        nx, ny = ny, nx
        all_x_lst, all_y_lst = all_y_lst, all_x_lst


    # plot the 3d fitting
    fig3d = plt.figure(dpi=200, figsize=(4.3,4))
    ax1 = fig3d.add_axes([0.0, 0.05, 0.92, 0.91], projection='3d')
    #ax1 = fig3d.add_subplot(projection='3d')
    ax1.scatter(all_x_lst[m], all_y_lst[m], all_dc_lst[m], c='C0', s=10, lw=0)
    ax1.scatter(all_x_lst[~m], all_y_lst[~m], all_dc_lst[~m], c='C3', s=10, lw=0)
    # plot the 3d surface
    yy, xx = np.mgrid[:ny:, :nx:]
    zz = polyval2d(coeff, xx/nx, yy/ny)
    ax1.plot_surface(xx, yy, zz, lw=0.5, alpha=0.5)
    ax1.set_xlabel('X (pixel)')
    ax1.set_ylabel('Y (pixel)')
    ax1.set_zlabel(u'\u0394 {}'.format(disp_axis.upper()))
    
    # plot the 2D figure
    fig1 = plt.figure(dpi=200, figsize=(10, 5.4))
    ax0 = fig1.add_axes([0.08, 0.05, 0.44, 0.89])  # correction map
    ax1 = fig1.add_axes([0.63, 0.60, 0.34, 0.35]) # X residual
    ax2 = fig1.add_axes([0.63, 0.05, 0.34, 0.35]) # Y residual
    # plot difference
    cax = ax0.imshow(zz, origin='lower')
    cs = ax0.contour(zz, linewidths=0.5, colors='k')
    ax0.clabel(cs, inline=1, fontsize=9, use_clabeltext=True)
    ax0.set_xlabel('X (pixel)')
    ax0.set_ylabel('Y (pixel)')
    bbox0 = ax0.get_position()
    # add color bar
    axc = fig1.add_axes([bbox0.x1+0.02, bbox0.y0, 0.015, bbox0.height])
    fig1.colorbar(cax, cax=axc)
    # plot residuals
    bbox1 = ax1.get_position()
    ax1.set_position([bbox1.x0, bbox0.y1-bbox1.height, bbox1.width, bbox1.height])
    bbox2 = ax2.get_position()
    ax2.set_position([bbox2.x0, bbox0.y0, bbox1.width, bbox1.height])
    ax1.plot(all_x_lst[m],  res_lst[m],  'o', ms=2, c='C0', alpha=0.6)
    ax1.plot(all_x_lst[~m], res_lst[~m], 'o', ms=2, c='C3', alpha=0.6)
    ax2.plot(all_y_lst[m],  res_lst[m],  'o', ms=2, c='C0', alpha=0.6)
    ax2.plot(all_y_lst[~m], res_lst[~m], 'o', ms=2, c='C3', alpha=0.6)
    ax1.set_xlabel('X (pixel)')
    ax2.set_xlabel('Y (pixel)')
    ax1.set_xlim(0, nx-1)
    ax2.set_xlim(0, ny-1)
    for ax in [ax1, ax2]:
        ax.set_ylim(-6*std, 6*std)
        ax.axhline(0, c='k', ls='-', lw=0.5, zorder=-1)
        ax.axhline(-std, c='k', ls='--', lw=0.5, zorder=-1)
        ax.axhline(std, c='k', ls='--', lw=0.5, zorder=-1)

    if transpose:
        axis = 'x'
    else:
        axis = 'y'

    return Distortion(coeff, nx=nx, ny=ny, axis=axis), fig1, fig3d

def find_longslit_wavelength(spec, ref_wave, ref_flux, shift_range, linelist,
        window=17, deg=4, clipping=3, q_threshold=0):

    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)
    def fitline(p, x):
        nline = int((len(p)-1)/4)
        y = np.ones_like(x, dtype=np.float64) + p[0]
        for i in range(nline):
            A, alpha, beta, center = p[i*4+1:i*4+5]
            y = y + gengaussian(A, alpha, beta, center, x)
        return y
    
    ref_pixel = np.arange(ref_wave.size)

    shift_lst = np.arange(shift_range[0], shift_range[1])
    ccf_lst = get_simple_ccf(spec, ref_flux, shift_lst)
    shift = shift_lst[ccf_lst.argmax()]

    # construct an interpolate function that converts wavelength to pixel
    # before constructing the function, the reference wavelength should be
    # increased
    if ref_wave[0] > ref_wave[-1]:
        ref_wave = ref_wave[::-1]
        ref_flux = ref_flux[::-1]
        ref_pixel = ref_pixel[::-1]
    # construct the interpolate function
    f_wave_to_pix = intp.InterpolatedUnivariateSpline(
            ref_wave, ref_pixel, k=3)

    linelist.add_column([-1]*len(linelist), index=-1, name='pixel')
    linelist.add_column([-1]*len(linelist), index=-1, name='i1')
    linelist.add_column([-1]*len(linelist), index=-1, name='i2')
    linelist.add_column([-1]*len(linelist), index=-1, name='fitid')

    n = spec.size
    pixel_lst = np.arange(n)

    hwin = int(window/2)

    for iline, line in enumerate(linelist):
        pix1 = f_wave_to_pix(line['wave_air']) + shift
        cint1 = int(round(pix1))
        i1, i2 = cint1 - hwin, cint1 + hwin+1
        i1 = max(i1, 0)
        i2 = min(i2, n-1)
        linelist[iline]['pixel'] = cint1
        linelist[iline]['i1'] = i1
        linelist[iline]['i2'] = i2

    m = (linelist['pixel']>0)*(linelist['pixel']<n-1)
    linelist = linelist[m]

    # resort the linelist so that the pixels are increased
    linelist.sort('pixel')

    wave_lst, species_lst = [], []
    # fitting parameter results
    alpha_lst, beta_lst, center_lst = [], [], []
    A_lst, bkg_lst, std_lst, fwhm_lst, q_lst = [], [], [], [], []

    # initialize line-by-line figure
    fig_lbl_lst = []
    nrow, ncol = 6, 7       # the number of sub-axes in every figure

    count_line = 0  # fitting counter
    for iline, line in enumerate(linelist):
        if line['fitid'] >=0:
            continue

        i1 = line['i1']
        i2 = line['i2']
        # add background level
        ydata = spec[i1:i2]
        p0 = [ydata.min()]
        p0.append(spec[line['pixel']]-ydata.min())
        p0.append(3.6)
        p0.append(3.5)
        p0.append(line['pixel'])

        lower_bounds = [-np.inf, 0,      0.5, 0.1, i1]
        upper_bounds = [np.inf,  np.inf, 20,   20, i2]

        inextline = iline + 1
        idx_lst = [iline]
        while(True):
            if inextline >= len(linelist):
                break
            nextline = linelist[inextline]
            next_i1 = nextline['i1']
            next_i2 = nextline['i2']
            if next_i1 < i2 - 3:
                i2 = next_i2
                ydata = spec[i1:i2]
                # update backgroud
                p0[0] = ydata.min()
                p0.append(spec[nextline['pixel']] - ydata.min())
                p0.append(3.6)
                p0.append(3.5)
                p0.append(nextline['pixel'])
                lower_bounds.append(0)
                lower_bounds.append(0.5)
                lower_bounds.append(0.1)
                lower_bounds.append(next_i1)
                upper_bounds.append(np.inf)
                upper_bounds.append(20)
                upper_bounds.append(20)
                upper_bounds.append(next_i2)
                idx_lst.append(inextline)
            else:
                break

            inextline += 1

        xdata = np.arange(i1, i2)

        fitres = opt.least_squares(errfunc, p0,
                    bounds=(lower_bounds, upper_bounds),
                    args=(xdata, ydata, fitline),
        )

        param = fitres['x']

        fmt_str = (' - {:2s} {:3s} {:10.4f}: {:4d}-{:4d} '
                   'alpha={:6.3f} beta={:6.3f} center={:8.3f} fwhm={:6.2f} '
                   'q={:6.2f} FITID:{:3d}')

        nline = int((len(param)-1)/4)
        for i in range(nline):
            A, alpha, beta, center = param[i*4+1:i*4+5]
            fwhm = 2*alpha*np.power(np.log(2), 1/beta)
            std = np.sqrt(fitres['cost']*2/xdata.size)
            q = A/std

            line = linelist[iline+i]
            wave_lst.append(line['wave_air'])
            species_lst.append('{} {}'.format(line['element'], line['ion']))
            # append fitting results
            center_lst.append(center)
            A_lst.append(A)
            alpha_lst.append(alpha)
            beta_lst.append(beta)
            fwhm_lst.append(fwhm)
            bkg_lst.append(param[0])
            std_lst.append(std)
            q_lst.append(q)

            print(fmt_str.format(line['element'], line['ion'], line['wave_air'],
                    i1, i2, alpha, beta, center, fwhm, q, count_line))

            linelist['i1'][idx_lst[i]] = i1
            linelist['i2'][idx_lst[i]] = i2
            linelist['fitid'][iline+i] = count_line


        # plot line-by-line
        # create figure
        if count_line%(nrow*ncol)==0:
            fig_lbl = plt.figure(figsize=(16, 9), dpi=150, tight_layout=True)
            fig_lbl_lst.append(fig_lbl)
        # create small axes
        axs = fig_lbl.add_subplot(nrow, ncol, count_line%(nrow*ncol)+1)
        axs.plot(xdata, ydata*1e-3, 'o', c='C0', ms=4, alpha=0.7)
        # plot fitted curve
        newx = np.linspace(xdata[0], xdata[-1], 100)
        newy = fitline(param, newx)
        axs.plot(newx, newy*1e-3, ls='-', color='C1', lw=0.7, alpha=0.7)

        # draw text and plot line center as vertical lines
        text_lst = []
        for i in range(nline):
            A, alpha, beta, center = param[i*4+1:i*4+5]
            line = linelist[iline+i]
            axs.axvline(x=center, color='k', ls='--', lw=0.7)
            text = '{} {} {:9.4f}'.format(line['element'], line['ion'], line['wave_air'])
            text_lst.append(text)
        _text = '\n'.join(text_lst)
        axs.set_xlim(newx[0], newx[-1])
        _x1, _x2 = axs.get_xlim()
        _y1, _y2 = axs.get_ylim()
        _y2 = _y2 + 0.2*(_y2 - _y1)
        axs.text(0.95*_x1+0.05*_x2, 0.05*_y1+0.95*_y2, _text,
                 ha='left', va='top', fontsize=8)
        axs.set_ylim(_y1, _y2)
        if _x2 -_x1 < 25:
            major_tick, minor_tick = 5, 1
        else:
            major_tick, minor_tick = 10, 1
        axs.xaxis.set_major_locator(tck.MultipleLocator(major_tick))
        axs.xaxis.set_minor_locator(tck.MultipleLocator(minor_tick))

        for tick in axs.xaxis.get_major_ticks():
            tick.label1.set_fontsize(7)
        for tick in axs.yaxis.get_major_ticks():
            tick.label1.set_fontsize(7)

        count_line += 1

    ###################################################
    # fit wavelength solution
    # resort the pixel list and wavelength list
    wave_lst = np.array(wave_lst)
    center_lst = np.array(center_lst)
    A_lst = np.array(A_lst)
    std_lst = np.array(std_lst)
    q_lst = np.array(q_lst)

    # resort the center list
    idx = center_lst.argsort()
    center_lst = center_lst[idx]
    wave_lst = wave_lst[idx]

    # begin wavelength solution fit
    mask = q_lst > q_threshold
    #mask = np.ones_like(wave_lst, dtype=bool)

    for i in range(10):
        coeff_wave = np.polyfit(center_lst[mask], wave_lst[mask],
                                deg=deg)
        fitwave = np.polyval(coeff_wave, center_lst)
        reswave = wave_lst - fitwave
        stdwave = reswave[mask].std()
        newmask = (np.abs(reswave) < clipping*stdwave)*(q_lst>q_threshold)
        if newmask.sum()==mask.sum():
            break
        mask = newmask

    allwave = np.polyval(coeff_wave, pixel_lst)
    # prepare the wavelength calibrated arc lamp spectra
    newtable = Table([allwave, spec], names=['wavelength', 'flux'])

    # prepare the identified line list
    linelist = linelist['wave_air', 'element', 'ion', 'source',
                        'i1', 'i2', 'fitid']
    linelist = linelist[linelist['fitid']>=0]
    linelist.add_column(center_lst,     index=-1, name='pixel')
    linelist.add_column(reswave,        index=-1, name='residual')
    linelist.add_column(np.int16(mask), index=-1, name='use')
    linelist.add_column(A_lst,          index=-1, name='A')
    linelist.add_column(alpha_lst,      index=-1, name='alpha')
    linelist.add_column(beta_lst,       index=-1, name='beta')
    linelist.add_column(bkg_lst,        index=-1, name='bkg')
    linelist.add_column(fwhm_lst,       index=-1, name='fwhm')
    linelist.add_column(std_lst,        index=-1, name='std')
    linelist.add_column(q_lst,          index=-1, name='q')
    linelist.meta.clear()

    # plot wavelength solution
    fig_sol = plt.figure(figsize=(8, 6), dpi=200)
    axt1 = fig_sol.add_axes([0.10, 0.40, 0.86, 0.52])
    axt2 = fig_sol.add_axes([0.10, 0.10, 0.86, 0.26])
    #axt4 = fig_sol.add_axes([0.58, 0.54, 0.37, 0.38])
    #axt5 = fig_sol.add_axes([0.58, 0.10, 0.37, 0.38])

    axt1.plot(pixel_lst, allwave, c='k', lw=0.6, zorder=10)
    species = np.unique(species_lst)
    for ispecies, _species in enumerate(sorted(species)):
        m = np.array([v==_species for v in species_lst])
        c = 'C{}'.format(ispecies)
        axt1.plot(center_lst[mask*m], wave_lst[mask*m], 'o',
                  ms=4, c=c, label=_species, alpha=0.7)
        axt1.plot(center_lst[(~mask)*m], wave_lst[(~mask)*m], 'o',
                  ms=5, c='none', mec=c, alpha=0.7)
        axt2.plot(center_lst[mask*m], reswave[mask*m], 'o',
                  ms=4, c=c, alpha=0.7)
        axt2.plot(center_lst[(~mask)*m], reswave[(~mask)*m], 'o',
                  ms=5, c='none', mec=c, alpha=0.7)
    axt1.legend(loc='upper center', ncols=len(species))

    axt2.axhline(y=0, color='k', ls='--', lw=0.6, zorder=-1)
    axt2.axhline(y=stdwave, color='k', ls='--', lw=0.6, zorder=-1)
    axt2.axhline(y=-stdwave, color='k', ls='--', lw=0.6, zorder=-1)
    for ax in fig_sol.get_axes():
        ax.grid(True, color='k', alpha=0.4, ls='--', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(0, n-1)
    y1, y2 = axt2.get_ylim()
    z = max(abs(y1), abs(y2))
    z = max(z, 4*stdwave)
    # draw a text
    _text = u'Order = {}, N = {}, RMS = {:5.3f} \xc5'.format(
            deg, mask.sum(), stdwave)
    axt2.text(0.03*n, 0.85*z, _text, ha='left', va='top')
    axt2.set_ylim(-z, z)
    axt2.set_xlabel('Pixel')
    axt1.set_ylabel(u'\u03bb (\xc5)')
    axt2.set_ylabel(u'\u0394\u03bb (\xc5)')
    axt1.xaxis.set_major_locator(tck.MultipleLocator(200))
    axt1.xaxis.set_minor_locator(tck.MultipleLocator(50))
    axt2.xaxis.set_major_locator(tck.MultipleLocator(200))
    axt2.xaxis.set_minor_locator(tck.MultipleLocator(50))
    axt1.set_xticklabels([])
    #axt4.plot(pixel_lst[0:-1], -np.diff(allwave))
    #axt5.plot(pixel_lst[0:-1], -np.diff(allwave) / (allwave[0:-1]) * 299792.458)
    #axt4.set_ylabel(u'd\u03bb/dx (\xc5)')
    #axt5.set_xlabel('Pixel')
    #axt5.set_ylabel(u'dv/dx (km/s)')

    return {'wavelength': allwave,
            'linelist': linelist,
            'std': stdwave,
            'fig_solution': fig_sol,
            'fig_fitlbl': fig_lbl_lst,
            }


def find_cross_order_shift(spec, ref_spec, shift_lst, ordshift_range):

    # prepare CCF figure
    fig_ccf = plt.figure(figsize=(8, 5), dpi=200, tight_layout=True)

    oshift1, oshift2 = ordshift_range
    oshift_lst = np.arange(oshift1, oshift2+1)
    nshift = oshift_lst.size
    if nshift <= 4:
        nrow, ncol = 2, 2
    elif nshift <= 6:
        nrow, ncol = 2, 3
    elif nshift <= 8:
        nrow, ncol = 2, 4
    else:
        nrow, ncol = 3, 4

    shift_disp_lst = {}
    shift_mean_lst = {}
    all_order_shift_lst = {}
    for ishift, _oshift in enumerate(oshift_lst):
        ax_ccf = fig_ccf.add_subplot(nrow, ncol, ishift+1)
        order_shift_lst = []
        for irow in np.arange(len(spec)):
            irow2 = irow + _oshift

            if irow2 < 0 or irow2 >= len(ref_spec):
                # in case irow2 excess the range of ref_spec, skip.
                continue

            flux = spec[irow]['flux']
            ref_flux = ref_spec[irow2]['flux']

            ccf_lst = get_simple_ccf(flux, ref_flux, shift_lst)
            # plot this CCF
            ax_ccf.plot(shift_lst, ccf_lst, '-', lw=0.5, alpha=0.8)

            shift = shift_lst[ccf_lst.argmax()]
            order_shift_lst.append(shift)
        order_shift_lst = np.array(order_shift_lst)
        all_order_shift_lst[_oshift] = order_shift_lst
        shift_disp_lst[_oshift] = np.std(order_shift_lst)
        shift_mean_lst[_oshift] = np.mean(order_shift_lst)
        # set axes title
        ax_ccf.set_title('ordshift = {}'.format(_oshift), fontsize=8)

    result = sorted(shift_disp_lst.items(), key=lambda i: i[1])[0]
    oshift = result[0]
    order_shift_lst = all_order_shift_lst[oshift]

    return oshift, order_shift_lst, fig_ccf


def find_echelle_wavelength(spec, ref_spec, shift_range, linelist,
        ordshift_range=(-2,2),
        window=15, xdeg=4, ydeg=4, clipping=3, q_threshold=10):
    
    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)
    def fitline(p, x):
        nline = int((len(p)-1)/4)
        y = np.ones_like(x, dtype=np.float64) + p[0]
        for i in range(nline):
            A, alpha, beta, center = p[i*4+1:i*4+5]
            y = y + gengaussian(A, alpha, beta, center, x)
        return y

    shift_lst = np.arange(shift_range[0], shift_range[1])

    allwave = {}
    all_res_lst = []

    # find cross-order shift
    oshift, order_shift_lst, fig_ccf = find_cross_order_shift(
            spec, ref_spec, shift_lst, ordshift_range)


    ################################

    # initialize line-by-line figure
    fig_lbl_lst = []
    nrow, ncol = 6, 7       # the number of sub-axes in every figure
    count_line = 0  # fitting counter

    fig_sol = plt.figure(dpi=200, figsize=(10,5))
    axsol = fig_sol.add_axes([0.08, 0.10, 0.42, 0.85])
    axresx = fig_sol.add_axes([0.58, 0.57, 0.4, 0.38])
    axresy = fig_sol.add_axes([0.58, 0.10, 0.4, 0.38])

    for irow in np.arange(len(spec)):
        irow2 = irow + oshift

        shift = order_shift_lst[irow]

        if irow2 < 0 or irow2 >= len(ref_spec):
            continue

        ref_wave = ref_spec[irow2]['wavelength']
        ref_flux  = ref_spec[irow2]['flux']
        ref_pixel = np.arange(ref_wave.size)

        # construct an interpolate function that converts wavelength to pixel
        # before constructing the function, the reference wavelength should be
        # increased
        if ref_wave[0] > ref_wave[-1]:
            ref_wave = ref_wave[::-1]
            ref_flux = ref_flux[::-1]
            ref_pixel = ref_pixel[::-1]

        # constrcut the interpolate function
        f_wave_to_pix = intp.InterpolatedUnivariateSpline(
                ref_wave, ref_pixel, k=3)

        order = ref_spec[irow2]['order']
        sublinelist = linelist[linelist['order']==order]
        sublinelist.add_column([-1]*len(sublinelist), index=-1, name='pixel')
        sublinelist.add_column([-1]*len(sublinelist), index=-1, name='i1')
        sublinelist.add_column([-1]*len(sublinelist), index=-1, name='i2')

        n = spec[irow]['flux'].size
        pixel_lst = np.arange(n)

        hwin = int(window/2)
        
        for iline, line in enumerate(sublinelist):
            pix1 = f_wave_to_pix(line['wavelength']) + shift
            cint1 = int(round(pix1))
            i1, i2 = cint1 - hwin, cint1 + hwin+1
            i1 = max(i1, 0)
            i2 = min(i2, n-1)
            sublinelist[iline]['pixel'] = cint1
            sublinelist[iline]['i1'] = i1
            sublinelist[iline]['i2'] = i2

        m = (sublinelist['pixel']>0)*(sublinelist['pixel']<n-1)
        sublinelist = sublinelist[m]

        # resort
        sublinelist.sort('pixel')

        wave_lst, species_lst = [], []
        # fitting parameter results
        alpha_lst, beta_lst, center_lst = [], [], []
        A_lst, bkg_lst, std_lst, fwhm_lst, q_lst = [], [], [], [], []

        for iline, line in enumerate(sublinelist):
            i1 = line['i1']
            i2 = line['i2']
            ic = line['pixel']
            xdata = np.arange(i1, i2)
            ydata = spec[irow]['flux'][i1:i2]
            # add background level
            p0 = [ydata.min()]
            p0.append(spec[irow]['flux'][ic]-ydata.min())
            p0.append(3.6)
            p0.append(3.5)
            p0.append(line['pixel'])

            lower_bounds = [-np.inf, 0,      0.5, 0.1, i1]
            upper_bounds = [np.inf,  np.inf, 20,   20, i2]

            fitres = opt.least_squares(errfunc, p0,
                        bounds=(lower_bounds, upper_bounds),
                        args=(xdata, ydata, fitline),
            )

            param = fitres['x']

            # now run a second round by adjusting the center
            ic = param[4]
            cint1 = int(round(ic))
            i1, i2 = cint1 - hwin, cint1 + hwin+1
            i1 = max(i1, 0)
            i2 = min(i2, n-1)
            xdata = np.arange(i1, i2)
            ydata = spec[irow]['flux'][i1:i2]
            # add background level
            p0 = [ydata.min()]
            p0.append(spec[irow]['flux'][cint1]-ydata.min())
            p0.append(param[2])
            p0.append(param[3])
            p0.append(ic)
            # set lower and upper bounds
            lower_bounds = [-np.inf, 0,      0.5, 0.1, i1]
            upper_bounds = [np.inf,  np.inf, 20,   20, i2]

            # fit
            fitres = opt.least_squares(errfunc, p0,
                        bounds=(lower_bounds, upper_bounds),
                        args=(xdata, ydata, fitline),
            )
            param = fitres['x']

            fmt_str = (' - {:2s} {:3s} {:10.4f}: {:4d}-{:4d} '
                       'alpha={:6.3f} beta={:6.3f} center={:8.3f} fwhm={:6.2f} '
                       'q={:6.2f}')

            A, alpha, beta, center = param[1:5]
            fwhm = 2*alpha*np.power(np.log(2), 1/beta)
            std = np.sqrt(fitres['cost']*2/xdata.size)
            q = A/std

            wave_lst.append(line['wavelength'])
            species_lst.append('{} {}'.format(line['element'], line['ion']))
            # append fitting results
            center_lst.append(center)
            A_lst.append(A)
            alpha_lst.append(alpha)
            beta_lst.append(beta)
            fwhm_lst.append(fwhm)
            bkg_lst.append(param[0])
            std_lst.append(std)
            q_lst.append(q)

            print(fmt_str.format(line['element'], line['ion'], line['wavelength'],
                    i1, i2, alpha, beta, center, fwhm, q, count_line))

            sublinelist['i1'][iline] = i1
            sublinelist['i2'][iline] = i2

            # plot line-by-line
            # create figure
            if count_line%(nrow*ncol)==0:
                fig_lbl = plt.figure(figsize=(16, 9), dpi=150,
                                     tight_layout=True)
                fig_lbl_lst.append(fig_lbl)
            # create small axes
            axs = fig_lbl.add_subplot(nrow, ncol, count_line%(nrow*ncol)+1)
            axs.plot(xdata, ydata*1e-3, 'o', c='C0', ms=4, alpha=0.7)
            # plot fitted curve
            newx = np.linspace(xdata[0], xdata[-1], 100)
            newy = fitline(param, newx)
            axs.plot(newx, newy*1e-3, ls='-', color='C1', lw=0.7, alpha=0.7)
            
            # draw text and plot line center as vertical lines
            axs.axvline(x=center, color='k', ls='--', lw=0.7)
            text = 'Order {}, {} {} {:9.4f}'.format(order,
                    line['element'], line['ion'], line['wavelength'])
            
            axs.set_xlim(newx[0], newx[-1])
            _x1, _x2 = axs.get_xlim()
            _y1, _y2 = axs.get_ylim()
            _y2 = _y2 + 0.2*(_y2 - _y1)
            axs.text(0.95*_x1+0.05*_x2, 0.05*_y1+0.95*_y2, text,
                     ha='left', va='top', fontsize=8)
            axs.set_ylim(_y1, _y2)
            if _x2 -_x1 < 25:
                major_tick, minor_tick = 5, 1
            else:
                major_tick, minor_tick = 10, 1
            axs.xaxis.set_major_locator(tck.MultipleLocator(major_tick))
            axs.xaxis.set_minor_locator(tck.MultipleLocator(minor_tick))

            for tick in axs.xaxis.get_major_ticks():
                tick.label1.set_fontsize(7)
            for tick in axs.yaxis.get_major_ticks():
                tick.label1.set_fontsize(7)

            count_line += 1

        ###################################################
        # fit wavelength solution
        # resort the pixel list and wavelength list
        wave_lst = np.array(wave_lst)
        center_lst = np.array(center_lst)
        A_lst = np.array(A_lst)
        std_lst = np.array(std_lst)
        q_lst = np.array(q_lst)

        # resort the center list
        idx = center_lst.argsort()
        center_lst = center_lst[idx]
        wave_lst = wave_lst[idx]
    
        # begin wavelength solution fit
        mask = q_lst > q_threshold
        if mask.sum() >= xdeg+1:
            # fit the wavelength of this order
            for i in range(10):
                coeff_wave = np.polyfit(center_lst[mask], wave_lst[mask],
                                        deg=xdeg)
                fitwave = np.polyval(coeff_wave, center_lst)
                reswave = wave_lst - fitwave
                stdwave = reswave[mask].std()
                newmask = (np.abs(reswave) < clipping*stdwave)*(q_lst>q_threshold)
                if newmask.sum()==mask.sum():
                    break
                mask = newmask
                if mask.sum() < xdeg+1:
                    break
            # get wavelength of each pixel in this order
            wave = np.polyval(coeff_wave, pixel_lst)
            allwave[irow] = (order, wave)
            has_wave = True
            for res in reswave[mask]:
                all_res_lst.append(res)
        else:
            wave = np.zeros_like(pixel_lst, dtype=np.float64)
            has_wave = False

        # plot wavelength solutions and all the emission lines
        color = 'C{}'.format(order%10)

        # plot the emission lines used in the fitting
        if mask.sum()>0:
            # plot wavelength solution of each order
            axsol.plot(center_lst[mask], 1/wave_lst[mask], 'o',
                       c=color, ms=4, alpha=0.8, mew=0)
            # plot residuals only when there is a fitting
            if has_wave:
                axresx.plot(center_lst[mask], reswave[mask], 'o',
                            c=color, ms=4, alpha=0.8, mew=0, lw=0.6)
                axresy.plot(np.repeat(order, mask.sum()), reswave[mask], 'o',
                            c=color, ms=4, alpha=0.8, mew=0, lw=0.6)
        # plot the emission lines rejectted
        if (~mask).sum() > 0:
            axsol.plot(center_lst[~mask], 1/wave_lst[~mask], 'o',
                       c='none', mec=color, ms=3, mew=0.5)
            # plot residuals only when there is a fitting
            if has_wave:
                axresx.plot(center_lst[~mask], reswave[~mask], 'o',
                            c='none', mec=color, ms=3, alpha=0.8, mew=0.5)
                axresy.plot(np.repeat(order, (~mask).sum()), reswave[~mask], 'o',
                            c='none', mec=color, ms=3, alpha=0.8, mew=0.5)
        # plot the wavelength solution

        if (wave > 0).sum()>0:
            axsol.plot(pixel_lst, 1/wave, '-', c=color, lw=0.5)


    all_res_lst = np.array(all_res_lst)
    allstd = all_res_lst.std()

    nused = all_res_lst.size

    # interpolate wavelength

    axresx.axhline(0, ls='--', color='k', lw=0.5)
    axresx.axhline(allstd, ls='--', color='k', lw=0.5)
    axresx.axhline(-allstd, ls='--', color='k', lw=0.5)
    axresy.axhline(0, ls='--', color='k', lw=0.5)
    axresy.axhline(allstd, ls='--', color='k', lw=0.5)
    axresy.axhline(-allstd, ls='--', color='k', lw=0.5)

    # adjust axsol
    _y1, _y2 = axsol.get_ylim()
    axsol.set_ylim(_y2, _y1)
    yticks, ytick_labels = [],[]
    for _w in range(1000, 20000, 1000):
        if _y1 < 1/_w < _y2:
            yticks.append(1/_w)
            ytick_labels.append(_w)
    axsol.set_yticks(yticks)
    axsol.set_yticklabels(ytick_labels)
    axsol.set_xlabel('Pixel')
    axsol.set_ylabel(u'Wavelength (\xc5)')
    axsol.set_xlim(0, n-1)

    axresx.set_xlabel('Pixel')
    axresy.set_xlabel('Order')
    axresx.set_ylabel(u'\u0394\u03bb (\xc5)')
    axresy.set_ylabel(u'\u0394\u03bb (\xc5)')

    axresx.set_ylim(-5*allstd, 5*allstd)
    axresy.set_ylim(-5*allstd, 5*allstd)
    axresx.set_xlim(0, n-1)

    # add text in residual X axes
    _x1, _x2 = axresx.get_xlim()
    _y1, _y2 = axresx.get_ylim()
    axresx.text(0.95*_x1+0.05*_x2, 0.1*_y1+0.9*_y2,
                u'PolyOrder = {}'.format(xdeg))
    axresx.set_xlim(_x1, _x2)
    axresx.set_ylim(_y1, _y2)


    # add text in residual Y axes
    _x1, _x2 = axresy.get_xlim()
    _y1, _y2 = axresy.get_ylim()
    axresy.text(0.95*_x1+0.05*_x2, 0.1*_y1+0.9*_y2,
                u'N (used) = {}, R.M.S. = {:.4f} \xc5'.format(nused, allstd))
    axresy.set_xlim(_x1, _x2)
    axresy.set_ylim(_y1, _y2)

    return {'wavelength': allwave,
            'linelist': linelist,
            'std': allstd,
            'norders': len(allwave),    # number of orders that has wavelengths
            'nlines': nused,
            'fig_solution': fig_sol,
            'fig_fitlbl': fig_lbl_lst,
            'fig_ccf': fig_ccf,
            }
