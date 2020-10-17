#!/usr/bin/env python3

import time
import numpy as np
import matplotlib.pyplot as plt


# constants
ev_a3_2_GPa=1.602176462e2
Ry2eV=13.605698066
ang32au=6.74833449394997 # Ang^3 --> a.u.
v_ratio=1.00
ntv=400


try:
    from scipy.optimize import curve_fit

except ImportError:
    try:
        from scipy.optimize import leastsq

        def _general_function(params, xdata, ydata, function):
            return function(xdata, *params) - ydata

        def curve_fit(f, x, y, p0):
            func = _general_function
            args = (x, y, f)
            popt, pcov, infodict, mesg, ier = leastsq(func, p0, args=args,
                                                      full_output=1)
            if ier not in [1, 2, 3, 4]:
                raise RuntimeError("Optimal parameters not found: " + mesg)
            return popt, pcov

    except ImportError:
        curve_fit = None


eos_names = ['sj', 'taylor', 'murnaghan', 'birch', 'birchmurnaghan',
             'pouriertarantola', 'vinet', 'antonschmidt', 'p3']


def parabola(x, a, b, c):
    """
    parabola polynomial function
    This function is used to fit the data to get good guesses for
    the equation of state fits.
    4th order polynomial fit to get good guesses is not a good idea, 
    because for noisy data the fit is too wiggly.
    2nd order seems to be sufficient, and guarantees a single minimum.
    """
    return a + b * x + c * x**2


def murnaghan(V, E0, B0, BP, V0):
    """
    From PRB 28,5480 (1983)
    """
    E = E0 + B0*V/BP*(((V0/V)**BP)/(BP-1)+1)-V0*B0/(BP-1)
    return E


def birchmurnaghan(V, E0, B0, BP, V0):
    """
    BirchMurnaghan equation from PRB 70, 224107
    Eq. (3) in the paper. Note that there's a typo in the paper and it uses
    inversed expression for eta.
    """
    eta = (V0 / V)**(1 / 3)
    E = E0 + 9 * B0 * V0 / 16 * (eta**2 - 1)**2 * (
        6 + BP * (eta**2 - 1) - 4 * eta**2)
    return E


def check_birchmurnaghan():
    from sympy import symbols, Rational, diff, simplify
    v, b, bp, v0 = symbols('v b bp v0')
    x = (v0 / v)**Rational(2, 3)
    e = 9 * b * v0 * (x - 1)**2 * (6 + bp * (x - 1) - 4 * x) / 16
    print(e)
    B = diff(e, v, 2) * v
    BP = -v * diff(B, v) / b
    print(simplify(B.subs(v, v0)))
    print(simplify(BP.subs(v, v0)))


def pouriertarantola(V, E0, B0, BP, V0):
    """
    Pourier-Tarantola equation from PRB 70, 224107
    """
    eta = (V / V0)**(1 / 3)
    squiggle = -3 * np.log(eta)
    E = E0 + B0 * V0 * squiggle**2 / 6 * (3 + squiggle * (BP - 2))
    return E


def vinet(V, E0, B0, BP, V0):
    """
    Vinet equation from PRB 70, 224107
    """
    eta = (V / V0)**(1 / 3)
    E = (E0 + 2 * B0 * V0 / (BP - 1)**2 *
         (2 - (5 + 3 * BP * (eta - 1) - 3 * eta) *
          np.exp(-3 * (BP - 1) * (eta - 1) / 2)))
    return E


def birch(V, E0, B0, BP, V0):
    """
    From Intermetallic compounds: Principles and Practice, Vol. I: Principles
    Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
    case where n=0
    """
    E = (E0 +
         9 / 8 * B0 * V0 * ((V0 / V)**(2 / 3) - 1)**2 +
         9 / 16 * B0 * V0 * (BP - 4) * ((V0 / V)**(2 / 3) - 1)**3)
    return E


def taylor(V, E0, beta, alpha, V0):
    """
    Taylor Expansion up to 3rd order about V0
    """
    E = E0 + beta / 2 * (V - V0)**2 / V0 + alpha / 6 * (V - V0)**3 / V0
    return E


def antonschmidt(V, Einf, B, n, V0):
    """
    From Intermetallics 11, 23-32 (2003)
    Einf should be E_infinity, i.e. infinite separation, but
    according to the paper it does not provide a good estimate
    of the cohesive energy. They derive this equation from an
    empirical formula for the volume dependence of pressure,
    E(vol) = E_inf + int(P dV) from V=vol to V=infinity
    But the equation breaks down at large volumes, so E_inf
    is not that meaningful.
    n should be about -2 according to the paper.
    I find this equation does not fit volumetric data as well
    as the other equations do.
    """
    E = B * V0 / (n + 1) * (V / V0)**(n + 1) * (np.log(V / V0) -
                                                (1 / (n + 1))) + Einf
    return E


def p3(V, c0, c1, c2, c3):
    """
    polynomial fit
    """
    E = c0 + c1 * V + c2 * V**2 + c3 * V**3
    return E



class EquationOfState:
    """
    Fit equation of state for bulk systems.
    The following equation is used,

        murnaghan
            PRB 28, 5480 (1983)

        birch
            Intermetallic compounds: Principles and Practice,
            Vol I: Principles. pages 195-210

        birchmurnaghan (default)
            PRB 70, 224107

        pouriertarantola
            PRB 70, 224107

        vinet
            PRB 70, 224107

        p3
            A third order polynomial fit
            
        taylor
            A third order Taylor series expansion about the minimum volume            
        
        sjeos 
            A third order inverse polynomial fit 10.1103/PhysRevB.67.026103

                                    2      3        -1/3
                E(V) = c + c t + c t  + c t ,  t = V
                        0   1     2      3

    Usage:

        eos = EquationOfState(volumes, energies, eos='murnaghan')
        v0, e0, B, Bp = eos.fit()
        eos.plot()
    """
    def __init__(self, volumes, energies, eos='birchmurnaghan'):
        self.v = np.array(volumes)
        self.e = np.array(energies)
        self.eos_string = eos
        self.v0 = None


    def fit(self):
        """
        Calculate volume, energy, and bulk modulus.
        Returns the optimal volume, the minimum energy, and the bulk
        modulus. Notice that the ASE units for the bulk modulus is
        eV/Angstrom^3. To get the value in GPa, do this,
            v0, e0, B = eos.fit()
            print(B / kJ * 1.0e24, 'GPa')
        """

        if self.eos_string == 'sjeos':
            return self.fit_sjeos()
        elif self.eos_string == 'p3':
            return self.fit_p3()

        self.func = globals()[self.eos_string]
        p0 = [min(self.e), 1, 1]
        popt, pcov = curve_fit(parabola, self.v, self.e, p0)
        parabola_parameters = popt
        
        # Here I just make sure the minimum is bracketed by the volumes
        # this if for the solver
        minvol = min(self.v)
        maxvol = max(self.v)
        # the minimum of the parabola is at dE/dV = 0, or 2 * c V +b =0
        c = parabola_parameters[2]
        b = parabola_parameters[1]
        a = parabola_parameters[0]
        parabola_vmin = -b / 2 / c

        if not (minvol < parabola_vmin and parabola_vmin < maxvol):
            print('Warning the minimum volume of a fitted parabola is not in '
                  'your volumes. You may not have a minimum in your dataset')

        # evaluate the parabola at the minimum to estimate the groundstate
        # energy
        E0 = parabola(parabola_vmin, a, b, c)
        # estimate the bulk modulus from Vo * E''.  E'' = 2 * c
        B0 = 2 * c * parabola_vmin

        BP = 4 # normally, Bp=4 os a good guess for most minerals

        initial_guess = [E0, B0, BP, parabola_vmin]

        # now fit the equation of state
        p0 = initial_guess
        popt, pcov = curve_fit(self.func, self.v, self.e, p0)

        self.eos_parameters = popt

        self.v0 = self.eos_parameters[3]
        self.e0 = self.eos_parameters[0]
        self.B = self.eos_parameters[1]
        self.Bp=self.eos_parameters[2]

        return self.v0, self.e0, self.B*ev_a3_2_GPa,self.Bp
  

    def fit_sjeos(self):
        """
        Calculate volume, energy, and bulk modulus.
        Returns the optimal volume, the minimum energy, and the bulk
        modulus.  Notice that the units for the bulk modulus is
        eV/Angstrom^3. To get the value in GPa, do this,
            v0, e0, B = eos.fit()
            print(B*ev_a3_2_GPa, 'GPa')
        """

        fit0 = np.poly1d(np.polyfit(self.v**-(1 / 3), self.e, 3))
        fit1 = np.polyder(fit0, 1)
        fit2 = np.polyder(fit1, 1)
        fit3 = np.polyder(fit2, 1)

        self.v0 = None
        for t in np.roots(fit1):
            if isinstance(t, float) and t > 0 and fit2(t) > 0:
                self.v0 = t**-3
                break

        if self.v0 is None:
            raise ValueError('No minimum could be found!')

        self.e0 = fit0(t)
        self.B = (t**5)*fit2(t)/9+fit1(t)/12*t**2
        # fit1(t)==0 here, cause at this t, P=0
        self.Bp=(5/9*fit2(t)*t**4 + 1/9*fit3(t)*t**5)/(1/3*fit2(t)*t**4)
        self.fit0 = fit0

        return self.v0, self.e0, self.B*ev_a3_2_GPa,self.Bp


    def fit_p3(self):
        """
        The energy fitted this way is okay, but Pressure is bad
        """
        self.func = globals()[self.eos_string]
        p0 = [min(self.e), 1, 1]
        popt, pcov = curve_fit(parabola, self.v, self.e, p0)
        parabola_parameters = popt
        
        # Here I just make sure the minimum is bracketed by the volumes
        # this if for the solver
        minvol = min(self.v)
        maxvol = max(self.v)
        vgrid=np.linspace(minvol,maxvol,ntv)
        # the minimum of the parabola is at dE/dV = 0, or 2 * c V +b =0
        c = parabola_parameters[2]
        b = parabola_parameters[1]
        a = parabola_parameters[0]
        parabola_vmin = -b / 2 / c

        if not (minvol < parabola_vmin and parabola_vmin < maxvol):
            print('Warning the minimum volume of a fitted parabola is not in '
                  'your volumes. You may not have a minimum in your dataset')

        # evaluate the parabola at the minimum to estimate the groundstate
        # energy
        E0 = parabola(parabola_vmin, a, b, c)
        # estimate the bulk modulus from Vo * E''.  E'' = 2 * c
        B0 = 2 * c * parabola_vmin

        BP = 4 # normally, 4 is a good guess for Bp for most minerals

        initial_guess = [E0, B0, BP, parabola_vmin]

        # now fit the equation of state
        p0 = initial_guess
        popt, pcov = curve_fit(self.func, self.v, self.e, p0)

        self.eos_parameters = popt        
        c0, c1, c2, c3 = self.eos_parameters
        # find minimum E in E = c0 + c1 * V + c2 * V**2 + c3 * V**3
        # dE/dV = c1+ 2 * c2 * V + 3 * c3 * V**2 = 0
        # solve by quadratic formula with the positive root
        a = 3 * c3
        b = 2 * c2
        c = c1
        self.v0 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        self.e0 = p3(self.v0, c0, c1, c2, c3)
        self.B = (2 * c2 + 6 * c3 * self.v0) * self.v0
        self.Bp=-1.0*(2*c2+12*c3*self.v0)/(2*c2+6*c3*self.v0)
        self.En_V=p3(vgrid, c0, c1, c2, c3)
        self.P_V=-(c1+2*c2*vgrid+3*c3*vgrid**2)
        self.vgrid=vgrid
        return self.v0, self.e0, self.B*ev_a3_2_GPa,self.Bp,self.En_V,self.P_V*ev_a3_2_GPa, self.vgrid


#   EnPB_vgrid
    def EnPB_V(self):
        """
        Calculate En(V), P(V),B(V) using calculated V0, E0, B0 and Bp
        """
        minvol = min(self.v)/v_ratio
        maxvol = max(self.v)*v_ratio
        vgrid=np.linspace(minvol,maxvol,ntv)

        if self.eos_string == 'birchmurnaghan':
            self.En_V=birchmurnaghan(vgrid, self.e0, self.B, self.Bp, self.v0)
            eta = (self.v0/vgrid)**(1/3)
            self.P_V=3/2*self.B*(eta**7-eta**5)*(1+3/4*(self.Bp-4)*(eta**2-1))
                
        elif self.eos_string == 'murnaghan':
            self.En_V=murnaghan(vgrid, self.e0, self.B, self.Bp, self.v0)
            self.P_V=self.B/self.Bp*((self.v0/vgrid)**(self.Bp) - 1)
         
        elif self.eos_string =='pouriertarantola':
            self.En_V=pouriertarantola(vgrid, self.e0, self.B, self.Bp, self.v0)
            eta = (self.v0/vgrid)
            squiggle =np.log(eta)
            self.P_V=self.B*eta*(squiggle+((self.Bp-2)/2)*(squiggle**2))
            
        elif self.eos_string == 'vinet':
            self.En_V=vinet(vgrid, self.e0, self.B, self.Bp, self.v0)
            eta = (vgrid/self.v0)**(1/3)
            self.P_V=3*self.B*(eta**(-2))*(1-eta)*np.exp(-3/2*(self.Bp-1)*(eta-1))
               
        else:
            print('ERROR, eos="birchmurnaghan, murnaghan, pouriertarantola, vinet"')
            
        self.vgrid=vgrid
        return self.En_V, self.P_V*ev_a3_2_GPa,self.vgrid


#   additional Input and Output functions
def readev(evfile):
    """
    read Energy-Volume data from evfile
    """
    En=[]
    Vol=[]
    with open(evfile,'r') as f:
        for line in f:
            tmp=line.strip().split()
            Vol.append(float(tmp[0]))
            En.append(float(tmp[1]))
    return np.array(Vol), np.array(En)


def write_to_file(filename,text):
    with open(filename, "a") as f:
        f.write(text)
    return


def plot_set():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    fontweight='normal' # normal, bold, bolder, lighter

    plt.rcParams['font.size']=SMALL_SIZE
    plt.rcParams['font.weight']=fontweight
    plt.rcParams['axes.titlesize']=MEDIUM_SIZE # fontsize of the axes title
    plt.rcParams['axes.labelsize']=MEDIUM_SIZE # fontsize of the x and y labels
    plt.rcParams['legend.fontsize']=SMALL_SIZE # legend fontsize
    plt.rcParams['figure.titlesize']=BIGGER_SIZE # fontsize of the figure title
    
    plt.rcParams['lines.linewidth']=2
    
    plt.rcParams['xtick.labelsize']=SMALL_SIZE # fontsize of the tick labels
    plt.rcParams['ytick.labelsize']=SMALL_SIZE # fontsize of the tick labels
    plt.rcParams['xtick.direction']='in' # in, out, or inout
    plt.rcParams['ytick.direction']='in' # in, out, or inout
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['ytick.minor.visible']=True
    
    plt.rcParams['axes.grid']=True # True
    plt.rcParams['grid.alpha']=0.5
    plt.rcParams['grid.color']='b'
    
    plt.rcParams['figure.figsize']=[10,8] # figure size in inches

    plt.rcParams['figure.dpi']=300


def eosplot(V,E,V0,E0,B0,Bp,Vgrid, E_fit,P_fit,label_fit):
    plot_set()
    plt.figure(figsize=(10,6))

    ax1=plt.subplot2grid((1,2),(0,0)) 
    #(shape, loc, rowspan=1, colspan=1, **kwargs)
    
    ax1.grid(True)
    ax1.plot(Vgrid,E_fit-E0, 'r-', lw=1.5, label=label_fit)
    ax1.plot(V,E-E0, 'go', ms=6, label='CALC.')
    ax1.plot(V0,0,'bo',markersize=4)
    
    dx=(max(V)-min(V))/15
    dy=(max(E)-min(E))/15
    ax1.set_xlim((min(V)-dx, max(V)+dx))
    ax1.set_ylim((min(E-E0)-dy, max(E-E0)+dy))
    ax1.set_xlabel(r'Volume ($\AA^3$)')
    ax1.set_ylabel(r'$E-E_0$ (eV)')
    ax1.grid(False)
    ax1.legend(loc='upper right')
    
    ax2=plt.subplot2grid((1,2),(0,1)) 
    #(shape, loc, rowspan=1, colspan=1, **kwargs)
    
    ax2.grid(True)
    ax2.plot(Vgrid,P_fit, 'r-', lw=1.5, label=label_fit)
    #ax2.plot(V,P, 'go', ms=6, label='CALC.')
    ax2.plot(V0,0,'bo',markersize=4)
    
    dx=(max(V)-min(V))/15
    dy=(max(P_fit)-min(P_fit))/15
    ax2.set_xlim((min(V)-dx, max(V)+dx))
    ax2.set_ylim((min(P_fit)-dy, max(P_fit)+dy))
    ax2.set_xlabel(r'Volume ($\AA^3$)')
    ax2.set_ylabel(r'Pressure (GPa)')
    ax2.grid(False)
    ax2.legend(loc='upper right')
    
    #text = '$V_{min}$=%.4f $\AA^3$ (%.4f $a.u.^3$) ; $E_{min}$=%.6f eV (%.6f Ry) ; $B_{0}$=%.2f GPa ; $B^{,}$=%.2f ;'%(V0,V0*ang32au,E0,E0/Ry2eV,B0,Bp)
    #props = dict(boxstyle='round', facecolor='red', alpha=0.3)
    #ax1.text(-0.10, 1.10, text, transform=ax1.transAxes, fontsize=10, verticalalignment='top')#, bbox=props
    text = '$E_{min}$ = %.6f eV\n' %E0 +\
           '         ( %.6f Ry)\n' %(E0/Ry2eV)+\
           ' \n'+\
           '$V_{min}$ = %.4f $\AA^3$\n' %V0 +\
           '         ( %.4f $a.u.^3$)\n'%(V0*ang32au) +\
           ' \n'+\
           '$B_{0}$    = %.2f GPa\n' %B0 +\
           ' \n'+\
           '$B_{0}^{\'}$    = %.2f' %Bp
           
    ax1.text(0.45, 0.80, text, transform=ax1.transAxes, fontsize=10, verticalalignment='top')#, bbox=props

    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0.25)
    
    pltfn_png='eos_'+label_fit+'.png'
    pltfn_pdf='eos_'+label_fit+'.pdf'
    plt.savefig(pltfn_png, format='png', dpi=200,bbox_inches='tight') 
    # plt.savefig(pltfn_pdf, format='pdf',bbox_inches='tight') 


#def eosplot_4eos(V,E,V0,E0,Vgrid, E_m,P_m,E_bm,P_bm,E_log,P_log,E_v,P_v):
def eosplot_4eos(V,E,V0,E0,Vgrid, E_fit,P_fit,label_fit):
    
    plot_set()
    plt.figure(figsize=(10,6))

    ax1=plt.subplot2grid((1,2),(0,0)) 
    #(shape, loc, rowspan=1, colspan=1, **kwargs)
    
    ax1.grid(True)
    for y, label in zip((E_fit-E0),label_fit):
        ax1.plot(Vgrid,y, lw=1.5,label=label)
    ax1.plot(V,E-E0, 'go', ms=6, label='CALC.')
    ax1.plot(V0,0,'bo',markersize=4)
    
    dx=(max(V)-min(V))/15
    dy=(max(E)-min(E))/15
    ax1.set_xlim((min(V)-dx, max(V)+dx))
    ax1.set_ylim((min(E-E0)-dy, max(E-E0)+dy))
    ax1.set_xlabel(r'Volume ($\AA^3$)')
    ax1.set_ylabel(r'$E-E_0$ (eV)')
    ax1.grid(False)
    ax1.legend(loc='upper right')
    
    ax2=plt.subplot2grid((1,2),(0,1)) 
    #(shape, loc, rowspan=1, colspan=1, **kwargs)
    
    ax2.grid(True)
    for y, label in zip(P_fit,label_fit):
        ax2.plot(Vgrid,y, lw=1.5,label=label)
    #ax2.plot(V,P, 'go', ms=6, label='CALC.')
    ax2.plot(V0,0,'bo',markersize=4)
    
    dx=(max(V)-min(V))/15
    dy=(max(P_fit[0])-min(P_fit[0]))/15
    ax2.set_xlim((min(V)-dx, max(V)+dx))
    ax2.set_ylim((min(P_fit[0])-dy, max(P_fit[0])+dy))
    ax2.set_xlabel(r'Volume ($\AA^3$)')
    ax2.set_ylabel(r'Pressure (GPa)')
    ax2.grid(False)
    ax2.legend(loc='upper right')
    
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0.25)
    
    pltfn_png='eos_'+'all'+'.png'
    pltfn_pdf='eos_'+'all'+'.pdf'
    # plt.savefig(pltfn_png, format='png', dpi=200,bbox_inches='tight') 
    # plt.savefig(pltfn_pdf, format='pdf',bbox_inches='tight') 
    

def saveEOSfitdata(V,E):
    
    eos_m=EquationOfState(V,E,eos='murnaghan')
    V0_m,E0_m,B0_m,Bp_m=eos_m.fit()
    En_V_m, P_V_m, Vgrid_m=eos_m.EnPB_V()
    
    eos_bm=EquationOfState(V,E,eos='birchmurnaghan')#
    V0_bm,E0_bm,B0_bm,Bp_bm=eos_bm.fit()
    En_V_bm, P_V_bm, Vgrid_bm=eos_bm.EnPB_V()
    #eos.check_birchmurnaghan()
    
    eos_v=EquationOfState(V,E,eos='vinet')
    V0_v,E0_v,B0_v,Bp_v=eos_v.fit()
    En_V_v, P_V_v, Vgrid_v=eos_v.EnPB_V()

    eos_log=EquationOfState(V,E,eos='pouriertarantola')
    V0_log,E0_log,B0_log,Bp_log=eos_log.fit()
    En_V_log,P_V_log, Vgrid_log=eos_log.EnPB_V()     
    
    eos_p3=EquationOfState(V,E,eos='p3')
    V0_p3,E0_p3,B0_p3,Bp_p3,En_V_p3,P_V_p3, Vgrid_p3=eos_p3.fit()  
    
#    eos_b=eos.EquationOfState(V,E,eos='birch')
#    V0_b,E0_b,B0_b,Bp_b=eos_b.fit()   
#    
#
#    eos_sj=eos.EquationOfState(V,E,eos='sjeos')
#    V0_sj,E0_sj,B0_sj,Bp_sj=eos_sj.fit()
    eos_file="eos_fitted_data.txt"
    eos_title="EQUATION OF STATE              VOL(A^3)     E(eV)           BM(GPa)   BM PRIME\n"
    dash_line="-------------------------------------------------------------------------------\n"
    double_line="===============================================================================\n"
    ieos_m=  "MURNAGHAN 1944               "+\
            str('{:10.4f}'.format(V0_m)) +\
            str('{:19.8f}'.format(E0_m)) +\
            str('{:8.2f}'.format(B0_m)) +\
            str('{:8.2f}'.format(Bp_m)) +"\n"
    ieos_bm= "BIRCH-MURNAGHAN 3rd 1947     "+\
            str('{:10.4f}'.format(V0_bm)) +\
            str('{:19.8f}'.format(E0_bm)) +\
            str('{:8.2f}'.format(B0_bm)) +\
            str('{:8.2f}'.format(Bp_bm)) +"\n"    
    ieos_log="POIRIER-TARANTOLA 1998       "+\
            str('{:10.4f}'.format(V0_log)) +\
            str('{:19.8f}'.format(E0_log)) +\
            str('{:8.2f}'.format(B0_log)) +\
            str('{:8.2f}'.format(Bp_log)) +"\n"
    ieos_v=  "VINET 1987                   "+\
            str('{:10.4f}'.format(V0_v)) +\
            str('{:19.8f}'.format(E0_v)) +\
            str('{:8.2f}'.format(B0_v)) +\
            str('{:8.2f}'.format(Bp_v)) +"\n"
    ieos_p3=  "THIRD ORDER POLYNOMIAL       "+\
            str('{:10.4f}'.format(V0_p3)) +\
            str('{:19.8f}'.format(E0_p3)) +\
            str('{:8.2f}'.format(B0_p3)) +\
            str('{:8.2f}'.format(Bp_p3)) +"\n"
            
    write_to_file(eos_file,double_line)        
    write_to_file(eos_file,eos_title)
    write_to_file(eos_file,double_line)
    write_to_file(eos_file,ieos_m)
    write_to_file(eos_file,ieos_bm)
    write_to_file(eos_file,ieos_log)
    write_to_file(eos_file,ieos_v)
    write_to_file(eos_file,dash_line)
    write_to_file(eos_file,ieos_p3)
    write_to_file(eos_file,double_line)  
    # Change the unit and save data again
    eos_title="EQUATION OF STATE              VOL(au^3)     E(Ry)           BM(GPa)   BM PRIME\n"

    write_to_file(eos_file,"\n\n")
    ieos_m=  "MURNAGHAN 1944               "+\
            str('{:10.4f}'.format(V0_m*ang32au)) +\
            str('{:19.8f}'.format(E0_m/Ry2eV)) +\
            str('{:8.2f}'.format(B0_m)) +\
            str('{:8.2f}'.format(Bp_m)) +"\n"
    ieos_bm= "BIRCH-MURNAGHAN 3rd 1947     "+\
            str('{:10.4f}'.format(V0_bm*ang32au)) +\
            str('{:19.8f}'.format(E0_bm/Ry2eV)) +\
            str('{:8.2f}'.format(B0_bm)) +\
            str('{:8.2f}'.format(Bp_bm)) +"\n"    
    ieos_log="POIRIER-TARANTOLA 1998       "+\
            str('{:10.4f}'.format(V0_log*ang32au)) +\
            str('{:19.8f}'.format(E0_log/Ry2eV)) +\
            str('{:8.2f}'.format(B0_log)) +\
            str('{:8.2f}'.format(Bp_log)) +"\n"
    ieos_v=  "VINET 1987                   "+\
            str('{:10.4f}'.format(V0_v*ang32au)) +\
            str('{:19.8f}'.format(E0_v/Ry2eV)) +\
            str('{:8.2f}'.format(B0_v)) +\
            str('{:8.2f}'.format(Bp_v)) +"\n"
    ieos_p3=  "THIRD ORDER POLYNOMIAL       "+\
            str('{:10.4f}'.format(V0_p3*ang32au)) +\
            str('{:19.8f}'.format(E0_p3/Ry2eV)) +\
            str('{:8.2f}'.format(B0_p3)) +\
            str('{:8.2f}'.format(Bp_p3)) +"\n"
            
    write_to_file(eos_file,double_line)        
    write_to_file(eos_file,eos_title)
    write_to_file(eos_file,double_line)
    write_to_file(eos_file,ieos_m)
    write_to_file(eos_file,ieos_bm)
    write_to_file(eos_file,ieos_log)
    write_to_file(eos_file,ieos_v)
    write_to_file(eos_file,dash_line)
    write_to_file(eos_file,ieos_p3)
    write_to_file(eos_file,double_line) 
    
    E_fit=np.array([En_V_m,En_V_bm,En_V_v,En_V_log])
    P_fit=np.array([P_V_m,P_V_bm,P_V_v,P_V_log])
    label_fit=np.array(['MURNAGHAN','BIRCH-MURNAGHAN','VINET','POIRIER-TARANTOLA'])
    eosplot_4eos(V,E,V0_m,E0_m,Vgrid_m, E_fit,P_fit,label_fit)
    
    eosplot(V,E,V0_m,E0_m,B0_m,Bp_m,Vgrid_m,En_V_m,P_V_m, 'MURNAGHAN')
    eosplot(V,E,V0_bm,E0_bm,B0_bm,Bp_bm,Vgrid_bm,En_V_bm,P_V_bm, 'BIRCH-MURNAGHAN')
    eosplot(V,E,V0_v,E0_v,B0_v,Bp_v,Vgrid_v,En_V_v,P_V_v, 'VINET')
    eosplot(V,E,V0_log,E0_log,B0_log,Bp_log,Vgrid_log,En_V_log,P_V_log, 'POIRIER-TARANTOLA')


def mergeplots():
    from PyPDF2 import PdfFileMerger

    pdfs = ['eos_all.pdf', 'eos_BIRCH-MURNAGHAN.pdf', 'eos_VINET.pdf', 
            'eos_POIRIER-TARANTOLA.pdf', 'eos_MURNAGHAN.pdf']

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write("result.pdf")


if __name__ == '__main__':
    start_time_total = time.time()
    # all parameters from thermo_config
    
    V,E=readev("ev.file")
    E=E*Ry2eV
    
    saveEOSfitdata(V,E)

    # show the running time
    end_time_total = time.time()
    time_elapsed = end_time_total-start_time_total
    print("time elapsed: " + '{:8.2f}'.format(time_elapsed) + " seconds")
