from kepler import KeplerPropagator
from astropy.utils.iers import conf
from SGP4 import Sgp4Propagator
from MoonSun import *
from astropy.table import Table
from TLEtoKepler import TLEtoKeplerConverter
import time as t




class Transformator:
    def __init__(self, site: str, observerLocation: EarthLocation, Elements: List[KeplerianElements] or Path, objectID: str,
                 TimeStartIsot: str, TimeEndIsot: str, TimeStep: float, mode: str = 'Kepler', verbose: bool = False,
                 savePath: Path = None, phaseParams: Path = Path('./Resources/summaryPhaseCurveTable.json')):
        self.site = site
        self.inputID = objectID
        self.mode = mode
        self.runID = Time.now().to_datetime().strftime('%Y%m%d%H%M%S')
        if type(Elements) == list:
            self.elements = np.array(Elements)
            print(f'TLEs read!\nObjects to be evaluaeted: {len(self.elements)}')
        else:
            converter = TLEtoKeplerConverter(Elements, objectID)
            self.elements = np.array(converter.converter())
            print(f'TLEs read!\nObjects to be evaluaeted: {len(self.elements)}')
        self.GM = c.GM_earth.value
        self.stateVector = []
        self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
        self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
        self.stepTime = TimeDelta(TimeStep, format='sec')
        self.obs = observerLocation
        self.verbose = verbose
        self.savePath = savePath
        print(f'Propagator {mode} initialization ...')
        if mode == 'Kepler':
            self.propagator = KeplerPropagator(site, observerLocation, Elements, objectID, TimeStartIsot,
                                               TimeEndIsot, TimeStep, verbose=self.verbose)
        elif mode =='SGP4' and type(Elements) != List:
            self.propagator = Sgp4Propagator(site, observerLocation, Elements, objectID, TimeStartIsot,
                                             TimeEndIsot, TimeStep, verbose=self.verbose)
        else:
            exit('Bad mode selected!')
        print(f'Propagator {mode} initialization done!')
        self.phaseParams = getTableFromJson(phaseParams)




    def GetGMSTofTime(self, time):
        """
        Compute greenwich mean sidereal time for given time
        https://astronomy.stackexchange.com/questions/21002/how-to-find-greenwich-mean-sideral-time
        @param time: astropy.time.Time
        @return: float
        """
        d = time.jd - 2451545.0
        T = d / 36525
        fullCircleRad = (fullCircle * u.deg).to(u.rad)
        gmst = 24110.54841 + 8640184.812866 * T + 0.093104 * T ** 2 - 0.0000062 * T ** 3
        gmst = gmst * fullCircleRad / secondsInDay + fullCircleRad / 2 + fullCircleRad * time.jd
        return gmst.value % fullCircleRad.value

    def ConvertGcrsToItrs(self, obstime, gcrs_pos):
        """
        Convert object position in gcrs frame to itrs
        @param obstime: astropy.time.Time
        @param gcrs_pos: n x 3 array
        @return: n x 3 array itrs coordinates
        """
        gmst_angle = -self.GetGMSTofTime(obstime)
        gmst_angle_cos = np.cos(gmst_angle)
        gmst_angle_sin = np.sin(gmst_angle)
        gmst_matrix = np.array([[gmst_angle_cos, gmst_angle_sin, 0], [-gmst_angle_sin, gmst_angle_cos, 0], [0, 0, 1]])
        return np.array(gcrs_pos).dot(gmst_matrix)

    def ConvertAltAzToRADEC(self,satelliteAltAz):
        """
        Convert object position in altaz frame to radec topocentric
        @param satelliteAltAz: astropy.coordinates.SkyCoord (altaz frame)
        @return: astropy.coordinates.SkyCoord (gcrs frame)
        """
        satellite_altaz_2 = AltAz(az=satelliteAltAz.spherical.lon, alt=satelliteAltAz.spherical.lat,
                                        obstime=satelliteAltAz.obstime, location=satelliteAltAz.location)
        res = satellite_altaz_2.transform_to(GCRS(obstime=satelliteAltAz.obstime))
        return SkyCoord(res)

    def ConvertItrsToGeodetic(self, satellite):
        """
        Convert object location in itrs frame to geodetic location
        @param satellite: astropy.coordinates.SkyCoord (itrs frame, in cartesian representation)
        @return: astropy.coordinates.earth.GeodeticLocation
        """
        earth_loc = EarthLocation(x=satellite.x, y=satellite.y, z=satellite.z)
        return earth_loc.to_geodetic()

    def GetEarthShadowVectorised(self, obj_gcrs, sun):
        """
        Compute whether or not an object is in shadow
        @param obj_gcrs: n x 3 array
        @param sun: astropy.coordinates.SkyCoord (altaz frame)
        @return: n sized array containing bool values
        """
        c = np.array(obj_gcrs) * 1000
        c_dist = np.linalg.norm(c, axis=1)
        anti_sun = -np.array([sun.gcrs.cartesian.x.value,
                              sun.gcrs.cartesian.y.value,
                              sun.gcrs.cartesian.z.value]).transpose()
        alpha = np.arccos(np.sum(c * anti_sun, axis=1) / (c_dist * np.linalg.norm(anti_sun, axis=1)))
        a = c_dist * np.sin(alpha)

        return (np.degrees(alpha) < rightAngle) * (a < earthRadiusInMetres)

    def GetRatesVectorised(self, val, deltaTime):
        """
        compute rates for given values and time step (for n sized input array returns n-1 sized array)
        @param val: astropy.units.quantity.Quantity containing n sized array (in degree or rad)
        @param deltaTime: astropy.units.quantity.Quantity
        @return: astropy.units.quantity.Quantity containing n-1 sized array (in deg / h )
        """
        res = val[1:] - val[:-1]
        res = np.where(res > halfCircle * u.deg, res - fullCircle * u.deg, res)
        res = np.where(res < -halfCircle * u.deg, res + fullCircle * u.deg, res)
        return (res / deltaTime).to(u.deg / u.h)

    def GetPhaseAngleVectorised(self, obs, obj, sun):
        """
        Compute phase angle for given observer location, object location and sun location
        https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
        @param obs: astropy.coordinates.earth.EarthLocation
        @param obj: astropy.coordinates.SkyCoord (altaz frame) containing n sized array
        @param sun: astropy.coordinates.SkyCoord (altaz frame) containing n sized array
        @return: n sized array
        """
        a = list(obs.value)
        b = np.array([obj.itrs.x.value, obj.itrs.y.value, obj.itrs.z.value]).transpose()
        c = np.array([sun.itrs.x.value, sun.itrs.y.value, sun.itrs.z.value]).transpose()

        v1 = b - a
        v2 = b - c
        res = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        return np.arccos(res)

    def GetRatesVectorised(self, val, deltaTime):
        """
        compute rates for given values and time step (for n sized input array returns n-1 sized array)
        @param val: astropy.units.quantity.Quantity containing n sized array (in degree or rad)
        @param deltaTime: astropy.units.quantity.Quantity
        @return: astropy.units.quantity.Quantity containing n-1 sized array (in deg / h )
        """
        res = val[1:] - val[:-1]
        res = np.where(res > halfCircle * u.deg, res - fullCircle * u.deg, res)
        res = np.where(res < -halfCircle * u.deg, res + fullCircle * u.deg, res)
        return (res / deltaTime).to(u.deg / u.h)

    def GetAngularDistance(self, obj1, obj2):
        """
        Compute angular distance between two points. https://en.wikipedia.org/wiki/Angular_distance
        @param obj1: astropy.coordinates.SkyCoord (gcrs frame)
        @param obj2: astropy.coordinates.SkyCoord (gcrs frame)
        @return: float value or array representing angular distance in degrees
        """
        return np.arccos(np.sin(obj1.dec) * np.sin(obj2.dec) +
                         np.cos(obj1.dec) * np.cos(obj2.dec) * np.cos(obj1.ra - obj2.ra)).to(u.deg).value

    def GetRatesProjection(self, dRA, dDE):
        return np.sqrt(dRA**2+dDE**2)

    def getBeta(self, phaseParams, element):
        #TODO - properly selected which Beta shall be used
        if len(phaseParams[phaseParams['norad']==element.id.id])>0:
            return np.median(phaseParams[phaseParams['norad']==element.id.id]['Hejduk_med_beta'])
        elif len(isStringInTable(phaseParams,element.id.name,2))>0:
            return np.median(isStringInTable(phaseParams, element.id.name,2)['Hejduk_med_beta'])
        else:
            return np.median(phaseParams['Hejduk_med_beta'])

    def getArho(self, phaseParams, element):
        #TODO - properly selected which AreaRho shall be used

        if len(phaseParams[phaseParams['norad']==element.id.id])>0:
            return np.median(phaseParams[phaseParams['norad']==element.id.id]['Hejduk_med_AreaRo'])
        elif len(isStringInTable(phaseParams,element.id.name,2))>0:
            return np.median(isStringInTable(phaseParams, element.id.name,2)['Hejduk_med_AreaRo'])
        else:
            return np.median(phaseParams['Hejduk_med_AreaRo'])

    def Run(self):
        ti = t.time()
        print('Propagation Started')
        self.propagator.propagate()
        print('Propagation completed, time: ' + str(t.time() - ti))
        ti = t.time()
        print('Get Moon and SUn')

        self.getMoonSun(self.propagator.timeArray)
        print('Moon and Sun generated, time: ' + str(t.time() - ti))

        ti = t.time()
        print('Transformation sterted')

        self.coordsITRS = []
        self.shadow = []
        for obj in self.propagator.stateVector:
            sat = []
            for i, time in enumerate(self.propagator.timeArray):
                satItrs = self.ConvertGcrsToItrs(Time(time,format='isot'),obj[i].r)
                sat.append(satItrs)
            self.coordsITRS.append(SkyCoord(sat,
                                    unit=u.m, representation_type='cartesian',frame='itrs',
                                    obstime=Time(self.propagator.timeArray,format='isot', location=self.obs),
                                    location=self.obs))
        self.coordsAltAz = [self.coordsITRS[obj].transform_to(AltAz(obstime=self.propagator.timeArray, location=self.obs))
                            for obj in range(len(self.coordsITRS))]



        self.coordsGCRS = [self.ConvertAltAzToRADEC(self.coordsAltAz[obj]) for obj in range(len(self.coordsITRS))]

        self.coordGeodetic = [self.ConvertItrsToGeodetic(self.coordsITRS[obj]) for obj in range(len(self.coordsITRS))]
        self.shadow = [self.GetEarthShadowVectorised([self.propagator.stateVector[obj][i].r
                                                      for i in range(len(self.propagator.timeArray))],
                                                      self.sunAltAz) for obj in range(len(self.coordsITRS))]

        self.phaseAngle = [self.GetPhaseAngleVectorised(self.obs, self.coordsAltAz[obj], self.sunAltAz)
                           for obj in range(len(self.coordsAltAz))]

        self.dRA = [self.GetRatesVectorised(self.coordsGCRS[obj].ra, self.stepTime.to(u.day))
                    for obj in range(len(self.coordsGCRS))]
        self.dDE = [self.GetRatesVectorised(self.coordsGCRS[obj].dec, self.stepTime.to(u.day))
                    for obj in range(len(self.coordsGCRS))]

        self.beta = [self.getBeta(self.phaseParams, self.elements[obj]) for obj in range(len(self.elements))]
        self.areaRho = [self.getArho(self.phaseParams, self.elements[obj]) for obj in range(len(self.elements))]

        self.magApp = [Hejduk_F1F2_beta(self.phaseAngle[obj], self.areaRho[obj], self.beta[obj])
                       for obj in range(len(self.elements))]
        self.magAbs = [Hejduk_F1F2_beta(0, self.areaRho[obj], self.beta[obj])
                       for obj in range(len(self.elements))]


        self.magAbs = [Hejduk_F1F2_beta(0, self.areaRho[obj], self.beta[obj])
                       for obj in range(len(self.elements))]
        #TODO - to implement Luminosity function GetLuminosity
        self.luminosity = [GetLuminosity(self.propagator.timeArray) for obj in range(len(self.elements))]

        print('Tranformation done, time: ' + str(t.time() - ti))
        print('Generating output Table ...')
        self.generateOutputTable()
        return self.outputTable

    def generateOutputTable(self):
        self.outputTable = Table(names=outputTablenames, dtype=outputTabletypes, units=outputTableunits)
        for i in range(len(self.propagator.elements)):
            for j in range(len(self.propagator.timeArray)-1):
                line=[self.propagator.elements[i].id.id,  #ObjectID
                      self.propagator.timeArray[j].mjd*u.day,  #MJD
                      self.coordsAltAz[i][j].distance,  #Range
                      self.phaseAngle[i][j] * u.deg,  #Phase Angle
                      self.coordsGCRS[i][j].ra.deg * u.deg,  #RA
                      self.coordsGCRS[i][j].dec.deg * u.deg,  #DEC
                      self.dRA[i][j].to(u.arcsec / u.s),  #dRA - only single value for now
                      self.dDE[i][j].to(u.arcsec / u.s),  #dDE - only single value for now
                      self.GetRatesProjection(self.dRA[i][j].to(u.arcsec / u.s),
                                              self.dDE[i][j].to(u.arcsec / u.s)) * self.stepTime,  #Length
                      self.beta[i], # 'Beta'
                      self.areaRho[i], # 'A*rho'
                      self.magAbs[i], # 'm_abs'
                      self.magApp[i][j], # 'm_app'
                      self.luminosity[i][j], # 'Luminosity'
                      self.shadow[i][j],  #Shadow
                      self.coordGeodetic[i].lon[j].deg* u.deg,  #Longitude
                      self.coordGeodetic[i].lat[j].deg* u.deg  # Latitude
                      ]
                self.outputTable.add_row(line)
        if self.verbose:
            self.outputTable.info()
            self.outputTable.pprint_all()

        if self.savePath is not None:
            self.SaveOutputTable()


    def SaveOutputTable(self):
        if self.inputID == '':
            self.filename = f'{self.site}_{self.runID}_all_{self.mode}.txt'
        else:
            self.filename = f'{self.site}_{self.runID}_{self.inputID}_{self.mode}.txt'

        out = self.outputTable
        rounding = dict(zip(outputTablenames, outputTableround))
        out.round(rounding)
        with open(Path(self.savePath,self.filename), 'w') as f:
            f.write('#'+ '\t'.join(outputTablenames)+'\n')
            f.write('#'+ '\t'.join([str(un) for un in outputTableunits])+'\n')
            for line in out:
                f.write('\t'.join(str(n) for n in line)+'\n')

        print(f'Output saved at: {Path(self.savePath,self.filename).resolve()}')


    def getMoonSun(self, timeArr):
        moonSun = MoonSunGenerator(self.site, self.obs, timeArr)
        res = moonSun.getMoonSun()
        mjd = [x for x in res[self.site].keys()]
        self.sunAltAz = SkyCoord(location=self.obs, obstime=Time(mjd,format='mjd'), frame='altaz',
                                 az=[res[self.site][t]['az_sun'] for t in mjd]* u.deg,
                                 alt=[res[self.site][t]['alt_sun'] for t in mjd]* u.deg,
                                 distance=(1*u.au).to(u.m))

        self.moonAltAz = SkyCoord(location=self.obs, obstime=Time(mjd, format='mjd'), frame='altaz',
                                 az=[res[self.site][t]['az_moon'] for t in mjd] * u.deg,
                                 alt=[res[self.site][t]['alt_moon'] for t in mjd] * u.deg,
                                 distance=(1 * u.au).to(u.m))

        self.sunRADEC = SkyCoord(location=self.obs, obstime=Time(mjd, format='mjd'), frame='gcrs',
                                 ra=[res[self.site][t]['RA_t_sun'] for t in mjd] * u.deg,
                                 dec=[res[self.site][t]['DE_t_sun'] for t in mjd] * u.deg)

        self.moonRADEC = SkyCoord(location=self.obs, obstime=Time(mjd, format='mjd'), frame='gcrs',
                                  ra=[res[self.site][t]['RA_t_moon'] for t in mjd] * u.deg,
                                  dec=[res[self.site][t]['DE_t_moon'] for t in mjd] * u.deg)





if __name__ == '__main__':
    import time as t
    ti = t.time()
    print('computation started')

    #observer Location
    obs = EarthLocation(lon=17.2736306*u.deg, lat=48.372528*u.deg, height=536.1*u.m)
    #inputTLE file
    tleData = Path(r'./3le.txt')
    #output Table name and path
    outPath = Path(r'./')

    #Main class initialization
    #site - name of the desired site - only for naming
    #observerLocation - astropy.coordinates.EarthLocation filled with Lon, Lat and Height of the site
    #Elements - List of Keplerian element (see Utils) or Path to the Tle file
    #ObjectID - ID of the target - Norad, Cospar, or specific name of population or empty string (than whole
    # population is taken)
    #TimeStartIsot - Isot date and time of the start
    #TimeEndIsot - Isot date and time of the end
    #TimeStep - in second - length of the ephermeris step also serves as Exposure time to calculate the Length
    #mode - Kepler or SGP4 - defines which propagator shall be used
    #verbose - Bool - whether the more talkative output shall be shown or not
    #savePath - Path to the file where output table shall be saved - if None, no output is saved only returned
    #phaseParams - Path to the summary json file with result from the phase curve fitting. Particular files can
    # be merged into the sinlge json with outside function readJsonDat.py
    a = Transformator(site='AGO',observerLocation=obs, Elements=tleData, objectID='37775',
                      TimeStartIsot='2023-09-15T10:00:00', TimeEndIsot='2023-11-15T10:00:00',
                      TimeStep=600, mode='Kepler', verbose=False, savePath=outPath,
                      phaseParams=Path('./Resources/summaryPhaseCurveTable.json'))

    tbl = a.Run() #Main transformator's funtion - return astropy.table.Table with names shown in Utils
    print('computation completed, time: ' + str(t.time() - ti))

    #PLOTTING
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})
    from scipy.interpolate import CubicSpline
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import contextily as cx


    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)
        xbinwidth = 10
        ybinwidth = 10

        xmax = np.max(np.abs(x))
        ymax = np.max(np.abs(y))
        xlim = (int(xmax / xbinwidth) + 1) * xbinwidth
        ylim = (int(ymax / ybinwidth) + 1) * ybinwidth
        xbins = np.arange(-xlim, xlim + xbinwidth, xbinwidth)
        ybins = np.arange(-ylim, ylim + ybinwidth, ybinwidth)

        ax_histx.hist(x, bins=xbins)
        ax_histy.hist(y, bins=ybins, orientation='horizontal')

    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))
    countries.head()
    fig = plt.figure(figsize=(20,10),layout='constrained')
    plt.suptitle(a.filename.replace('.txt',''))
    # Create the main axes, leaving 25% of the figure space at the top and on the
    # right to position marginals.

    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    # The main axes' aspect can be fixed.
    ax.set(aspect=0.5)
    # whole world
    ax.set_xlim(-180, 180)
    ax.set_ylim(-85, 85)
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    x = [line[0] for line in tbl.iterrows('Lon')]
    y = [line[0] for line in tbl.iterrows('Lat')]
    labels = [line[0] for line in tbl.iterrows('ObjectID')]
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    cx.add_basemap(ax, crs=countries.crs.to_string(), source=cx.providers.CartoDB.Voyager)

    plt.xlabel("Longitude [°]")
    plt.ylabel("Latitude [°]")
    plt.savefig(Path(outPath,a.filename.replace('.txt','.png')), format='png',dpi=300)
    plt.show()

