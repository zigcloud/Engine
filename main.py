import astropy.table

from kepler import KeplerPropagator
from SGP4 import Sgp4Propagator
from MoonSun import *
import astropy.constants as c
from TLEtoKepler import TLEtoKeplerConverter
from Utils import *
from typing import List
from pathlib import Path
import time
from FieldOfView import FieldOfView

class Transformator:
    def __init__(self, site: str, observerLocation: EarthLocation, Elements: List[KeplerianElements] or Path,
                 objectID: str, TimeStartIsot: str, TimeEndIsot: str, TimeStep: float, mode: str = 'Kepler',
                 verbose: bool = False, savePath: Path = None,
                 phaseParams: Path = Path('./Resources/summaryPhaseCurveTable.json')):

        self.site = site
        self.inputID = objectID
        self.mode = mode
        self.runID = Time.now().to_datetime().strftime('%Y%m%d%H%M%S')
        if self.inputID == '':
            self.filename = f'{self.site}_{self.runID}_all_{self.mode}.txt'
        else:
            self.filename = f'{self.site}_{self.runID}_{self.inputID}_{self.mode}.txt'

        if type(Elements) == list:
            self.elements = np.array(Elements)
            print(f'TLEs read!\nObjects to be evaluated: {len(self.elements)}')
        else:
            converter = TLEtoKeplerConverter(Elements, objectID)
            self.elements = np.array(converter.converter())
            print(f'TLEs read!\nObjects to be evaluated: {len(self.elements)}')
        self.GM = c.GM_earth.value
        self.stateVector = []
        if TimeStartIsot is None or TimeStartIsot == '':
            self.startTime = Time(Time.now().isot, format='isot', scale='utc')
        else:
            self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
        if TimeStartIsot is None or TimeStartIsot == '':
            self.endTime = Time(Time.now().isot, format='isot', scale='utc')
        else:
            self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
        self.stepTime = TimeDelta(TimeStep, format='sec')
        self.obs = observerLocation
        self.verbose = verbose
        self.savePath = savePath
        print(f'Propagator {mode} initialization ...')
        if mode == 'Kepler':
            self.propagator = KeplerPropagator(site, observerLocation, Elements, objectID, self.startTime.isot,
                                               self.endTime.isot, TimeStep, verbose=self.verbose)
        elif mode == 'SGP4' and type(Elements) != List:
            self.propagator = Sgp4Propagator(site, observerLocation, Elements, objectID, self.startTime.isot,
                                             self.endTime.isot, TimeStep, verbose=self.verbose)
        else:
            exit('Bad mode selected!')
        print(f'Propagator {mode} initialization done!')
        self.phaseParams = getTableFromJson(phaseParams)
        self.outputTable = Table(names=outputTablenames, dtype=outputTabletypes, units=outputTableunits)

    def Run(self):
        import time
        actualRunTime = time.time()
        print('Propagation Started')
        self.propagator.propagate()
        print('Propagation completed, time: ' + str(time.time() - actualRunTime))
        actualRunTime = time.time()
        print('Get Moon and SUn')

        self.getMoonSun(self.propagator.timeArray)
        print('Moon and Sun generated, time: ' + str(time.time() - actualRunTime))

        actualRunTime = time.time()
        print('Transformation started')

        self.coordsITRS = []
        self.shadow = []
        for obj in self.propagator.stateVector:
            sat = []
            for i, step in enumerate(self.propagator.timeArray):
                satItrs = ConvertGcrsToItrs(Time(step, format='isot'), obj[i].r)
                sat.append(satItrs)
            self.coordsITRS.append(SkyCoord(sat,
                                            unit=u.m, representation_type='cartesian', frame='itrs',
                                            obstime=Time(self.propagator.timeArray, format='isot', location=self.obs),
                                            location=self.obs))
        self.coordsAltAz = [self.coordsITRS[obj].transform_to(AltAz(obstime=self.propagator.timeArray,
                                                                    location=self.obs))
                            for obj in range(len(self.coordsITRS))]

        self.coordsGCRS = [ConvertAltAzToRADEC(self.coordsAltAz[obj]) for obj in range(len(self.coordsITRS))]

        self.coordGeodetic = [ConvertItrsToGeodetic(self.coordsITRS[obj]) for obj in range(len(self.coordsITRS))]
        self.shadow = [GetEarthShadowVectorised([self.propagator.stateVector[obj][i].r
                                                for i in range(len(self.propagator.timeArray))],
                                                self.sunAltAz) for obj in range(len(self.coordsITRS))]

        self.phaseAngle = [GetPhaseAngleVectorised(self.obs, self.coordsAltAz[obj], self.sunAltAz)
                           for obj in range(len(self.coordsAltAz))]

        self.dRA = [GetRatesVectorised(self.coordsGCRS[obj].ra, self.stepTime.to(u.day))
                    for obj in range(len(self.coordsGCRS))]
        self.dDE = [GetRatesVectorised(self.coordsGCRS[obj].dec, self.stepTime.to(u.day))
                    for obj in range(len(self.coordsGCRS))]

        self.beta = [getBeta(self.phaseParams, self.elements[obj]) for obj in range(len(self.elements))]
        self.areaRho = [getArho(self.phaseParams, self.elements[obj]) for obj in range(len(self.elements))]

        self.magApp = [Hejduk_F1F2_beta(self.phaseAngle[obj], self.areaRho[obj], self.beta[obj])
                       for obj in range(len(self.elements))]
        self.magAbs = [Hejduk_F1F2_beta(0, self.areaRho[obj], self.beta[obj])
                       for obj in range(len(self.elements))]

        self.magAbs = [Hejduk_F1F2_beta(0, self.areaRho[obj], self.beta[obj])
                       for obj in range(len(self.elements))]
        # TODO - to implement Luminosity function GetLuminosity
        self.luminosity = [GetLuminosity(np.array(self.magAbs)) for _ in range(len(self.elements))]

        print('Transformation done, time: ' + str(time.time() - actualRunTime))
        print('Generating output Table ...')
        self.generateOutputTable()
        return self.outputTable

    def generateOutputTable(self):
        for i in range(len(self.propagator.elements)):
            for j in range(len(self.propagator.timeArray) - 1):
                line = [self.propagator.elements[i].id.id,  # ObjectID
                        self.propagator.timeArray[j].mjd * u.day,  # MJD
                        self.coordsAltAz[i][j].distance,  # Range
                        self.phaseAngle[i][j] * u.deg,  # Phase Angle
                        self.coordsGCRS[i][j].ra.deg * u.deg,  # RA
                        self.coordsGCRS[i][j].dec.deg * u.deg,  # DEC
                        self.dRA[i][j].to(u.arcsec / u.s),  # dRA - only single value for now
                        self.dDE[i][j].to(u.arcsec / u.s),  # dDE - only single value for now
                        GetRatesProjection(self.dRA[i][j].to(u.arcsec / u.s),
                                           self.dDE[i][j].to(u.arcsec / u.s)) * self.stepTime,  # Length
                        self.beta[i],  # 'Beta'
                        self.areaRho[i],  # 'A*rho'
                        self.magAbs[i],  # 'm_abs'
                        self.magApp[i][j],  # 'm_app'
                        self.luminosity[i][j],  # 'Luminosity'
                        self.shadow[i][j],  # Shadow
                        self.coordGeodetic[i].lon[j].deg * u.deg,  # Longitude
                        self.coordGeodetic[i].lat[j].deg * u.deg  # Latitude
                        ]
                self.outputTable.add_row(line)
        if self.verbose:
            self.outputTable.info()
            self.outputTable.pprint_all()

        if self.savePath is not None:
            self.SaveOutputTable()

    def SaveOutputTable(self):

        out = self.outputTable
        out.pprint_all()
        with open(Path(self.savePath, self.filename), 'w') as f:
            f.write('#' + '\t'.join(outputTablenames) + '\n')
            f.write('#' + '\t'.join([str(un) for un in outputTableunits]) + '\n')
            for line in out:
                f.write(format_string.format(*line))
        print(f'Output saved at: {Path(self.savePath, self.filename).resolve()}')

    def getMoonSun(self, timeArr):
        moonSun = MoonSunGenerator(self.site, self.obs, timeArr)
        res = moonSun.getMoonSun()
        mjd = [step for step in res[self.site].keys()]
        self.sunAltAz = SkyCoord(location=self.obs, obstime=Time(mjd, format='mjd'), frame='altaz',
                                 az=[res[self.site][t]['az_sun'] for t in mjd] * u.deg,
                                 alt=[res[self.site][t]['alt_sun'] for t in mjd] * u.deg,
                                 distance=(1 * u.au).to(u.m))

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
    ti = time.time()
    print('computation started')

    # observer Location
    obs = EarthLocation(lon=17.2736306 * u.deg, lat=48.372528 * u.deg, height=536.1 * u.m)
    # inputTLE file
    tleData = Path(r'./starlinkGEN1_tle.txt')
    # output Table name and path
    outPath = Path(r'./')

    # Main class initialization
    # site - name of the desired site - only for naming
    # observerLocation - astropy.coordinates.EarthLocation filled with Lon, Lat and Height of the site
    # Elements - List of Keplerian element (see Utils) or Path to the Tle file
    # ObjectID - ID of the target - Norad, Cospar, or specific name of population or empty string (than whole
    # population is taken)
    # TimeStartIsot - Isot date and time of the start - if None or empty string "" actual utc time is taken
    # TimeEndIsot - Isot date and time of the end - if None or empty string "" actual utc time is taken
    # TimeStep - in second - length of the ephemeris step also serves as Exposure time to calculate the Length
    # mode - Kepler or SGP4 - defines which propagator shall be used
    # verbose - Bool - whether the more talkative output shall be shown or not
    # savePath - Path to the file where output table shall be saved - if None, no output is saved only returned
    # phaseParams - Path to the summary json file with result from the phase curve fitting. Particular files can
    # be merged into the single json with outside function readJsonDat.py
    a = Transformator(site='AGO', observerLocation=obs, Elements=tleData, objectID='',
                      TimeStartIsot='', TimeEndIsot='',
                      TimeStep=600, mode='SGP4', verbose=False, savePath=outPath,
                      phaseParams=Path('./Resources/summaryPhaseCurveTable.json'))

    tbl = a.Run()  # Main transformator's function - return astropy.table.Table with names shown in Utils
    print('computation completed, time: ' + str(time.time() - ti))

    # PLOTTING
    import matplotlib

    matplotlib.rcParams.update({'font.size': 12})
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import contextily as cx

    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))
    countries.head()
    fig = plt.figure(figsize=(20, 10), layout='constrained')
    plt.suptitle(a.filename.replace('.txt', ''))
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
    plt.savefig(Path(outPath, a.filename.replace('.txt', '.png')), format='png', dpi=300)
    plt.show()


    # Visible objects filter
    # FiledOfView input parameters
    # width - width of the FoV in degrees
    # height - height of the FoV in degrees
    # ra - Right ascension of the center of the FoV in degrees
    # dec - Declination of the center of the FoV in degrees
    # population - Transformator class output table or astropy.Table
    # verbose - whether to print also the table for the visible objects

    teleskop = FieldOfView(width=5, height=5, ra=0, dec=0, population=tbl, verbose=True)

    teleskop.prettyPrintOutput()