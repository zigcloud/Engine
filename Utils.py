import numpy as np
from astropy.table import Table
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from astropy.coordinates import AltAz, GCRS, SkyCoord, EarthLocation
import astropy.units as u
import json
import astropy.constants as c
from typing import List
from pathlib import Path

ssr = './Resources/SSR-Full-20230821.csv'
ssr = Table.read(ssr, format='ascii.csv', delimiter=';')

fullCircle = 360
halfCircle = 180
secondsInMinute = 60
secondsInHour = 3600
secondsInDay = 86400
minutesInHour = 60
minutesInDay = 1440
hoursInDay = 24
dummyAPmag = 20
rightAngle = 90
earthRadiusInMetres = 6378150
mjdJdShift = 2400000.5
magSun = -26.74
format_string = '{}\t{:.9f}\t{:.3e}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.3e}\t{}\t{:.5f}\t{:.5f}\n'
outputTablenames = ['ObjectID','MJD','Range','Phase angle','RA','DE',
                    'dRA','dDE','Length','Beta','A*rho', 'm_abs','m_app','Luminosity', 'Shadow', 'Lon', 'Lat']
outputTabletypes = [str, float, float, float,float,float,
                    float,float,float,float,float,float,float,float, bool, float, float]
outputTableunits = ['',u.day, u.m, u.deg, u.deg, u.deg,
                    u.arcsec / u.s, u.arcsec / u.s, u.arcsec, '--',u.m*u.m,u.mag,u.mag, u.W,'--', u.deg, u.deg]
phaseParamsTabNames = ["run_id", "norad", "name", "phase_min", "phase_max", "mjd_min", "mjd_max",
                       "Hejduk_med_beta", "Hejduk_med_AreaRo","Hejduk_med_AbsMag", "Hejduk_med_beta_sigma",
                       "Hejduk_med_AreaRo_sigma", "Hejduk_med_AbsMag_sigma", "Hejduk_med_no_of_points",
                       "Hejduk_predictionBand_upper_AreaRo", "Hejduk_predictionBand_lower_AreaRo",
                       "Hejduk_predictionBand_upper_AbsMag", "Hejduk_predictionBand_lower_AbsMag"]
phaseParamsTabTypes = [str, str, str, float, float, float, float, float, float, float, float, float, float, int, float, float, float, float]


@dataclass_json
@dataclass
class inputID(object):
    id: str = ''
    cospar: bool = False
    norad: bool = False
    population: bool = False
    directTles: bool = False

    def __post_init__(self):
        if self.id != '':
            if len(self.id)<5:
                self.id = '0'*(5-len(self.id)) + self.id
            if self.id[0:4].isdigit() and self.id[4].isalpha():
                self.cospar = True
            elif self.id.isdigit():
                self.norad = True
            elif self.id[0:5].isdigit() != True and self.id != '':
                self.population = True
        else:
            self.directTles = True
@dataclass_json
@dataclass
class ObjectID(object):
    id: str = ''
    name: str = ''
    cospar: bool = False
    norad: bool = False
    name: str = ''

    def __post_init__(self):
        if len(self.id)<5:
            self.id = '0'*(5-len(self.id)) + self.id

        if self.id[0:4].isdigit() and self.id[4].isalpha():
            self.cospar = True
        elif self.id.isdigit():
            self.norad = True

        # if self.norad or self.cospar:
        #     self.name = ssrInfo(self, ssr)[1]

def isStringInTable(table, string, columnid):
    newTable = Table(names=table.columns, dtype=table.dtype)
    for row in table.iterrows():
        if string in row[columnid]:
            newTable.add_row(row)

    return newTable
@dataclass_json
@dataclass
class KeplerianElements(object):
    a: float = 0 #Semimajor axis [m]
    e: float = 0 #Eccentricity [-]
    i: float = 0 #Inclination [rad]
    Omega: float = 0 #Longitude of the ascending node [rad]
    omega: float = 0 #Argument of pericenter  [rad]
    M0: float = 0 #Mean anomaly at epoch [rad]
    epoch: str = None #Epoch for elements validity - used if TLEs are used [isot format]
    timeSincePerigee: str = None # days since the last perigee
    n: float = None # mean motion [rad/s]
    id: ObjectID = None



@dataclass_json
@dataclass
class stateVector(object):
    r: np.array
    v: np.array
    def values(self):
        return self.r,self.v
    def pprint(self):
        r = np.round(self.r, 3)
        v = np.round(self.v, 3)
        print(f'GCRS State vector   : x={np.round(r[0], 3)}m y={np.round(r[1], 3)}m z={np.round(r[2], 3)}m\n'
              f'GCRS Velocity vector: x={np.round(v[0], 3)}m/s y={np.round(v[1], 3)}m/s z={np.round(v[2], 3)}m/s')


@dataclass_json
@dataclass
class RotationMatrix(object):
    theta: float = 0

    def Rx(theta):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    def Rz(theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])

def Hejduk_F1F2_beta(phaseAngle, a_ro, beta):
    x = np.array(phaseAngle)
    F_diff = 2/(3*np.pi*np.pi)*((np.pi-x)*np.cos(x)+np.sin(x))
    F_spec = 1/(4*np.pi)
    return magSun - 2.5*np.log10(a_ro*(beta*F_diff+(1-beta)*F_spec)) + 5*np.log10(149597871000.0)

def GetLuminosity(absMag):
    #only a dummy function
    luminosity = c.L_sun * np.power(10,(-(absMag-magSun)*(2/5)))
    return luminosity

def getTableFromJson(jsonFile):
    phaseParamsTab = Table(names=phaseParamsTabNames, dtype=phaseParamsTabTypes)
    with open(jsonFile, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        phaseParamsTab.add_row(value)

    return phaseParamsTab


def GetGMSTofTime(time):
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


def ConvertGcrsToItrs(obstime, gcrs_pos):
    """
    Convert object position in gcrs frame to itrs
    @param obstime: astropy.time.Time
    @param gcrs_pos: n x 3 array
    @return: n x 3 array itrs coordinates
    """
    gmst_angle = -GetGMSTofTime(obstime)
    gmst_angle_cos = np.cos(gmst_angle)
    gmst_angle_sin = np.sin(gmst_angle)
    gmst_matrix = np.array([[gmst_angle_cos, gmst_angle_sin, 0], [-gmst_angle_sin, gmst_angle_cos, 0], [0, 0, 1]])
    return np.array(gcrs_pos).dot(gmst_matrix)


def ConvertAltAzToRADEC(satelliteAltAz):
    """
    Convert object position in altaz frame to radec topocentric
    @param satelliteAltAz: astropy.coordinates.SkyCoord (altaz frame)
    @return: astropy.coordinates.SkyCoord (gcrs frame)
    """
    satellite_altaz_2 = AltAz(az=satelliteAltAz.spherical.lon, alt=satelliteAltAz.spherical.lat,
                                    obstime=satelliteAltAz.obstime, location=satelliteAltAz.location)
    res = satellite_altaz_2.transform_to(GCRS(obstime=satelliteAltAz.obstime))
    return SkyCoord(res)


def ConvertItrsToGeodetic(satellite):
    """
    Convert object location in itrs frame to geodetic location
    @param satellite: astropy.coordinates.SkyCoord (itrs frame, in cartesian representation)
    @return: astropy.coordinates.earth.GeodeticLocation
    """
    earth_loc = EarthLocation(x=satellite.x, y=satellite.y, z=satellite.z)
    return earth_loc.to_geodetic()


def GetEarthShadowVectorised(obj_gcrs, sun):
    """
    Compute whether an object is in shadow
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


def GetPhaseAngleVectorised(obs, obj, sun):
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


def GetRatesVectorised(val, deltaTime):
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


def GetAngularDistance(obj1, obj2):
    """
    Compute angular distance between two points. https://en.wikipedia.org/wiki/Angular_distance
    @param obj1: astropy.coordinates.SkyCoord (gcrs frame)
    @param obj2: astropy.coordinates.SkyCoord (gcrs frame)
    @return: float value or array representing angular distance in degrees
    """
    return np.arccos(np.sin(obj1.dec) * np.sin(obj2.dec) +
                     np.cos(obj1.dec) * np.cos(obj2.dec) * np.cos(obj1.ra - obj2.ra)).to(u.deg).value


def GetRatesProjection(dRA, dDE):
    print(dRA, dDE, np.sqrt(dRA**2+dDE**2))
    return np.sqrt(dRA**2+dDE**2)


def getBeta(phaseParams, element):
    #TODO - properly selected which Beta shall be used
    if len(phaseParams[phaseParams['norad']==element.id.id])>0:
        return np.median(phaseParams[phaseParams['norad']==element.id.id]['Hejduk_med_beta'])
    elif len(isStringInTable(phaseParams,element.id.name,2))>0:
        return np.median(isStringInTable(phaseParams, element.id.name,2)['Hejduk_med_beta'])
    else:
        return np.median(phaseParams['Hejduk_med_beta'])


def getArho(phaseParams, element):
    #TODO - properly selected which AreaRho shall be used

    if len(phaseParams[phaseParams['norad']==element.id.id])>0:
        return np.median(phaseParams[phaseParams['norad']==element.id.id]['Hejduk_med_AreaRo'])
    elif len(isStringInTable(phaseParams,element.id.name,2))>0:
        return np.median(isStringInTable(phaseParams, element.id.name,2)['Hejduk_med_AreaRo'])
    else:
        return np.median(phaseParams['Hejduk_med_AreaRo'])


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

# def ssrInfo(objectID, ssr):
#     if objectID.norad:
#         try:
#             idx = int(np.where(ssr['NORAD_CAT_ID'] == int(objectID.id))[0])
#             cospar = ssr[idx]['OBJECT_ID'].split('-')
#             cospar = cospar[0][2:5] + cospar[1]
#             return cospar, f'{ssr[idx]["OBJECT_NAME"]} / {ssr[idx]["COUNTRY"]}'
#         except TypeError:
#             print(objectID.id)
#
#     elif objectID.cospar:
#         if int(objectID.id[0:2]) > 24:
#             cospar = '19' + objectID.id[0:2] + '-' + objectID.id[2:]
#         else:
#             cospar = '20' + objectID.id[0:2] + '-' + objectID.id[2:]
#         idx = np.where(ssr['OBJECT_ID'] == cospar)[0]
#         norad = ssr[idx]['NORAD_CAT_ID']
#         return norad.value[0], f'{ssr[idx]["OBJECT_NAME"]} / {ssr[idx]["COUNTRY"]}'
