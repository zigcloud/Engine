from astropy.io import fits
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from astropy.time import Time
from typing import List, Dict, Optional, Tuple
from astropy.coordinates import AltAz, GCRS, SkyCoord
import astropy.constants as c
import astropy.units as u

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
outputTablenames = ['ObjectID','MJD','Range','Phase angle','RA','DE',
                    'dRA','dDE','Length','Beta','A*rho', 'm_abs','m_app','Luminosity', 'Shadow', 'Lon', 'Lat']
outputTabletypes = [str, float, float, float,float,float,
                    float,float,float,float,float,float,float,float, bool, float, float]
outputTableunits = ['',u.day, u.m, u.deg, u.deg, u.deg,
                    u.arcsec / u.s, u.arcsec / u.s, u.arcsec, '--','--','--','--','--', '--', u.deg, u.deg]

@dataclass_json
@dataclass
class ObjectID(object):
    id: str = ''
    cospar: bool = False
    norad: bool = False
    population: bool = False
    directTles: bool = False

    def __post_init__(self):
        if self.id[0:4].isdigit() and self.id[4].isalpha():
            self.cospar = True
        elif self.id.isdigit():
            self.norad = True
        elif self.id[0:5].isdigit() != True and self.id != '':
            self.population = True
        elif self.id == '':
            self.directTles = True

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
    r: np.array= np.zeros(3)
    v: np.array = np.zeros(3)
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


def ConvertAltAzToRADEC(satelliteAltAz):
    """
    Convert object position in altaz frame to radec topocentric
    @param satelliteAltAz: astropy.coordinates.SkyCoord (altaz frame)
    @return: astropy.coordinates.SkyCoord (gcrs frame)
    """
    satellite_altaz = AltAz(az=satelliteAltAz.spherical.lon, alt=satelliteAltAz.spherical.lat,
                            obstime=satelliteAltAz.obstime, location=satelliteAltAz.location)
    res = satellite_altaz.transform_to(GCRS(obstime=satelliteAltAz.obstime))
    return SkyCoord(res)


class TLEtoKeplerConverter:
    def __init__(self, tle: Path, objectID: str):
        self.tle = tle
        self.objectID = ObjectID(objectID)
        self.lines0 = []
        self.lines1 = []
        self.lines2 = []

    def decide3leFunction(self):
        if self.objectID.norad or self.objectID.cospar:
            self._getSatFrom3le()
        elif self.objectID.population:
            self._getNSatsFrom3le(30)
        elif self.objectID.directTles:
            self._getAllSatsFrom3le()
    def _getSatFrom3le(self):
        with open(self.tle, 'r') as tle:
            while True:
                line0 = tle.readline()
                line1 = tle.readline()
                line2 = tle.readline()

                if line0 == '':
                    raise Exception('no such satellite')

                if self.objectID.id == line1.split()[2] or self.objectID.id == line2.split()[1]:
                    self.lines0.append(line0)
                    self.lines1.append(line1)
                    self.lines2.append(line2)
                    break
    def _getNSatsFrom3le(self, desiredLength):
        with open(self.tle, 'r') as tle:
            while len(self.lines0) <= desiredLength:
                line0 = tle.readline()
                line1 = tle.readline()
                line2 = tle.readline()
                # if line0 == '':
                #     raise Exception('no such satellite')

                if self.objectID.id in line0:
                    self.lines0.append(line0)
                    self.lines1.append(line1)
                    self.lines2.append(line2)

    def _getAllSatsFrom3le(self):
        with open(self.tle, 'r') as tle:
            text = tle.read().splitlines()
            for i in range(0, len(text), 3):
                self.lines0.append(text[i])
                self.lines1.append(text[i+1])
                self.lines2.append(text[i+2])

    def readLines(self):
        with open(self.tle, 'r') as file:
            text = file.read().splitlines()
            for i in range(0,len(text),3):
                self.lines0.append(text[i])
                self.lines1.append(text[i+1])
                self.lines2.append(text[i+2])

    def converter(self):
        self.decide3leFunction()
        keplerian_elements = []
        periods = []
        epochs = []
        for line0, line1, line2 in zip(self.lines0, self.lines1, self.lines2):
            obj = ObjectID(str(line2.split()[1]))
            e = float(f'0.{line2[26:33]}')
            i = float(line2[8:16])
            Omega = float(line2[17:25])
            omega = float(line2[34:42])
            M = float(line2[43:51])
            n = float(line2[52:63])
            n_rad_per_day = n * 2*np.pi
            if int(line1[18:20])>57:
                year = int(f'19{line1[18:20]}')
            else:
                year = int(f'20{line1[18:20]}')
            yearJD = Time(f'{year}-01-01T00:00:00', format='isot').jd
            epoch = Time(yearJD + float(f'{line1[20:32]}'), format='jd')

            a = (np.cbrt((c.G.value*c.M_earth.value)/np.power(n*2*np.pi/86400,2)))/1000
            keplerian_elements.append(KeplerianElements(a=a*1000,e=e,i=np.radians(i),Omega=np.radians(Omega),
                                                        omega=np.radians(omega), M0=np.radians(M),
                                                        epoch=epoch, n=n_rad_per_day, id=obj))
        return keplerian_elements

if __name__ == "__main__":
    from pathlib import Path
    tle = Path('/Users/matoz/Documents/Ephemeris/data/20230228/3le.txt')

    con = TLEtoKeplerConverter(tle,'')
    a = con.converter()
    print(len(a))

    # calc the dt - time since perigee


