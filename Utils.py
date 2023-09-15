from astropy.io import fits
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from astropy.time import Time
from typing import List, Dict, Optional, Tuple
from astropy.coordinates import AltAz, GCRS, SkyCoord

@dataclass_json
@dataclass
class KeplerianElements(object):
    a: float = 0 #Semimajor axis [m]
    e: float = 0 #Eccentricity [-]
    i: float = 0 #Inclination [rad]
    Omega: float = 0 #Longitude of the ascending node [rad]
    omega: float = 0 #Argument of pericenter  [rad]
    M0: float = 0 #Mean anomaly at epoch [rad]

@dataclass_json
@dataclass
class stateVector(object):
    r: np.array = np.zeros(3)
    v: np.array = np.zeros(3)


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

@dataclass_json
@dataclass
class ObjectID(object):
    id: str = ''
    cospar: bool = False
    norad: bool = False
    population: bool = False
    directTles: bool = False

    def __post_init__(self):
        if self.id[0:5].isdigit() and self.id[5].isalpha():
            self.cospar = True
        elif self.id.isdigit():
            self.norad = True
        elif self.id[0:5].isdigit()!=True and self.id != '':
            self.population = True
        elif self.id == '':
            self.directTles = True

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

