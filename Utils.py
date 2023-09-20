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

def Hejduk_F1F2_beta(phaseAngle, a_ro, beta):
    x = np.array(phaseAngle)
    F_diff = 2/(3*np.pi*np.pi)*((np.pi-x)*np.cos(x)+np.sin(x))
    F_spec = 1/(4*np.pi)
    return -26.74 - 2.5*np.log10(a_ro*(beta*F_diff+(1-beta)*F_spec)) + 5*np.log10(149597871000.0)

