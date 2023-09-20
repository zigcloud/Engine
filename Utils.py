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
import json

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
outputTablenames = ['ObjectID','MJD','Range','Phase angle','RA','DE',
                    'dRA','dDE','Length','Beta','A*rho', 'm_abs','m_app','Luminosity', 'Shadow', 'Lon', 'Lat']
outputTabletypes = [str, float, float, float,float,float,
                    float,float,float,float,float,float,float,float, bool, float, float]
outputTableunits = ['',u.day, u.m, u.deg, u.deg, u.deg,
                    u.arcsec / u.s, u.arcsec / u.s, u.arcsec, '--','--','--',u.mag,u.mag, '--', u.deg, u.deg]
outputTableround = [-1, 9, 5,5,5,5,5,5,5,5,5,5,5,5,-1,5,5]

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

def GetLuminosity(timeArray):
    #only a dummy function
    return [0.0 for i in range(len(timeArray))]

def getTableFromJson(jsonFile):
    phaseParamsTab = Table(names=phaseParamsTabNames, dtype=phaseParamsTabTypes)
    with open(jsonFile, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        phaseParamsTab.add_row(value)

    return phaseParamsTab

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
