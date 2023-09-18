from Utils import *
import numpy as np
import astropy.units as u
from pathlib import Path
from sgp4 import omm
from sgp4.api import Satrec
from astropy.time import Time, TimeDelta
from datetime import datetime
from MoonSun import MoonSunGenerator
from astropy.coordinates import EarthLocation


class Sgp4Propagator:
    def __init__(self, observerLocation: EarthLocation, TlePath: Path, objID: str,
                 TimeStartIsot: str, TimeEndIsot: str, TimeStep: float):

        self.observer = observerLocation
        self.tle = TlePath
        self.objectID = ObjectID(objID)
        self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
        self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
        self.stepTime = TimeDelta(TimeStep, format='sec')
        self.satellites = []
        converter = TLEtoKeplerConverter(TlePath, objID)
        self.elements = np.array(converter.converter())
        self.stateVector = []
    def _pprint_stateVector(self):
        for i,obj in enumerate(self.stateVector):
            for j, step in enumerate(obj):
                r = (np.round(step.r,3) *u.m).to(u.km)
                v = (np.round(step.v,3) * (u.m/u.s)).to(u.km/u.s)
                print(f'Object ID position  : {self.elements[i].id.id}\n'
                      f'Time of the step    : {self.timeArray[j].isot}\n'
                      f'GCRS State vector   : x={np.round(r[0],3)} y={np.round(r[1],3)} z={np.round(r[2],3)}\n'
                      f'GCRS Velocity vector: x={np.round(v[0],3)} y={np.round(v[1],3)} z={np.round(v[2],3)}')

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

                # if line0 == '':
                #     raise Exception('no such satellite')

                if self.objectID == line1.split()[2] or self.objectID == line2.split()[1]:
                    self.satellites.append(Satrec.twoline2rv(line1, line2))
                    break

    def _getNSatsFrom3le(self, desiredLength):
        with open(self.tle, 'r') as tle:
            while len(self.satellites) <= desiredLength:
                line0 = tle.readline()
                line1 = tle.readline()
                line2 = tle.readline()

                if line0 == '':
                    raise Exception('no such satellite')

                if self.objectID.id in line0:
                    self.satellites.append(Satrec.twoline2rv(line1, line2))

    def _getAllSatsFrom3le(self):
        with open(self.tle, 'r') as tle:
            data = tle.read().splitlines()
            for i in range(0, len(data), 3):
                line1 = data[i+1]
                line2 = data[i+2]
                self.satellites.append(Satrec.twoline2rv(line1, line2))

    def _getTimeArray(self):
        duration = np.ceil((self.endTime - self.startTime).jd * 86400)
        nSteps = int(duration/self.stepTime.sec)+1
        self.timeArray = Time([(self.startTime + i*self.stepTime).isot for i in range(nSteps)], format='isot')

    def propagate(self):
        self._getTimeArray()
        self.decide3leFunction()

        moonSun = MoonSunGenerator(self.observer, self.startTime.isot, self.endTime.isot, self.stepTime.sec)
        satGCRS = [[sat.sgp4(obsTime.jd, 0) for obsTime in self.timeArray] for sat in self.satellites]
        self.stateVector = [[stateVector(r=np.array(sat[0][1])*1000, v=np.array(sat[0][2])*1000) for time in self.timeArray] for sat in satGCRS]
        self._pprint_stateVector()


if __name__ == "__main__":
    from pathlib import Path
    tle = Path('/Users/matoz/Documents/Ephemeris/data/20230228/selection.txt')

    import json
    import astropy.units as u
    import time as t
    with open('/Users/matoz/Documents/FMPH/PECS7-LightPolution/GitEngine/Stations.json','r') as js:
        data = json.load(js)

    obs = EarthLocation(lon=data['AGO']['Lon']*u.deg, lat=data['AGO']['Lat']*u.deg, height=data['AGO']['Alt']*u.m)
    ti = t.time()
    print('computation started')
    sP = Sgp4Propagator(obs, Path('/Users/matoz/Documents/Ephemeris/data/20230228/selection.txt'),'',
                            '2023-09-15T10:00:00', '2023-09-15T10:10:00', 600)

    sP.propagate()
    a = sP.stateVector
    print('computation completed, time: ' + str(t.time() - ti))
