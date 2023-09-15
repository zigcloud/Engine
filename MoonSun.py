from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time, TimeDelta
import json
import numpy as np
from Utils import *
from astropy.utils.iers import conf

conf.auto_max_age = None
class MoonSunGenerator:
    def __init__(self, observerLocation: EarthLocation, TimeStartIsot: str, TimeEndIsot: str, TimeStep: float):
        self.observer = observerLocation
        self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
        self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
        self.stepTime = TimeDelta(TimeStep, format='sec')

    def _getTimeArray(self):
        duration = np.ceil((self.endTime - self.startTime).jd*86400)
        nSteps = int(duration/self.stepTime.sec)+1
        print(duration,nSteps)
        self.timeArray = Time([(self.startTime + i*self.stepTime).isot for i in range(nSteps)], format='isot')

    def sun(self, loc, time):
        self.sunaltaz = get_body("sun",time).transform_to(AltAz(location=loc, obstime=time))
        self.sunradec = ConvertAltAzToRADEC(self.sunaltaz)

    def moon(self, loc, time):
        self.moonaltaz = get_body("moon",time).transform_to(AltAz(location=loc, obstime=time))
        self.moonradec = ConvertAltAzToRADEC(self.moonaltaz)

    def saveMoonSun(self):
        self._getTimeArray()
        self.sun(self.observer,self.timeArray)
        self.moon(self.observer,self.timeArray)
        keywords = ['mjd', 'isot', 'az_sun', 'alt_sun', 'az_moon', 'alt_moon', 'RA_t_sun', 'DE_t_sun',
                    'RA_t_moon', 'DE_t_moon']
        data = np.array([self.timeArray.mjd, self.timeArray.isot, self.sunaltaz.az.deg, self.sunaltaz.alt.deg,
                         self.moonaltaz.az.deg, self.moonaltaz.alt.deg, self.sunradec.ra.deg, self.sunradec.dec.deg,
                         self.moonradec.ra.deg, self.moonradec.dec.deg], dtype='O').transpose()

        return dict(zip(self.timeArray.mjd, [dict(zip(keywords, x)) for x in data]))


if __name__ == "__main__":
    import json
    import astropy.units as u
    import time as t
    with open('/Users/matoz/Documents/FMPH/PECS7-LightPolution/GitEngine/Stations.json','r') as js:
        data = json.load(js)

    obs = EarthLocation(lon=data['AGO']['Lon']*u.deg, lat=data['AGO']['Lat']*u.deg, height=data['AGO']['Alt']*u.m)
    ms = MoonSunGenerator(obs,'2023-01-01T00:00:00','2024-01-01T00:00:00',600)
    ti = t.time()
    print('computation started')
    result = {'AGO': ms.saveMoonSun()}
    with open('AGO_moonSun.json','w') as js:
        json.dump(result,js,indent=3)
    print('computation completed, time: ' + str(t.time() - ti))