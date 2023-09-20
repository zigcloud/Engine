from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time, TimeDelta
import json
import numpy as np
from Utils import *
from astropy.utils.iers import conf

conf.auto_max_age = None
class MoonSunGenerator:
    def __init__(self, siteName: str, observerLocation: EarthLocation, timeArray = None,
                 TimeStartIsot: str= None, TimeEndIsot: str= None, TimeStep: float= None):
        self.site = siteName
        self.observer = observerLocation
        if timeArray is not None:
            self.timeArray = timeArray
        else:
            self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
            self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
            self.stepTime = TimeDelta(TimeStep, format='sec')
            self._getTimeArray()

    def _getTimeArray(self):
        duration = np.ceil((self.endTime - self.startTime).jd*86400)
        nSteps = int(duration/self.stepTime.sec)+2
        self.timeArray = Time([(self.startTime + i*self.stepTime).isot for i in range(nSteps)], format='isot')

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
    def sun(self, loc, time):
        self.sunaltaz = get_body("sun",time).transform_to(AltAz(location=loc, obstime=time))
        self.sunradec = self.ConvertAltAzToRADEC(self.sunaltaz)

    def moon(self, loc, time):
        self.moonaltaz = get_body("moon",time).transform_to(AltAz(location=loc, obstime=time))
        self.moonradec = self.ConvertAltAzToRADEC(self.moonaltaz)

    def getMoonSun(self):
        self.sun(self.observer,self.timeArray)
        self.moon(self.observer,self.timeArray)
        keywords = ['mjd', 'isot', 'az_sun', 'alt_sun', 'az_moon', 'alt_moon', 'RA_t_sun', 'DE_t_sun',
                    'RA_t_moon', 'DE_t_moon']
        data = np.array([self.timeArray.mjd, self.timeArray.isot, self.sunaltaz.az.deg, self.sunaltaz.alt.deg,
                         self.moonaltaz.az.deg, self.moonaltaz.alt.deg, self.sunradec.ra.deg, self.sunradec.dec.deg,
                         self.moonradec.ra.deg, self.moonradec.dec.deg], dtype='O').transpose()

        moonSunJson = {self.site: dict(zip(self.timeArray.mjd, [dict(zip(keywords, x)) for x in data]))}


        return moonSunJson


if __name__ == "__main__":
    import json
    import astropy.units as u
    import time as t

    obs = EarthLocation(lon=17.2736306*u.deg, lat=48.372528*u.deg, height=536.1*u.m)
    ms = MoonSunGenerator(siteName='AGO', observerLocation=obs,TimeStartIsot='2023-01-01T00:00:00',
                          TimeEndIsot='2024-01-01T00:00:00',TimeStep=600)
    ti = t.time()
    print('computation started')
    result = ms.getMoonSun()
    print(result)
    # with open('AGO_moonSun.json','w') as js:
    #     json.dump(result,js,indent=3)
    print('computation completed, time: ' + str(t.time() - ti))