from Utils import KeplerianElements, TLEtoKeplerConverter, stateVector, ObjectID
import numpy as np
from typing import List
from pathlib import Path
import astropy.units as u
import astropy.constants as c
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
from kepler import KeplerPropagator
from astropy.utils.iers import conf
from SGP4 import Sgp4Propagator

conf.auto_max_age = None

class Transformator:
    def __init__(self, observerLocation: EarthLocation, Kepler: List[KeplerianElements] or Path, objectID: str,
                 TimeStartIsot: str, TimeEndIsot: str, TimeStep: float, mode: str = 'Kepler'):

        if type(Kepler) == list:
            self.elements = np.array(Kepler)
        else:
            converter = TLEtoKeplerConverter(Kepler, objectID)
            self.elements = np.array(converter.converter())

        self.GM = c.GM_earth.value
        self.stateVector = []
        self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
        self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
        self.stepTime = TimeDelta(TimeStep, format='sec')
        self.obs = observerLocation
        if mode == 'Kepler':
            self.propagator = KeplerPropagator(observerLocation, Kepler, objectID, TimeStartIsot, TimeEndIsot, TimeStep)
        elif mode =='SGP4' and type(Kepler) != List:
            self.propagator = Sgp4Propagator(observerLocation, Kepler, objectID, TimeStartIsot, TimeEndIsot, TimeStep)
        else:
            exit('Bad mode selected!')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
