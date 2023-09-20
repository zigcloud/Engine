from Utils import *
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
from TLEtoKepler import TLEtoKeplerConverter

class KeplerPropagator:
    #TODO - Kepler Propagator is giving strange values not coresponding to the reality
    def __init__(self, site: str, observerLocation: EarthLocation, Kepler: List[KeplerianElements] or Path, objectID: str,
                 TimeStartIsot: str, TimeEndIsot: str, TimeStep: float, verbose: bool = False):
        self.site = site
        if type(Kepler) == list:
            self.elements = np.array(Kepler)
        else:
            converter = TLEtoKeplerConverter(Kepler, objectID)
            self.elements = np.array(converter.converter())
        #self.dt = Kepler.timeSincePerigee
        self.GM = c.GM_earth.value
        self.stateVector = []
        self.startTime = Time(TimeStartIsot, format='isot', scale='utc')
        self.endTime = Time(TimeEndIsot, format='isot', scale='utc')
        self.stepTime = TimeDelta(TimeStep, format='sec')
        self.obs = observerLocation
        self.verbose = verbose


    def _pprint(self):
        for obj in self.elements:
            print(f'{"="*80}\n'
                  f'Semimajor axis                 : {np.round(obj.a,6)} [m]\n'+
                  f'Eccentricity                   : {np.round(obj.e,6)} [-]\n'+
                  f'Inclination                    : {np.round(obj.i,6)} [rad]\n'+
                  f'Longitude of the ascending node: {np.round(obj.Omega,6)} [rad]\n'+
                  f'Argument of pericenter         : {np.round(obj.omega,6)} [rad]\n'+
                  f'Mean anomaly at epoch          : {np.round(obj.M0,6)} [rad]\n'
                  f'Gravitational constant         : {np.round(self.GM, 6)} [m3/s2]\n')
            if obj.timeSincePerigee is not None:
                print(f'Julian days since perigee      : {np.round(obj.timeSincePerigee, 3)} [days]\n{"="*80}')
            if obj.epoch is not None and obj.n is not None and obj.id is not None:
                print(f'Epoch of the elements validity : {obj.epoch.isot}\n'+
                      f'Mean motion of the satellite   : {np.round(obj.n,6)} [rad/s]\n'
                      f'Object NORAD ID                : {obj.id.id}\n {"="*80}')

    def _pprint_stateVector(self):
        for i,obj in enumerate(self.stateVector):
            for j, step in enumerate(obj):
                r = (np.round(step.r,3) * u.m).to(u.km)
                v = (np.round(step.v,3) * (u.m/u.s)).to(u.km/u.s)
                print(f'Object ID position  : {self.elements[i].id.id}\n'
                      f'Time of the step    : {self.timeArray[j].isot}\n'
                      f'GCRS State vector   : x={np.round(r[0],3)} y={np.round(r[1],3)} z={np.round(r[2],3)}\n'
                      f'GCRS Velocity vector: x={np.round(v[0],3)} y={np.round(v[1],3)} z={np.round(v[2],3)}')

    def _checkElements(self):
        for obj in self.elements:
            assert (0 <= obj.e <= 1), "Wrong Eccentricity input"
            assert (0 <= abs(obj.omega) <= 2*np.pi), "Wrong Longitude of the ascending node input"
            assert (0 <= obj.Omega <= 2*np.pi), "Wrong Argument of pericenter input"
            assert (0 <= obj.i <= 2*np.pi), "Wrong Inclination input"

    def _getTimeArray(self):
        duration = np.ceil((self.endTime - self.startTime).jd * 86400)
        nSteps = int(duration/self.stepTime.sec)+2
        self.timeArray = Time([(self.startTime + i*self.stepTime).isot for i in range(nSteps)], format='isot')


    def _getEccAnomaly(self, keplerElements: KeplerianElements, M):
        maxit = 15
        eps = 10e-9

        i=0
        M = M % np.pi/2

        if keplerElements.e<0.8:
            E=M
        else:
            E=np.pi;

        f = E - keplerElements.e * np.sin(E) - M
        E = E - f / (1.0 - keplerElements.e * np.cos(E))
        while abs(f) > eps:
            i+=1
            if i == maxit:
                print('Maxit reached!')
                break
            else:
                f = E - keplerElements.e * np.sin(E) - M
                E = E - f / (1.0 - keplerElements.e * np.cos(E))
        return E

    def _getStateVector(self, keplerElements: KeplerianElements, time: Time):
        if keplerElements.timeSincePerigee is not None and keplerElements.timeSincePerigee == 0:
            M = keplerElements.M0
        else:
            if keplerElements.n is not None:
                M = keplerElements.M0 + keplerElements.n * (keplerElements.epoch.jd - time.jd)
            else:
                n = np.sqrt(self.GM / (keplerElements.a * keplerElements.a*keplerElements.a))
                M = keplerElements.M0 + n * keplerElements.timeSincePerigee

        M = M % np.pi
        E = self._getEccAnomaly(keplerElements, M)

        cosE = np.cos(E)
        sinE = np.sin(E)

        # Perifocal coordinates
        fac = (1.0 + keplerElements.e) * (1.0 - keplerElements.e)

        R = keplerElements.a * (1.0 - keplerElements.e * cosE) # Distance
        V = np.sqrt(self.GM * keplerElements.a) / R # Velocity

        state = stateVector
        state.r = [keplerElements.a * (cosE - keplerElements.e), -keplerElements.a * np.sqrt(1-keplerElements.e**2) * sinE, 0]
        state.v = [V * sinE, V * fac * cosE, 0]

        sideProd = np.matmul(RotationMatrix.Rz(keplerElements.Omega), RotationMatrix.Rx(keplerElements.i))
        PQW = np.matmul(sideProd, RotationMatrix.Rz(keplerElements.omega))

        state.r = PQW.dot(state.r).tolist()[0]
        state.v = PQW.dot(state.v).tolist()[0]

        # posvelObs = self.obs.get_gcrs_posvel(time)
        # state.r = [state.r[0] + posvelObs[0].y.value, state.r[1] - posvelObs[0].x.value,
        #            state.r[2]]
        # state.v = [state.v[0] + posvelObs[1].y.value, state.v[1] + posvelObs[1].x.value,
        #            state.v[2]]


        return state

    def propagate(self):
        self._getTimeArray()
        for obj in self.elements:
            objState = []
            for time in self.timeArray:
                s = self._getStateVector(obj, time)
                objState.append(stateVector(s.r,s.v))
        # state = [[self._getStateVector(obj, time) for time in self.timeArray] for obj in self.elements]
            self.stateVector.append(objState)
        if self.verbose:
            self._pprint_stateVector()

if __name__ == "__main__":
    #TEST
    import astropy.constants as c

    omega = np.radians(131.821109)
    Omega = np.radians(32.702746)
    incl = np.radians(64.850138)
    e = 0.92813
    a = 11.747335 * c.au.value
    M = 0.000000
    dt = 0.0
    GM_Earth = 398600.4415e+9
    #

    obs = EarthLocation(lon=17.2736306*u.deg, lat=48.372528*u.deg, height=536.1*u.m)

    # test = KeplerPropagator(obs,[KeplerianElements(a,e,incl,Omega,omega,M,timeSincePerigee=dt)],
    #                         c.GM_sun.value,'2023-09-15T10:00:00', '2023-09-17T10:10:00', 600)
    # test.propagate()
    # test.pp


    # test._pprint()
    # test._getStateVector()
    # TEST results shall be
    #State vector x=-9.248159788726267E7km, y=-1.1841295634767203E7km, z=8.520148585292272E7km
    #Velocity vector x=-21334.850862085492m/s, y=-28856.374856938717m/s, z=-27168.28164284154m/s

    A = 20828674.192445
    E = 0.6733951495
    I = np.radians(28.3044368)
    NODE = np.radians(150.5656450)
    PER = np.radians(-26.3798536)
    TPER = -12998.4727669
    M = 0
    # test = KeplerPropagator(KeplerianElements(A,E,I,NODE,PER,M),c.GM_earth.value, TPER)

    import time as t
    ti = t.time()
    print('computation started')
    test = KeplerPropagator('AGO',obs, Path('/Users/matoz/Documents/Ephemeris/data/20230228/selection.txt'),'',
                            '2023-09-15T10:00:00', '2023-09-17T10:10:00', 600)
    # test._pprint()
    test.propagate()

    print('computation completed, time: ' + str(t.time() - ti))
