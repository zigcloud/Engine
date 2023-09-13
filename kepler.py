from Utils import *
import numpy as np
import astropy.units as u
class KeplerPropagator:
    def __init__(self, Kepler: KeplerianElements, GM: float, dt: float):
        self.elements = Kepler
        self.dt = dt
        self.GM = GM
        self.stateVector = stateVector



    def _pprint(self):
        print(f'Semimajor axis                 : {np.round(self.elements.a,3)} [m]\n'+
              f'Eccentricity                   : {np.round(self.elements.e,3)} [-]\n'+
              f'Inclination                    : {np.round(self.elements.i,3)} [rad]\n'+
              f'Longitude of the ascending node: {np.round(self.elements.Omega,3)} [rad]\n'+
              f'Argument of pericenter         : {np.round(self.elements.omega,3)} [rad]\n'+
              f'Mean anomaly at epoch          : {np.round(self.elements.M0,3)} [rad]\n\n'
              f'Gravitational constant         : {np.round(self.GM, 3)} [m3/s2]\n' +
              f'Julian days since perigee      : {np.round(self.dt, 3)} [days]\n')

    def _pprint_stateVector(self):
        r = np.round(self.stateVector.r,3) * u.m
        v = np.round(self.stateVector.v,3) * u.m/u.s
        print(f'GCRS State vector   : x={np.round(r[0][0],3)} y={np.round(r[0][1],3)} z={np.round(r[0][2],3)}\n'
              f'GCRS Velocity vector: x={np.round(v[0][0],3)} y={np.round(v[0][1],3)} z={np.round(v[0][2],3)}')

    def _checkElements(self):
        assert (0 <= self.elements.e <= 1), "Wrong Eccentricity input"
        assert (0 <= self.elements.omega <= 2*np.pi), "Wrong Longitude of the ascending node input"
        assert (0 <= self.elements.Omega <= 2*np.pi), "Wrong Argument of pericenter input"
        assert (0 <= self.elements.i <= 2*np.pi), "Wrong Inclination input"



    def _getEccAnomaly(self, M):
        maxit = 15
        eps = 10e-9

        i=0
        M = M % 2*np.pi
        if self.elements.e < 0.8:
            E=M
        else:
            E=np.pi
        f = E - self.elements.e * np.sin(E) - M
        E = E - f / (1.0 - self.elements.e * np.cos(E))
        while abs(f) > eps:
            i+=1
            if i == maxit:
                print('Maxit reached!')
                break
            else:
                f = E - self.elements.e * np.sin(E) - M
                E = E - f / (1.0 - self.elements.e * np.cos(E))
        return E

    def _getStateVector(self):
        self._checkElements()
        if self.dt == 0:
            M = self.elements.M0
        else:
            n = np.sqrt(self.GM / (self.elements.a *self.elements.a*self.elements.a))
            M = self.elements.M0 + n * self.dt

        M = M % np.pi
        E = self._getEccAnomaly(M)

        cosE = np.cos(E)
        sinE = np.sin(E)

        # Perifocal coordinates
        fac = np.sqrt((1.0 - self.elements.e) * (1.0 + self.elements.e))

        R = self.elements.a * (1.0 - self.elements.e * cosE) # Distance
        V = np.sqrt(self.GM * self.elements.a) / R # Velocity


        self.stateVector.r[0] = self.elements.a * (cosE - self.elements.e)
        self.stateVector.r[1] = self.elements.a * fac * sinE
        self.stateVector.v[0] = -V * sinE
        self.stateVector.v[1] = V * fac * cosE

        sideProd = np.matmul(RotationMatrix.Rz(self.elements.Omega), RotationMatrix.Rx(self.elements.i))
        PQW = np.matmul(sideProd, RotationMatrix.Rz(self.elements.omega))

        stateVector.r = np.dot(PQW, self.stateVector.r)
        stateVector.v = np.dot(PQW, self.stateVector.v)

        self._pprint_stateVector()



if __name__ == "__main__":
    import astropy.constants as c

    omega = np.radians(131.821109)
    Omega = np.radians(32.702746)
    incl = np.radians(64.850138)
    e = 0.92813
    a = 11.747335 * c.au.value
    M = 0.000000
    dt = 0.0
    GM_Earth = 398600.4415e+9

    test = KeplerPropagator(KeplerianElements(a,e,incl,Omega,omega,M),c.GM_sun.value, dt)
    test._pprint()
    test._getStateVector()
