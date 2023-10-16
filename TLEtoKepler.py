from Utils import *
from astropy.time import Time
import astropy.constants as c

class TLEtoKeplerConverter:
    def __init__(self, tle: Path, objectID: str):
        self.tle = tle
        self.inputID = inputID(objectID)
        self.lines0 = []
        self.lines1 = []
        self.lines2 = []

    def decide3leFunction(self):
        if self.inputID.norad or self.inputID.cospar:
            self._getSatFrom3le()
        elif self.inputID.population:
            self._getNSatsFrom3le(30)
        elif self.inputID.directTles:
            self._getAllSatsFrom3le()
    def _getSatFrom3le(self):
        with open(self.tle, 'r') as tle:
            while True:
                line0 = tle.readline()
                line1 = tle.readline()
                line2 = tle.readline()

                if line0 == '':
                    raise Exception('no such satellite')

                if self.inputID.id == line1.split()[2] or self.inputID.id == line2.split()[1]:
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

                if self.inputID.id in line0:
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

    def converter(self):
        self.decide3leFunction()
        keplerian_elements = []

        for line0, line1, line2 in zip(self.lines0, self.lines1, self.lines2):
            obj = ObjectID(str(line2.split()[1]), line0.split()[1])
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

            a = (np.cbrt((c.G.value*c.M_earth.value)/np.power(n*2*np.pi/86400, 2)))/1000
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



