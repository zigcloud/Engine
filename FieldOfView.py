from Utils import *
from astropy.time import Time


class FieldOfView:
    def __init__(self, width: float, height: float, ra: float, dec: float, population: Table,
                 verbose: bool=False):
        self.width = width
        self.height = height
        self.center = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        self.population = population
        self.verbose = verbose


    def addSkyCoordToPopulation(self):
        self.population['Coords'] = SkyCoord(ra=self.population['RA'], dec=self.population['DE'],
                                             obstime=Time(self.population['MJD'], format='mjd', scale='utc'))

    def determineCircularFoV(self):
        self.circularFoVRiadus = np.sqrt((self.width * self.height)/np.pi)

    def findObjectsInFov(self):
        self.determineCircularFoV()
        self.addSkyCoordToPopulation()
        self.population['Separation'] = self.population['Coords'].separation(self.center)
        return self.population[self.population['Separation'] <= self.circularFoVRiadus]

    def prettyPrintOutput(self):
        visibleObjects = self.findObjectsInFov()
        epochs = list(set(visibleObjects['MJD']))

        for epoch in epochs:
            mask = visibleObjects['MJD']==epoch
            group = visibleObjects[mask]
            print(f'At epoch {Time(epoch, format="mjd").isot} were visible objects:\n'
                  f'NORADs: {", ".join([i[0] for i in group.iterrows()])}')

            if self.verbose:
                group.pprint_all()

if __name__ == '__main__':
    from astropy.table import Table

    tbl = Table(names=outputTablenames, dtype=outputTabletypes, units=outputTableunits)

    with open('example.txt', 'r') as outputTbl:
        data = outputTbl.read().splitlines()
        for line in data:
            if line.startswith('#')!= True and len(line)>0:
                tbl.add_row(line.split('\t'))

    teleskop = FieldOfView(width=5, height=5, ra=0, dec=0, population=tbl, verbose=True)
    teleskop.prettyPrintOutput()