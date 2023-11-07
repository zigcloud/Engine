from dataclasses_json import dataclass_json
from astropy.table import Table
import astropy.units as u
from dataclasses import dataclass
import astropy.constants as c

@dataclass_json
@dataclass
class constant(object):
    Name: str = ""
    Value: float = 0.0
    Uncertainty: float = 0.0
    Unit: u.Unit = None

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

magSun = constant('Sun Magnitude in whole visible spectra at 1AU',-26.74, 0.0, u.mag)
LSun = constant('Sun Luminosity in whole visible spectra', c.L_sun.value, c.L_sun.uncertainty, c.L_sun.unit)
magSunV = constant('Sun Magnitude in V Johnsson filter',-25.036, 0.0, u.mag)
LSunV = constant('Sun Luminosity in V Johnsson filter', 8.0088e+25, c.L_sun.uncertainty, c.L_sun.unit)
#
format_string = '{}\t{:.9f}\t{:.3e}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.3e}\t{}\t{:.5f}\t{:.5f}\n'
outputTablenames = ['ObjectID','MJD','Range','Phase angle','RA','DE',
                    'dRA','dDE','Length','Beta','A*rho', 'm_abs','m_app','Luminosity', 'Shadow', 'Lon', 'Lat']
outputTabletypes = [str, float, float, float,float,float,
                    float,float,float,float,float,float,float,float, bool, float, float]
outputTableunits = ['',u.day, u.m, u.deg, u.deg, u.deg,
                    u.arcsec / u.s, u.arcsec / u.s, u.arcsec, '--',u.m*u.m,u.mag,u.mag, u.W,'--', u.deg, u.deg]
phaseParamsTabNames = ["run_id", "norad", "name", "phase_min", "phase_max", "mjd_min", "mjd_max",
                       "Hejduk_med_beta", "Hejduk_med_AreaRo","Hejduk_med_AbsMag", "Hejduk_med_beta_sigma",
                       "Hejduk_med_AreaRo_sigma", "Hejduk_med_AbsMag_sigma", "Hejduk_med_no_of_points",
                       "Hejduk_predictionBand_upper_AreaRo", "Hejduk_predictionBand_lower_AreaRo",
                       "Hejduk_predictionBand_upper_AbsMag", "Hejduk_predictionBand_lower_AbsMag"]
phaseParamsTabTypes = [str, str, str, float, float, float, float, float, float, float, float, float, float, int, float, float, float, float]

print(LSunV, magSunV)