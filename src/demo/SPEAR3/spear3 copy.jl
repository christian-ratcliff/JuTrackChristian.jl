function spear3()
RF = RFCA(name="RF", len=0.0000000000, volt=3200000.0000000000, h=372.0000000000, freq=4.7630000575e+08, energy=3.0000000000e+09)
DA1A = DRIFT(name="DA1A", len=3.6792386000)
BPM = MARKER(name="BPM")
DA1B = DRIFT(name="DA1B", len=0.1240666500)
QDX = KQUAD(name="QDX", len=0.3533895000, k1=-1.3864672452)
DA2A = DRIFT(name="DA2A", len=0.1153052500)
COR = CORRECTOR(name="COR", len=0.1500000000, xkick=0.0000000000, ykick=0.0000000000)
DA2B = DRIFT(name="DA2B", len=0.1177344500)
QFX = KQUAD(name="QFX", len=0.6105311000, k1=1.5731962784)
DA3A = DRIFT(name="DA3A", len=0.2088992500)
DA3B = DRIFT(name="DA3B", len=0.0541404500)
QDY = KQUAD(name="QDY", len=0.3533895000, k1=-0.4606409306)
DM4 = DRIFT(name="DM4", len=0.2158457200)
B34 = SBEND(name="B34", len=1.1432900000, angle=0.1385996759, e1=0.0692998379, e2=6.9299837947e-02, PolynomB=[0.0, -3.1537858000e-01, 0.0, 0.0])
DA5A = DRIFT(name="DA5A", len=0.1139774700)
DA5B = DRIFT(name="DA5B", len=0.1085630000)
SDM = KSEXT(name="SDM", len=0.2100000000, k2=-17.0000000000)
DA6A = DRIFT(name="DA6A", len=0.1266000000)
DA6B = DRIFT(name="DA6B", len=0.9047682800)
SFM = KSEXT(name="SFM", len=0.2100000000, k2=15.0000000000)
DA7A = DRIFT(name="DA7A", len=0.1106966000)
DA7B = DRIFT(name="DA7B", len=0.0631132500)
QFY = KQUAD(name="QFY", len=0.5123803000, k1=1.4814937098)
DM7 = DRIFT(name="DM7", len=0.1738098500)
DA6C = DRIFT(name="DA6C", len=0.0960000000)
DA6D = DRIFT(name="DA6D", len=0.9353700000)
DA5C = DRIFT(name="DA5C", len=0.0518450000)
DA5D = DRIFT(name="DA5D", len=0.1706954700)
DA8A = DRIFT(name="DA8A", len=0.3373594700)
DA8B = DRIFT(name="DA8B", len=0.1284862500)
QDZ = KQUAD(name="QDZ", len=0.3533895000, k1=-0.8782239377)
DA9A = DRIFT(name="DA9A", len=0.1093052500)
DA9B = DRIFT(name="DA9B", len=0.1373052500)
QFZ = KQUAD(name="QFZ", len=0.3533895000, k1=1.4279020070)
DA10A = DRIFT(name="DA10A", len=0.1239396500)
DA10B = DRIFT(name="DA10B", len=3.1459370000)
DC1A = DRIFT(name="DC1A", len=1.4059340000)
DC1B = DRIFT(name="DC1B", len=0.1240412500)
QF = KQUAD(name="QF", len=0.3533895000, k1=1.7686729041)
DC2A = DRIFT(name="DC2A", len=0.1157652500)
DC2B = DRIFT(name="DC2B", len=0.1158104500)
QD = KQUAD(name="QD", len=0.1634591000, k1=-1.5424742304)
DC3A = DRIFT(name="DC3A", len=0.0532206500)
DC3B = DRIFT(name="DC3B", len=0.1636824700)
BND = SBEND(name="BND", len=1.5048000000, angle=0.1847995700, e1=0.0923997850, e2=9.2399785000e-02, PolynomB=[0.0, -3.1537858000e-01, 0.0, 0.0])
DC4A = DRIFT(name="DC4A", len=0.1592146700)
DC4B = DRIFT(name="DC4B", len=0.0444180000)
SD = KSEXT(name="SD", len=0.2500000000, k2=-38.8015300000)
DC5A = DRIFT(name="DC5A", len=0.0905800000)
DC5B = DRIFT(name="DC5B", len=0.3613900000)
SF = KSEXT(name="SF", len=0.2100000000, k2=32.0477093000)
DC6A = DRIFT(name="DC6A", len=0.1106460000)
DC6B = DRIFT(name="DC6B", len=0.0631658500)
QFC = KQUAD(name="QFC", len=0.5123803000, k1=1.7486408311)
DC5C = DRIFT(name="DC5C", len=0.0958400000)
DC5D = DRIFT(name="DC5D", len=0.3561300000)
DC2C = DRIFT(name="DC2C", len=0.1021004500)
DC2D = DRIFT(name="DC2D", len=0.1294752500)
DI1 = DRIFT(name="DI1", len=0.9235741000)
K1 = CORRECTOR(name="K1", len=1.2000000000, xkick=0.0000000000, ykick=0.0000000000)
DI2 = DRIFT(name="DI2", len=0.6882939000)
DI3 = DRIFT(name="DI3", len=0.6834939000)
K2 = CORRECTOR(name="K2", len=0.6000000000, xkick=0.0000000000, ykick=0.0000000000)
DI4 = DRIFT(name="DI4", len=0.1224401000)
DI5 = DRIFT(name="DI5", len=1.2401300000)
DI6 = DRIFT(name="DI6", len=0.1658040000)
K3 = CORRECTOR(name="K3", len=1.2000000000, xkick=0.0000000000, ykick=0.0000000000)
DB10B = DRIFT(name="DB10B", len=3.1458354000)
DB10A = DRIFT(name="DB10A", len=0.1240412500)
DB9B = DRIFT(name="DB9B", len=0.1233052500)
DB9A = DRIFT(name="DB9A", len=0.1233052500)
DB8B = DRIFT(name="DB8B", len=0.1385954500)
DB8A = DRIFT(name="DB8A", len=0.3272502700)
DB5D = DRIFT(name="DB5D", len=0.1139772700)
DB5C = DRIFT(name="DB5C", len=0.1085632000)
DB6D = DRIFT(name="DB6D", len=0.1270000000)
DB6C = DRIFT(name="DB6C", len=0.9043700000)
DB7B = DRIFT(name="DB7B", len=0.1106968000)
DB7A = DRIFT(name="DB7A", len=0.0631130500)
DB6B = DRIFT(name="DB6B", len=0.0939985200)
DB6A = DRIFT(name="DB6A", len=0.9373700000)
DB5B = DRIFT(name="DB5B", len=0.0518450000)
DB5A = DRIFT(name="DB5A", len=0.1706954700)
DB3B = DRIFT(name="DB3B", len=0.1308128500)
DB3A = DRIFT(name="DB3A", len=0.1322268500)
DB2B = DRIFT(name="DB2B", len=0.1172344500)
DB2A = DRIFT(name="DB2A", len=0.1158052500)
DB1B = DRIFT(name="DB1B", len=0.0562232500)
DB1A = DRIFT(name="DB1A", len=3.7470820000)
line = Lattice(nothing)
append!(line, RF)
append!(line, DA1A)
append!(line, BPM)
append!(line, DA1B)
append!(line, QDX)
append!(line, DA2A)
append!(line, COR)
append!(line, DA2B)
append!(line, QFX)
append!(line, DA3A)
append!(line, BPM)
append!(line, DA3B)
append!(line, QDY)
append!(line, DM4)
append!(line, B34)
append!(line, DA5A)
append!(line, BPM)
append!(line, DA5B)
append!(line, SDM)
append!(line, DA6A)
append!(line, COR)
append!(line, DA6B)
append!(line, SFM)
append!(line, DA7A)
append!(line, BPM)
append!(line, DA7B)
append!(line, QFY)
append!(line, DM7)
append!(line, SFM)
append!(line, DA6C)
append!(line, COR)
append!(line, DA6D)
append!(line, SDM)
append!(line, DA5C)
append!(line, BPM)
append!(line, DA5D)
append!(line, B34)
append!(line, DA8A)
append!(line, BPM)
append!(line, DA8B)
append!(line, QDZ)
append!(line, DA9A)
append!(line, COR)
append!(line, DA9B)
append!(line, QFZ)
append!(line, DA10A)
append!(line, BPM)
append!(line, DA10B)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DI1)
append!(line, K1)
append!(line, DI2)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DI3)
append!(line, K2)
append!(line, DI4)
append!(line, DI5)
append!(line, DI6)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DI2)
append!(line, K3)
append!(line, DI1)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DB10B)
append!(line, BPM)
append!(line, DB10A)
append!(line, QFZ)
append!(line, DB9B)
append!(line, COR)
append!(line, DB9A)
append!(line, QDZ)
append!(line, DB8B)
append!(line, BPM)
append!(line, DB8A)
append!(line, B34)
append!(line, DB5D)
append!(line, BPM)
append!(line, DB5C)
append!(line, SDM)
append!(line, DB6D)
append!(line, COR)
append!(line, DB6C)
append!(line, SFM)
append!(line, DB7B)
append!(line, BPM)
append!(line, DB7A)
append!(line, QFY)
append!(line, DM7)
append!(line, SFM)
append!(line, DB6B)
append!(line, COR)
append!(line, DB6A)
append!(line, SDM)
append!(line, DB5B)
append!(line, BPM)
append!(line, DB5A)
append!(line, B34)
append!(line, DM4)
append!(line, QDY)
append!(line, DB3B)
append!(line, BPM)
append!(line, DB3A)
append!(line, QFX)
append!(line, DB2B)
append!(line, COR)
append!(line, DB2A)
append!(line, QDX)
append!(line, DB1B)
append!(line, BPM)
append!(line, DB1A)
append!(line, DA1A)
append!(line, BPM)
append!(line, DA1B)
append!(line, QDX)
append!(line, DA2A)
append!(line, COR)
append!(line, DA2B)
append!(line, QFX)
append!(line, DA3A)
append!(line, BPM)
append!(line, DA3B)
append!(line, QDY)
append!(line, DM4)
append!(line, B34)
append!(line, DA5A)
append!(line, BPM)
append!(line, DA5B)
append!(line, SDM)
append!(line, DA6A)
append!(line, COR)
append!(line, DA6B)
append!(line, SFM)
append!(line, DA7A)
append!(line, BPM)
append!(line, DA7B)
append!(line, QFY)
append!(line, DM7)
append!(line, SFM)
append!(line, DA6C)
append!(line, COR)
append!(line, DA6D)
append!(line, SDM)
append!(line, DA5C)
append!(line, BPM)
append!(line, DA5D)
append!(line, B34)
append!(line, DA8A)
append!(line, BPM)
append!(line, DA8B)
append!(line, QDZ)
append!(line, DA9A)
append!(line, COR)
append!(line, DA9B)
append!(line, QFZ)
append!(line, DA10A)
append!(line, BPM)
append!(line, DA10B)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DC1A)
append!(line, BPM)
append!(line, DC1B)
append!(line, QF)
append!(line, DC2A)
append!(line, COR)
append!(line, DC2B)
append!(line, QD)
append!(line, DC3A)
append!(line, BPM)
append!(line, DC3B)
append!(line, BND)
append!(line, DC4A)
append!(line, BPM)
append!(line, DC4B)
append!(line, SD)
append!(line, DC5A)
append!(line, COR)
append!(line, DC5B)
append!(line, SF)
append!(line, DC6A)
append!(line, BPM)
append!(line, DC6B)
append!(line, QFC)
append!(line, DC6B)
append!(line, DC6A)
append!(line, SF)
append!(line, DC5C)
append!(line, COR)
append!(line, DC5D)
append!(line, SD)
append!(line, DC4B)
append!(line, BPM)
append!(line, DC4A)
append!(line, BND)
append!(line, DC3B)
append!(line, DC3A)
append!(line, QD)
append!(line, DC2C)
append!(line, COR)
append!(line, DC2D)
append!(line, QF)
append!(line, DC1B)
append!(line, BPM)
append!(line, DC1A)
append!(line, DB10B)
append!(line, BPM)
append!(line, DB10A)
append!(line, QFZ)
append!(line, DB9B)
append!(line, COR)
append!(line, DB9A)
append!(line, QDZ)
append!(line, DB8B)
append!(line, BPM)
append!(line, DB8A)
append!(line, B34)
append!(line, DB5D)
append!(line, BPM)
append!(line, DB5C)
append!(line, SDM)
append!(line, DB6D)
append!(line, COR)
append!(line, DB6C)
append!(line, SFM)
append!(line, DB7B)
append!(line, BPM)
append!(line, DB7A)
append!(line, QFY)
append!(line, DM7)
append!(line, SFM)
append!(line, DB6B)
append!(line, COR)
append!(line, DB6A)
append!(line, SDM)
append!(line, DB5B)
append!(line, BPM)
append!(line, DB5A)
append!(line, B34)
append!(line, DM4)
append!(line, QDY)
append!(line, DB3B)
append!(line, BPM)
append!(line, DB3A)
append!(line, QFX)
append!(line, DB2B)
append!(line, COR)
append!(line, DB2A)
append!(line, QDX)
append!(line, DB1B)
append!(line, BPM)
append!(line, DB1A)
return line
end