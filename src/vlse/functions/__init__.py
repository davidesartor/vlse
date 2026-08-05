"""The 48 SFU test functions, grouped as the site groups them."""

from vlse.functions.base import TestFunction
from vlse.functions.many_local_minima import (
    Ackley,
    Bukin6,
    CrossInTray,
    DropWave,
    EggHolder,
    GramacyLee,
    Griewank,
    HolderTable,
    Langermann,
    Levy,
    Levy13,
    Rastrigin,
    Schaffer2,
    Schaffer4,
    Schwefel,
    Shubert,
)
from vlse.functions.bowl_shaped import (
    Bohachevsky,
    Perm0,
    RotatedHyperEllipsoid,
    Sphere,
    SumPowers,
    SumSquares,
    Trid,
)
from vlse.functions.plate_shaped import (
    Booth,
    Matyas,
    McCormick,
    PowerSum,
    Zakharov,
)
from vlse.functions.valley_shaped import (
    Camel3,
    Camel6,
    DixonPrice,
    Rosenbrock,
)
from vlse.functions.steep_drops import (
    DeJong5,
    Easom,
    Michalewicz,
)
from vlse.functions.other import (
    Beale,
    Branin,
    Colville,
    Forrester,
    ForresterLowFidelity,
    GoldsteinPrice,
    Hartmann3,
    Hartmann4,
    Hartmann6,
    Perm,
    Powell,
    Shekel,
    StyblinskiTang,
)
