# tests basic catenary function

import pytest
from numpy.testing import assert_allclose
from wisdem.moorpy.Catenary import catenary

# inputs of:  x, z, L, EA, w, CB, HF0, VF0
indata = [
    [400, 200, 500.0, 7510000000000.0, 800.0, 5.0, 0, 0],
    [400, 200, 500.0, 7510000000000.0, 800.0, 0.0, 0, 0],
    [400, 200, 500.0, 7510000000000.0, 800.0, 0.1, 0, 0],
    [400, 200, 500.0, 7510000000000.0, 200.0, -372.7, 0, 0],
    [89.9, 59.2, 130.0, 751000000.0, 881.05, -372.7, 0, 0],
    [
        37.96888656874307,
        20.49078283711694,
        100.0,
        751000000.0,
        -881.0549577007893,
        -1245.2679469540894,
        63442.20077641379,
        -27995.71383270186,
    ],
]

# desired results of:  fAH, fAV, fBH, fBV, LBot
desired = [
    [0.0, 0.0, -96643.43501616362, -237751.75997440016, 202.81030003199982],
    [96643.42815934008, 0.0, -96643.42815934008, -237751.75535996316, 202.81030580004602],
    [80418.60434855078, 0.0, -96643.42877136687, -237751.75577183915, 202.81030528520108],
    [43694.580989249596, -22365.599350216216, -43694.580989249596, -77634.40064978378, 0],
    [31373.229006023103, -26650.341270116318, -31373.229006023103, -87886.15872988368, 0],
    [6428.437434537766, 53178.729367882406, -6428.437434537766, 34926.76640219652, 0],
]


@pytest.mark.parametrize("index", range(len(indata)))
def test_catenary_solutions(index):
    """Run each of the test parameter sets with the catenary function and compare results to expected values."""

    ins = indata[index]

    (fAH, fAV, fBH, fBV, info) = catenary(*ins[:5], CB=ins[5], HF0=ins[6], VF0=ins[7], Tol=0.0001, MaxIter=50, plots=3)

    print(f"ProfileType is {info['ProfileType']}")
    assert_allclose([fAH, fAV, fBH, fBV, info["LBot"]], desired[index], rtol=1e-05, atol=0, verbose=True)


def test_catenary_symmetricU():
    """Tests the U shaped line with seabed contact against a simulation of half the line"""

    (fAH1, fAV1, fBH1, fBV1, info1) = catenary(50, 20, 65, 1e12, 100.0, CB=0, Tol=0.00001, MaxIter=50)
    (fAHU, fAVU, fBHU, fBVU, infoU) = catenary(100, 0, 130, 1e12, 100.0, CB=-20, Tol=0.00001, MaxIter=50)

    assert_allclose([fAHU, fAVU, fBHU, fBVU], [-fBH1, fBV1, fBH1, fBV1], rtol=1e-05, atol=0, verbose=True)


if __name__ == "__main__":
    for i in range(len(indata)):
        test_catenary_solutions(i)

        # catenary(0.007335040615956245, 46.969250518704726, 100.0, 257826627.22942558, 512.1255141001664, CB=-532.0307494812953, HF0=2169047.825684437, VF0=1165782.713912318, Tol=2.0000000000000003e-06, MaxIter=50, plots=1)
