#-------------------------------------------------------------------------
# Name:        SegIntersect.py
# Purpose: It Calculates Intersection of 2 segments in space. Also contains a simple 2-point distance calculator.
#
# Author:      RRD
#
# Created:     24/10/2012
# Copyright:   (c) rdamiani 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------


import numpy as np

def CalcDist(X1,X2):
    """This function returns the distance (norm) between the two points X1(x1,y1,z1) and X2(x2,y2,z2).
    X1 and X2 can be [3,n] arrays."""
    junk=X2-X1
    return np.sqrt(np.sum(junk**2,0))


def SegIntersect(A1, A2, B1, B2):
    """The function returns the intersection or the points of closest approach if lines are skewed.
    If lines are parallel, NaN is returned.
    INPUT:
        A1  -float(3,n), [x,y,z;nsegments] cordinates of 1st point(s) of 1st segment(s)
        A2  -float(3,n), [x,y,z;nsegments] cordinates of 2nd point(s) of 1st segment(s)
        B1  -float(3,n), [x,y,z;nsegments] cordinates of 1st point(s) of 2nd segment(s)
        B2  -float(3,n), [x,y,z;nsegments] cordinates of 2nd point(s) of 2nd segment(s)
    OUTPUT:
        A0  -float(3,n), [x,y,z;nsegments] coordinates of intersection point (=B0) or closet point to 2nd line on 1st segment,
        B0  -float(3,n), [x,y,z;nsegments] coordinates of intersection point (=A0) or closet point to 1st line on 2nd segment,
        OR  -NaN
    """

    #reshape A1..B2 in case they have 1 dimension only
    A1=A1.reshape(3,-1)
    A2=A2.reshape(3,-1)
    B1=B1.reshape(3,-1)
    B2=B2.reshape(3,-1)

    vec = np.cross(A2 - A1, B2 - B1, 0, 0, 0)
    nA = np.sum(np.cross(B2 - B1, A1 - B1, 0, 0, 0) * vec, axis=0)*np.ones(A1.shape[1])
    nB = np.sum(np.cross(A2 - A1, A1 - B1, 0, 0, 0) * vec, axis=0)*np.ones(A1.shape[1])
    d = np.sum(vec**2, axis=0)*np.ones(A1.shape[1])

    A0 = np.ones(A1.shape) * np.NaN
    B0 = A0.copy()
    idx = np.nonzero(d)[0]
    A0[:, idx] = A1[:, idx] + (nA[idx] / d[idx]) * (A2[:, idx] - A1[:, idx])
    B0[:, idx] = B1[:, idx] + (nB[idx] / d[idx]) * (B2[:, idx] - B1[:, idx])

    return A0, B0

if __name__ == '__main__':
    A1=np.array([-5.939,	-5.939,	-43.127])
    A2=np.array([4.016,	-4.016,	15.651])
    B1=np.array([-4.016,	-4.016,	15.651])
    B2=np.array([5.939,	-5.939,	-43.127])

#or
    A1=np.array([-5.939,	-5.939,	-50.001])
    A2=np.array([4.016,	-4.016,	15.651])
    B2=np.array([-4.016,	-4.016,	15.651])
    B1=np.array([5.939,	-5.939,	-50.001])


    print ('Intersection coordinates='+3*'{:8.4f}, ').format(*SegIntersect(A1, A2, B1, B2)[0].flatten())

    #main()
