import numpy as np
from scipy.interpolate import griddata
from numpy.linalg import norm


def project_points(points, x, N_vect):
    """
    Rotate the surface and the points in order to align the vector N_vect in the z direction
    """

    N_vect = N_vect / norm(N_vect)
    rot = calculate_rotation_matrix(N_vect)
    points_rot = dotX(rot, points)
    x_rot = dotX(rot, x)
    points_rot2 = np.array(
        [
            points_rot[:, 0],
            points_rot[:, 1],
            griddata(
                (x_rot[:, :, 0].flatten(), x_rot[:, :, 1].flatten()),
                x_rot[:, :, 2].flatten(),
                (points_rot[:, 0], points_rot[:, 1]),
                method="linear",
            ),
        ]
    ).T
    ### Rotate back
    inv_rot = np.linalg.inv(rot)
    points_final = dotX(inv_rot, points_rot2)
    return points_final


def normalize(v):

    return v / (np.dot(v, v) ** 2 + 1.0e-16)


def curvature(points):

    if len(points.shape) < 2:
        return None
    if points.shape[1] == 1:
        return None

    if points.shape[1] == 2:
        d1 = np.diff(points.T)
        d2 = np.diff(d1)
        x1 = d1[0, 1:]
        y1 = d1[1, 1:]
        x2 = d2[0, :]
        y2 = d2[1, :]
        curv = (x1 * y2 - y1 * x2) / (x1 ** 2 + y1 ** 2) ** (3.0 / 2.0)

    elif points.shape[1] == 3:
        d1 = np.diff(points.T)
        d2 = np.diff(d1)
        x1 = d1[0, 1:]
        y1 = d1[1, 1:]
        z1 = d1[2, 1:]
        x2 = d2[0, :]
        y2 = d2[1, :]
        z2 = d2[2, :]
        curv = ((z2 * y1 - y2 * z1) ** 2.0 + (x2 * z1 - z2 * x1) ** 2.0 + (y2 * x1 - x2 * y1) ** 2.0) ** 0.5 / (
            x1 ** 2.0 + y1 ** 2.0 + z1 ** 2.0 + 1.0e-30
        ) ** (3.0 / 2.0)

    curvt = np.zeros(points.shape[0])
    try:
        curvt[1:-1] = curv
        curvt[0] = curv[0]
        curvt[-1] = curv[-1]
    except:
        pass
    return curvt


def calculate_angle(v1, v2):
    """
    Calculate the signed angle between the vector, v1 and the vector, v2

    \param    v1      <c>array(3)</c>   :       vector 1
    \param    v2      <c>array(3)</c>   :       vector 2
    \retval   angle   <c>float radian</c> :     the angle in the two vector plane
    """

    v1 = v1 / norm(v1)
    v2 = v2 / norm(v2)
    return np.arctan2(norm(np.cross(v1, v2)), np.dot(v1, v2))


def calculate_rotation_matrix(vect):
    """
    Transpose (P1) and project the normal vector (vect) to the Z direction.

    \param    vect        vector to rotate                <c>numpy.array(3)</c>
    \retval   array       the rotation matrix             <c>numpy.array((3,3))</c>
    """
    return rotation_matrix_global(vect, np.array([0, 0, 1]))


def rotation_matrix_global(vect, direction):
    """
    Transpose (P1) and project the normal vector (vect) to the direction.

    \param    vect        vector to rotate               <c>numpy.array(3)</c>
    \param    direction   direction to rotate towards    <c>numpy.array(3)</c>
    \retval   array       the rotation matrix            <c>numpy.array((3,3))</c>
    """
    ### Normalize vect
    vect_norm = vect / norm(vect)
    ### Calculate the vector normal to the normal vector and the direction
    w = np.cross(vect_norm, direction)
    if norm(w) == 0:
        ### The two vectors are coplanar, exit with an identity matrix
        if np.dot(vect_norm, direction) > 0:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            return -np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    w_norm = w / norm(w)
    # print 'w_norm',w_norm
    ### The angle is found by taking the arccos of vect_norm[2]
    # q = math.acos(vect_norm[2])
    q = calculate_angle(vect_norm, direction)
    # print 'q',q
    ### calculate the rotation matrix of the vector w_norm and angle q
    rot = RotMat(w_norm, q)
    return rot


def dotX(rot, x, trans_vect=np.array([0.0, 0.0, 0.0])):
    """
    Transpose and Multiply the x array by a rotational matrix
    """
    if isinstance(x, list):
        x_tmp = np.array([x[0].flatten(), x[1].flatten(), x[2].flatten()]).T
        x_rot_tmp = np.zeros(x_tmp.shape)
        for i in range(x_tmp.shape[0]):
            x_rot_tmp[i, :] = dot(rot, x_tmp[i, :] - trans_vect)

        x_rot = []
        for iX in range(3):
            x_rot.append(x_rot_tmp[:, iX].reshape(x[0].shape))
    elif isinstance(x, np.ndarray):
        x_rot = np.zeros(x.shape)
        if len(x.shape) == 1:
            x_rot = np.dot(rot, x - trans_vect)
        elif len(x.shape) == 2:
            for i in range(x.shape[0]):
                x_rot[i, :] = np.dot(rot, x[i, :] - trans_vect)
        elif len(x.shape) == 3:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x_rot[i, j, :] = np.dot(rot, x[i, j, :] - trans_vect)

    return x_rot


# rotation matrix function for an x-rotation
RotX = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
# rotation matrix function for a y-rotation
RotY = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
# rotation matrix function for a z-rotation
RotZ = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def RotMat(u, theta):
    """ 
    Calculate the rotation matrix from a unit vector, u and an angle, theta
    
    http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    \f[
        R =\left[ \begin{array}{ccc}
        \cos\theta +u_x^2 \left(1-\cos \theta\right) & 
        u_x u_y \left(1-\cos \theta\right) - u_z \sin \theta & 
        u_x u_z \left(1-\cos \theta\right) + u_y \sin \theta \\ 
        u_y u_x \left(1-\cos \theta\right) + u_z \sin \theta & 
        \cos \theta + u_y^2\left(1-\cos \theta\right) & 
        u_y u_z \left(1-\cos \theta\right) - u_x \sin \theta \\
        u_z u_x \left(1-\cos \theta\right) - u_y \sin \theta &
        u_z u_y \left(1-\cos \theta\right) + u_x \sin \theta &
        \cos \theta + u_z^2\left(1-\cos \theta\right)
        \end{array} \right]
    \f]

    \param    u       <c> list/tuple/array(3) </c>       vector of the direction to rotate from
    \param    theta    <c> int radian </c>               angle to rotate with

    \retval   array   <c> array(3,3) </c>                rotation matrix
    """
    from numpy import cos, sin, array

    rot = array(
        [
            [
                cos(theta) + u[0] ** 2 * (1 - cos(theta)),
                u[0] * u[1] * (1 - cos(theta)) - u[2] * sin(theta),
                u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta),
            ],
            [
                u[1] * u[0] * (1 - cos(theta)) + u[2] * sin(theta),
                cos(theta) + u[1] ** 2 * (1 - cos(theta)),
                u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta),
            ],
            [
                u[2] * u[0] * (1 - cos(theta)) - u[1] * sin(theta),
                u[2] * u[1] * (1 - cos(theta)) + u[0] * sin(theta),
                cos(theta) + u[2] ** 2 * (1 - cos(theta)),
            ],
        ]
    )
    return rot


def dotXC(rot, x, center):
    """
    Transpose and Multiply the x array by a rotational matrix around a center
    """
    from numpy import zeros, array, dot

    if isinstance(x, list):
        x_tmp = array([x[0].flatten(), x[1].flatten(), x[2].flatten()]).T
        x_rot_tmp = zeros(x_tmp.shape)
        for i in range(x_tmp.shape[0]):
            x_rot_tmp[i, :] = dot(rot, x_tmp[i, :] - center) + center

        x_rot = []
        for iX in range(3):
            x_rot.append(x_rot_tmp[:, iX].reshape(x[0].shape))
    elif isinstance(x, np.ndarray):
        x_rot = zeros(x.shape)
        if len(x.shape) == 1:
            x_rot[:] = dot(rot, x - center) + center

        if len(x.shape) == 2:
            for i in range(x.shape[0]):
                x_rot[i, :] = dot(rot, x[i, :] - center) + center
        elif len(x.shape) == 3:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x_rot[i, j, :] = dot(rot, x[i, j, :] - center) + center
    return x_rot
