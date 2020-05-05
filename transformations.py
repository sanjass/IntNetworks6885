from scipy.spatial.transform import Rotation as R
import numpy as np



def rotate_vector(vector, alpha, beta, gamma):
	"""
	Input:
	vector: vector to be rotated
	alpha, beta, gamma -  rotation angles in degrees.
	Returns:
	rotated vector

	A yaw is a counterclockwise rotation of alpha about the z-axis.
	A pitch is a counterclockwise rotation of beta about the y-axis.
	A roll is a counterclockwise rotation of gamma about the x-axis.
	The order of rotation matters.
	See documentation here: 
	https://docs.scipy.org/doc/scipy/reference/generated/
	scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
	"""
	r = R.from_euler('xyz', [gamma, beta, alpha], degrees=True)
	return r.apply(vector)


def world2camera_coord(location, camera):
    def _world2camera_coord(p_w, c, pitch, yaw, roll, ignore_pitch=False):
        """
        Turn world coordinates into camera coordinates
        Inputs:
          coord: world coordinates P_w of the object
          camera: a dict specifying camera's world coordinates (x, y, z) and rotations (pitch, yaw, roll)
        Output: P_c = R (P_w - C)
        R is rotation matrix from world coordinate to camera coordinate
            [ Cos[a] Cos[b]   Cos[b] Sin[a]   Sin[b] ]
        R = [    -Sin[a]          Cos[a]        0    ]
            [-Cos[a] Sin[b]  -Sin[a] Sin[b]   Cos[b] ]
        where a = yaw, b = pitch and c = roll = 0
        """
        # if roll != 0:
        # 	print("WARNING: roll is not 0, but code expects it to be 0")
        #assert roll == 0
        x, y, z = p_w[0] - c["x"], p_w[1] - c["y"], p_w[2] - c["z"]  # P_w - C
        if ignore_pitch:
            pitch = 0
            z = p_w[2]
        a = yaw / 180 * np.pi
        b = pitch / 180 * np.pi
        return [
            np.cos(a) * np.cos(b) * x + np.cos(b) * np.sin(a) * y + np.sin(b) * z,
            -np.sin(a) * x + np.cos(a) * y,
            -np.cos(a) * np.sin(b) * x - np.sin(a) * np.sin(b) * y + np.cos(b) * z]

    camera_location = camera["location"]
    camera_rotation = camera["rotation"]
    return _world2camera_coord(location, camera_location, camera_rotation["pitch"], camera_rotation["yaw"],
                               camera_rotation["roll"])


def world2camera_rotation(obj_rot, camera):

	camera_rotation = camera["rotation"]
	cam_rot = [camera_rotation["pitch"], camera_rotation["yaw"], camera_rotation["roll"]]
	result = [obj_rot[0] - cam_rot[0],
	          obj_rot[1] - cam_rot[1],
	          obj_rot[2] - cam_rot[2]
	         ]
	return result