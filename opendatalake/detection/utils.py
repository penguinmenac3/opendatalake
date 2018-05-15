import abc
import math
import numpy as np
import cv2


class Detection(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def to_array(self):
        """
        Converts the detection into an array.
        Values depend on the data type but they are in the order specified in __slots__.
        The rough structure is:
        1 Class id (int)
        2 Center position
        3 Dimensions of object
        4 Orientation of object
        :return: An array containing the values.
        """
        return []

    def to_dict(self):
        """
        Converts the object into a dictionary.
        :return: The dictionary containing all variables of a detection.
        """
        return dict(zip(self.__slots__, self.to_array()))

    @abc.abstractmethod
    def iou(self, other):
        """
        Calculates the iou between two detections.
        :param other: The other detection that is used to compute the iou.
        :return: The intersection over union value.
        """
        return 0.0

    @abc.abstractmethod
    def visualize(self, image, color):
        """
        Draw the detection into the given image.

        :param image: A numpy nd array representing the image (shape=[h,w,3]).
        :param color: The color to use for drawing.
        :return: Nothing the input image is modified.
        """
        return

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Detection(" + str(self.to_dict()) + ")"


class Detection2d(Detection):
    __slots__ = ["class_id", "cx", "cy", "w", "h", "theta"]

    def __init__(self, class_id, cx, cy, w, h, theta=0.0):
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.theta = theta

    def iou(self, other):
        raise NotImplementedError("IoU not implemented for 2d yet.")

    def to_array(self):
        return [self.class_id, self.cx, self.cy, self.w, self.h, self.theta]

    def visualize(self, image, color):
        raise NotImplementedError("Visualization not implemented.")


class Detection25d(Detection):
    __slots__ = ["class_id", "cx", "cy", "dist", "w", "h", "l", "theta"]

    def __init__(self, class_id, cx, cy, dist, w, h, l, theta):
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.dist = dist
        self.w = w
        self.h = h
        self.l = l
        self.theta = theta

    def iou(self, other):
        raise NotImplementedError("IoU not implemented for 2.5d yet.")

    def to_array(self):
        return [self.class_id, self.cx, self.cy, self.dist, self.w, self.h, self.l, self.theta]

    def visualize(self, image, color, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("Visualization not possible without projection matrix.")

        corners = [[ self.l / 2.0,  self.w / 2.0,  self.h / 2.0],
                   [ self.l / 2.0, -self.w / 2.0,  self.h / 2.0],
                   [-self.l / 2.0, -self.w / 2.0,  self.h / 2.0],
                   [-self.l / 2.0,  self.w / 2.0,  self.h / 2.0],
                   [ self.l / 2.0,  self.w / 2.0, -self.h / 2.0],
                   [ self.l / 2.0, -self.w / 2.0, -self.h / 2.0],
                   [-self.l / 2.0, -self.w / 2.0, -self.h / 2.0],
                   [-self.l / 2.0,  self.w / 2.0, -self.h / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = uv_distance_to_xyz(self.cx, self.cy, self.dist, projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        corners = [apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        corners = [apply_projection(x, projection_matrix) for x in corners]
        corners = [(int(x[0][0]), int(x[1][0])) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose
        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, cv2.LINE_AA)


class Detection3d(Detection):
    __slots__ = ["class_id", "cx", "cy", "cz", "w", "h", "l", "q0", "q1", "q2", "q3"]

    def __init__(self, class_id, cx, cy, cz, w, h, l, q0, q1, q2, q3):
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.w = w
        self.h = h
        self.l = l
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def iou(self, other):
        raise NotImplementedError("IoU not implemented for 3d yet.")

    def to_array(self):
        return [self.class_id, self.cx, self.cy, self.cz, self.w, self.h, self.l, self.q0, self.q1, self.q2, self.q3]

    def visualize(self, image, color):
        raise NotImplementedError("Visualization not implemented.")


def apply_affine_transform(point, theta, translation):
    c = math.cos(theta)
    s = math.sin(theta)
    rot_mat = np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])

    result = np.dot(rot_mat, np.array(point))
    return result + np.array(translation)


def apply_projection(point, projection_matrix):
    homogenous_point = np.insert(np.array(point).reshape((3, -1)), 3, 1, axis=0)
    result = np.dot(projection_matrix, homogenous_point)
    d = result[-1]
    result = result / result[-1]
    result = np.delete(result, 2, axis=0)

    return result


def vec_len(vector):
    len2 = 0
    for x in vector:
        len2 += x * x
    return math.sqrt(len2)


def uv_distance_to_xyz(u, v, distance, projection_matrix):
    m00 = float(projection_matrix[0][0])
    m11 = float(projection_matrix[1][1])
    m02 = float(projection_matrix[0][2])
    m12 = float(projection_matrix[1][2])
    m22 = float(projection_matrix[2][2])
    t_1 = float(projection_matrix[0][3])
    t_2 = float(projection_matrix[1][3])
    t_3 = float(projection_matrix[2][3])

    alpha_1 = (u * m22 - m02) / m00
    alpha_2 = (v * m22 - m12) / m11
    beta_1 = (u * t_3 - t_1) / m00
    beta_2 = (v * t_3 - t_2) / m11

    a = 1 + alpha_1 * alpha_1 + alpha_2 * alpha_2
    b = alpha_1 * beta_1 + alpha_2 * beta_2
    c = beta_1 * beta_1 + beta_2 * beta_2 - distance * distance

    solution_1, solution_2 = solve_quadratic_equation(a, b, c)

    z = solution_1 if solution_1 > solution_2 else solution_2
    x = alpha_1 * z + beta_1
    y = alpha_2 * z + beta_2
    return [x, y, z]


def solve_quadratic_equation(a, b, c):
    solution_1 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    solution_2 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return solution_1, solution_2
