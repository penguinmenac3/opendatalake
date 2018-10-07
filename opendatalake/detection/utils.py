import abc
import math
import numpy as np
import cv2
import random

from opendatalake.texture_augmentation import full_texture_augmentation

try:
    LINE_TYPE = cv2.LINE_AA
except:
    LINE_TYPE = cv2.CV_AA


def modulus_ring(val, lower, upper):
    mod_range = upper - lower
    return ((val - lower) % mod_range) + lower


def top_down_overlap(g, p, tresh=0.0, projection_matrix=None):
    xyz_pred = p.get_xyz(projection_matrix=projection_matrix)
    xyz_true = g.get_xyz(projection_matrix=projection_matrix)

    theta_pred = p.theta
    theta_true = g.theta

    dx_raw = (xyz_true[0] - xyz_pred[0])
    dy_raw = (xyz_true[2] - xyz_pred[2])

    dx = math.cos(theta_true) * dx_raw + math.sin(theta_true) * dy_raw
    dy = -math.sin(theta_true) * dx_raw + math.cos(theta_true) * dy_raw
    dtheta = modulus_ring(theta_true - theta_pred, -math.pi, math.pi)
    l = g.l
    w = g.w

    condition = abs(dx) < tresh * l and abs(dy) < tresh * w and abs(dtheta) < tresh * math.radians(90)
    distance = abs(dx) + abs(dy) + abs(dtheta)
    return condition, distance, dx, dy, dtheta


class FusableDetection(object):
    def __init__(self, detection3dsimplified, conf_weighting=True, only_use_best_n=None):
        self.x = 0
        self.y = 0
        self.z = 0
        self.w = 0
        self.h = 0
        self.l = 0
        self.theta = 0
        self.conf = 0
        self.conf_weighting = conf_weighting
        self.only_use_best_n = only_use_best_n
        self.class_id = detection3dsimplified.class_id
        self.objs = [detection3dsimplified]
        self.update()

    def update(self):
        x = 0
        y = 0
        z = 0
        w = 0
        h = 0
        l = 0
        theta_s = 0
        theta_c = 0
        weights = 0
        max_conf = 0

        if self.only_use_best_n is not None:
            self.objs = sorted(self.objs, key=lambda x: -x.conf)

        for i, d in enumerate(self.objs):
            if self.only_use_best_n is not None and self.only_use_best_n <= i:
                break
            weight = 1
            if self.conf_weighting:
                weight = d.conf
            x += d.x * weight
            y += d.y * weight
            z += d.z * weight
            w += d.w * weight
            h += d.h * weight
            l += d.l * weight
            theta_s += math.sin(d.theta) * weight
            theta_c += math.cos(d.theta) * weight
            weights += weight
            max_conf = max(max_conf, d.conf)

        self.x = x / weights
        self.y = y / weights
        self.z = z / weights
        self.w = w / weights
        self.h = h / weights
        self.l = l / weights
        self.theta = math.atan2(theta_s, theta_c)
        self.conf = max_conf

    def add(self, fusable_detection):
        self.objs.extend(fusable_detection.objs)
        self.update()

    def to_detection3d_simplified(self):
        return Detection3dSimplified(self.objs[0].class_id, self.x, self.y, self.z, self.w, self.h, self.l, self.theta, self.conf)


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

    @abc.abstractmethod
    def copy(self):
        """
        Create a copy of the image.

        :return:
        """
        return

    @abc.abstractmethod
    def move_image(self, dx, dy):
        """
        Move the image the annotation is according to by some pixels.

        :param dx: The amount of pixels the origin of the image is moved to the right.
        :param dy: The amount of pixels the origin of the image is moved to the bottom.
        :return:
        """
        return

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Detection(" + str(self.to_dict()) + ")"


class Detection2d(Detection):
    __slots__ = ["class_id", "cx", "cy", "w", "h", "theta", "conf", "instance_mask"]

    def __init__(self, class_id, cx, cy, w, h, theta=0.0, conf=0, instance_mask=None):
        self.class_id = class_id
        self.conf = conf
        self.cx = cx
        self.cy = cy
        self.w = abs(w)
        self.h = abs(h)
        self.theta = theta
        self.instance_mask = instance_mask

    def move_image(self, dx, dy):
        self.cx = self.cx - dx
        self.cy = self.cy - dy
        if self.instance_mask is not None:
            self.instance_mask = list(self.instance_mask)
            for j in range(len(self.instance_mask)):
                self.instance_mask[j] = list(self.instance_mask[j])
                for i in range(int(len(self.instance_mask[j]) / 2)):
                    if type(self.instance_mask[j][2 * i + 0]) != str:
                        self.instance_mask[j][2 * i + 0] = self.instance_mask[j][2 * i + 0] - dx
                        self.instance_mask[j][2 * i + 1] = self.instance_mask[j][2 * i + 1] - dy

    def copy(self):
        return Detection2d(self.class_id, self.cx, self.cy, self.w, self.h, self.theta, self.conf, self.instance_mask)

    def iou(self, other):
        # Calculate edges of intersection area
        x_left = max(self.cx - self.w / 2.0, other.cx - other.w / 2.0)
        y_top = max(self.cy - self.h / 2.0, other.cy - other.h / 2.0)
        x_right = min(self.cx + self.w / 2.0, other.cx + other.w / 2.0)
        y_bottom = min(self.cy + self.h / 2.0, other.cy + other.h / 2.0)
        intersection_w = (x_right - x_left)
        intersection_h = (y_bottom - y_top)

        # When the intersection edges are of negative size then there is no intersection.
        if intersection_w < 0 or intersection_h < 0:
            return 0

        # Calculate areas
        intersection_area = intersection_w * intersection_h
        my_area = self.w * self.h
        other_area = other.w * other.h
        union = my_area + other_area - intersection_area

        # IOU is defined to be 1 if union area is 0.
        if abs(union) < 0.00001:
            return 1

        # Calculate and return iou
        return intersection_area / float(union)

    def to_array(self):
        return [self.class_id, self.cx, self.cy, self.w, self.h, self.theta, self.conf, self.instance_mask]

    def create_instance_mask(self, image_shape, color=255):
        mask = np.zeros(image_shape)
        cv2.fillPoly(mask, self._instance_mask_to_cv_pts(), color)
        return mask

    def _instance_mask_to_cv_pts(self):
        c = len(self.instance_mask)
        polys = []
        for poly in self.instance_mask:
            pts = []
            for i in range(int(len(poly)/ 2)):
                if type(poly[2 * i + 0]) != str:
                    pt = (int(poly[2 * i + 0]), int(poly[2 * i + 1]))
                    pts.append(pt)

            if len(pts) > 0:
                polys.append(np.array(pts))
        return polys

    def visualize(self, image, color=None, color_map=None, class_id_map=None):
        if color is None and color_map is None:
            return
        if color is None:
            color = color_map[self.class_id]
        cv2.rectangle(image,
                      (int(self.cx - self.w / 2.0), int(self.cy - self.h / 2.0)),
                      (int(self.cx + self.w / 2.0), int(self.cy + self.h / 2.0)),
                      (color[0], color[1], color[2]),
                      thickness=2)
        class_name = self.class_id
        if class_id_map is not None:
            class_name = class_id_map[self.class_id]
        cv2.putText(image,
                    "{}".format(class_name),
                    (int(self.cx - self.w / 2.0), int(self.cy - self.h / 2.0 - 3)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,
                    (color[0], color[1], color[2]),
                    1)
        if self.instance_mask is not None:
            pts = self._instance_mask_to_cv_pts()
            overlay = image.copy()
            cv2.fillPoly(overlay, pts, (color[0], color[1], color[2]))
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


class Detection25d(Detection):
    __slots__ = ["class_id", "cx", "cy", "dist", "w", "h", "l", "theta", "conf"]

    def __init__(self, class_id, cx, cy, dist, w, h, l, theta, conf=0):
        self.conf = conf
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.dist = dist
        self.w = w
        self.h = h
        self.l = l
        self.theta = theta

    def copy(self):
        return Detection25d(self.class_id, self.cx, self.cy, self.dist, self.w, self.h, self.l, self.theta, self.conf)

    def move_image(self, dx, dy):
        self.cx = self.cx - dx
        self.cy = self.cy - dy

    def to_detection3d_simplified(self, projection_matrix=None):
        xyz = self.get_xyz(projection_matrix=projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        return Detection3dSimplified(self.class_id, xyz[0], xyz[1], xyz[2], self.w, self.h, self.l, world_space_theta, self.conf)

    def to_2d_detection(self, projection_matrix):
        corners = self._project_corners(projection_matrix)
        min_x = 1000000
        max_x = 0
        min_y = 1000000
        max_y = 0
        for corner in corners:
            min_x = min(min_x, corner[0][0])
            max_x = max(max_x, corner[0][0])

            min_y = min(min_y, corner[1][0])
            max_y = max(max_y, corner[1][0])

        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2
        w = max_x - min_x
        h = max_y - min_y

        return Detection2d(self.class_id, cx, cy, w, h, conf=self.conf)

    def iou(self, other, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("IoU computation not possible without projection matrix.")

        detection2d = self.to_2d_detection(projection_matrix)
        other2d = other.to_2d_detection(projection_matrix)

        return detection2d.iou(other2d)

    def to_array(self):
        return [self.class_id, self.cx, self.cy, self.dist, self.w, self.h, self.l, self.theta, self.conf]

    def get_xyz(self, projection_matrix=None):
        return _uv_distance_to_xyz(self.cx, self.cy, self.dist, projection_matrix)

    def _project_corners(self, projection_matrix=None):
        corners = [[ self.l / 2.0,  self.h / 2.0,  self.w / 2.0],
                   [ self.l / 2.0,  self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0,  self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0,  self.h / 2.0,  self.w / 2.0],
                   [ self.l / 2.0, -self.h / 2.0,  self.w / 2.0],
                   [ self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0,  self.w / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = self.get_xyz(projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        corners = [_apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        return [apply_projection(x, projection_matrix) for x in corners]

    def visualize(self, image, color, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("Visualization not possible without projection matrix.")

        corners = self._project_corners(projection_matrix)
        corners = [(int(x[0][0]), int(x[1][0])) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose
        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, LINE_TYPE)
        cv2.circle(image, (int(self.cx), int(self.cy)), 5, color, thickness=2)

    def visualize_top_down(self, image, color, projection_matrix=None, scale=0.1):
        if projection_matrix is None:
            raise RuntimeError("Visualization not possible without projection matrix.")

        corners = [[self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = self.get_xyz(projection_matrix=projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        corners = [_apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        corners = [(int(x[0] / scale + image.shape[1] / 2), int(image.shape[0] - x[2] / scale)) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose

        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, LINE_TYPE)


class Detection25dSimplified(Detection):
    __slots__ = ["class_id", "cx", "y", "dist", "w", "h", "l", "theta", "conf"]

    def __init__(self, class_id, cx, y, dist, w, h, l, theta, conf=0):
        self.class_id = class_id
        self.conf = conf
        self.cx = cx
        self.y = y
        self.dist = dist
        self.w = w
        self.h = h
        self.l = l
        self.theta = theta

    def copy(self):
        return Detection25dSimplified(self.class_id, self.cx, self.y, self.dist, self.w, self.h, self.l, self.theta, self.conf)

    def move_image(self, dx, dy):
        self.cx = self.cx - dx

    def to_detection3d_simplified(self, projection_matrix=None):
        xyz = self.get_xyz(projection_matrix=projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        return Detection3dSimplified(self.class_id, xyz[0], xyz[1], xyz[2], self.w, self.h, self.l, world_space_theta)

    def to_2d_detection(self, projection_matrix):
        corners = self._project_corners(projection_matrix)
        min_x = 1000000
        max_x = 0
        min_y = 1000000
        max_y = 0
        for corner in corners:
            min_x = min(min_x, corner[0][0])
            max_x = max(max_x, corner[0][0])

            min_y = min(min_y, corner[1][0])
            max_y = max(max_y, corner[1][0])

        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2
        w = max_x - min_x
        h = max_y - min_y

        return Detection2d(self.class_id, cx, cy, w, h, conf=self.conf)

    def iou(self, other, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("IoU computation not possible without projection matrix.")

        detection2d = self.to_2d_detection(projection_matrix)
        other2d = other.to_2d_detection(projection_matrix)

        return detection2d.iou(other2d)

    def to_array(self):
        return [self.class_id, self.cx, self.y, self.dist, self.w, self.h, self.l, self.theta, self.conf]

    def get_xyz(self, projection_matrix=None):
        return _uy_distance_to_xyz(self.cx, self.y, self.dist, projection_matrix)

    def _project_corners(self, projection_matrix=None):
        corners = [[ self.l / 2.0,  self.h / 2.0,  self.w / 2.0],
                   [ self.l / 2.0,  self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0,  self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0,  self.h / 2.0,  self.w / 2.0],
                   [ self.l / 2.0, -self.h / 2.0,  self.w / 2.0],
                   [ self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0,  self.w / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = self.get_xyz(projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        corners = [_apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        return [apply_projection(x, projection_matrix) for x in corners]

    def visualize(self, image, color, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("Visualization not possible without projection matrix.")

        corners = self._project_corners(projection_matrix)
        corners = [(int(x[0][0]), int(x[1][0])) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose
        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, LINE_TYPE)

    def visualize_top_down(self, image, color, projection_matrix=None, scale=0.1):
        if projection_matrix is None:
            raise RuntimeError("Visualization not possible without projection matrix.")

        corners = [[self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = self.get_xyz(projection_matrix=projection_matrix)
        world_space_theta = self.theta - math.atan2(xyz[2], xyz[0]) + math.radians(90)
        corners = [_apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        corners = [(int(x[0] / scale + image.shape[1] / 2), int(image.shape[0] - x[2] / scale)) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose

        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, LINE_TYPE)


class Detection3dSimplified(Detection):
    __slots__ = ["class_id", "x", "y", "z", "w", "h", "l", "theta", "conf"]

    def __init__(self, class_id, x, y, z, w, h, l, theta, conf=0):
        self.class_id = class_id
        self.conf = conf
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.l = l
        self.theta = theta

    def move_image(self, dx, dy):
        raise NotImplementedError("Move image not implemented.")

    def copy(self):
        return Detection3dSimplified(self.class_id, self.x, self.y, self.z, self.w, self.h, self.l, self.theta, self.conf)

    def to_detection3d_simplified(self, projection_matrix=None):
        return self.copy()

    def to_2d_detection(self, projection_matrix):
        corners = self._project_corners(projection_matrix)
        min_x = 1000000
        max_x = 0
        min_y = 1000000
        max_y = 0
        for corner in corners:
            min_x = min(min_x, corner[0][0])
            max_x = max(max_x, corner[0][0])

            min_y = min(min_y, corner[1][0])
            max_y = max(max_y, corner[1][0])

        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2
        w = max_x - min_x
        h = max_y - min_y

        return Detection2d(self.class_id, cx, cy, w, h, conf=self.conf)

    def iou(self, other, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("IoU computation not possible without projection matrix.")

        detection2d = self.to_2d_detection(projection_matrix)
        other2d = other.to_2d_detection(projection_matrix)

        return detection2d.iou(other2d)

    def to_array(self):
        return [self.class_id, self.x, self.y, self.z, self.w, self.h, self.l, self.theta, self.conf]

    def get_xyz(self, projection_matrix=None):
        return self.x, self.y, self.z

    def _project_corners(self, projection_matrix=None):
        corners = [[self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = self.get_xyz()
        world_space_theta = self.theta
        corners = [_apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        return [apply_projection(x, projection_matrix) for x in corners]

    def visualize(self, image, color, projection_matrix=None):
        if projection_matrix is None:
            raise RuntimeError("Visualization not possible without projection matrix.")

        corners = self._project_corners(projection_matrix)
        corners = [(int(x[0][0]), int(x[1][0])) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose
        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, LINE_TYPE)

    def visualize_top_down(self, image, color, scale=0.1, projection_matrix=None):
        corners = [[self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, -self.w / 2.0],
                   [-self.l / 2.0, -self.h / 2.0, self.w / 2.0],
                   [self.l / 1.5, 0, 0]]

        xyz = self.get_xyz()
        world_space_theta = self.theta
        corners = [_apply_affine_transform(x, theta=world_space_theta, translation=xyz) for x in corners]
        corners = [(int(x[0] / scale + image.shape[1] / 2), int(image.shape[0] - x[2] / scale)) for x in corners]

        connections = [(1, 2), (3, 0),  # Sides
                       (5, 6), (7, 4),
                       (2, 3), (3, 7), (7, 6), (6, 2),  # Back
                       (0, 1), (1, 5), (5, 4), (4, 0),  # Front
                       (0, 8), (1, 8), (4, 8), (5, 8)]  # Pointy Nose

        lw = 2
        for i, j in connections:
            cv2.line(image, corners[i], corners[j], color, lw, LINE_TYPE)


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

    def copy(self):
        return Detection3d(self.class_id, self.cx, self.cy, self.cz, self.w, self.h, self.l, self.q0, self.q1, self.q2, self.q3)

    def move_image(self, dx, dy):
        raise NotImplementedError("Move image not implemented.")

    def iou(self, other):
        raise NotImplementedError("IoU not implemented for 3d yet.")

    def to_array(self):
        return [self.class_id, self.cx, self.cy, self.cz, self.w, self.h, self.l, self.q0, self.q1, self.q2, self.q3]

    def visualize(self, image, color):
        raise NotImplementedError("Visualization not implemented.")


def _apply_affine_transform(point, theta, translation):
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


def _uv_distance_to_xyz(u, v, distance, projection_matrix):
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
    b = 2 * (alpha_1 * beta_1 + alpha_2 * beta_2)
    c = beta_1 * beta_1 + beta_2 * beta_2 - distance * distance

    solution_1, solution_2 = solve_quadratic_equation(a, b, c)

    z = solution_1 if solution_1 > solution_2 else solution_2
    x = alpha_1 * z + beta_1
    y = alpha_2 * z + beta_2
    return [x, y, z]


def _uy_distance_to_xyz(u, y, distance, projection_matrix):
    m00 = float(projection_matrix[0][0])
    m02 = float(projection_matrix[0][2])
    m22 = float(projection_matrix[2][2])
    t_1 = float(projection_matrix[0][3])
    t_3 = float(projection_matrix[2][3])

    alpha_1 = (u * m22 - m02) / m00
    beta_1 = (u * t_3 - t_1) / m00

    a = 1 + alpha_1 * alpha_1
    b = 2 * (alpha_1 * beta_1)
    c = beta_1 * beta_1 + y * y - distance * distance

    solution_1, solution_2 = solve_quadratic_equation(a, b, c)

    z = solution_1 if solution_1 > solution_2 else solution_2
    x = alpha_1 * z + beta_1
    return [x, y, z]


def solve_quadratic_equation(a, b, c):
    solution_1 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    solution_2 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return solution_1, solution_2


def crop_image(img, start_y, start_x, h, w):
    """
    Crop an image given the top left corner.
    :param img: The image
    :param start_y: The top left corner y coord
    :param start_x: The top left corner x coord
    :param h: The result height
    :param w: The result width
    :return: The cropped image.
    """
    return img[start_y:start_y + h, start_x:start_x + w, :].copy()


def move_detections(label, dy, dx):
    """
    Move detections in direction dx, dy.

    :param label: The label dict containing all detection lists.
    :param dy: The delta in y direction as a number.
    :param dx: The delta in x direction as a number.
    :return:
    """
    for k in label.keys():
        if k.startswith("detection"):
            detections = label[k]
            for detection in detections:
                detection.move_image(-dx, -dy)


def hflip_detections(label, w):
    """
    Horizontally flip detections according to an image flip.

    :param label: The label dict containing all detection lists.
    :param w: The width of the image as a number.
    :return:
    """
    for k in label.keys():
        if k.startswith("detection"):
            detections = label[k]
            for detection in detections:
                detection.cx = w - detection.cx
                if k == "detections_2.5d":
                    detection.theta = math.pi - detection.theta


def augment_detections(hyper_params, feature, label):
    """
    Augment the detection dataset.

    In your hyper_parameters.problem.augmentation add configurations to enable features.
    Supports "enable_horizontal_flip", "enable_micro_translation", "random_crop" : {"shape": { "width", "height" }}
    and "enable_texture_augmentation". Make sure to also set the "steps" otherwise this method will not be used properly.

    Random crop ensures at least one detection is in the crop region.

    Sample configuration
    "problem": {
        "augmentation": {
            "steps": 40,
            "enable_texture_augmentation": true,
            "enable_micro_translation": true,
            "enable_horizontal_flip": true,
            "random_crop": {
                "shape": {
                    "width": 256,
                    "height": 256
                }
            }
        }
    }

    :param hyper_params: The hyper parameters object
    :param feature: A dict containing all features, must be in the style created by detection datasets.
    :param label: A label dict in the detection dataset style.
    :return: Modified feature and label dict (copied & modified).
    """
    # Do not augment these ways:
    # 1) Rotation is not possible
    # 3) Scaling is not possible, because it ruins depth perception
    # However, random crops can improve performance. (Training speed and accuracy)
    if hyper_params.problem.get("augmentation", None) is None:
        return feature, label

    img_h, img_w, img_c = feature["image"].shape
    augmented_feature = {}
    augmented_label = {}
    augmented_feature["image"] = feature["image"].copy()
    if "depth" in feature:
        augmented_feature["depth"] = feature["depth"].copy()
    if "calibration" in feature:
        augmented_feature["calibration"] = feature["calibration"]
    augmented_feature["hflipped"] = np.array([0], dtype=np.uint8)
    augmented_feature["crop_offset"] = np.array([0, 0], dtype=np.int8)

    for k in label.keys():
        augmented_label[k] = [detection.copy() for detection in label[k]]

    if hyper_params.problem.augmentation.get("enable_horizontal_flip", False):
        if random.random() < 0.5:
            img_h, img_w, img_c = augmented_feature["image"].shape
            augmented_feature["image"] = np.fliplr(augmented_feature["image"])
            if "depth" in feature:
                augmented_feature["depth"] = np.fliplr(augmented_feature["depth"])
            augmented_feature["hflipped"][0] = 1
            hflip_detections(augmented_label, img_w)

    if hyper_params.problem.augmentation.get("enable_micro_translation", False):
        img_h, img_w, img_c = augmented_feature["image"].shape
        dx = int(random.random() * 3)
        dy = int(random.random() * 3)

        augmented_feature["image"] = crop_image(augmented_feature["image"], dy, dx, img_h - dy, img_w - dx)
        if "depth" in feature:
            augmented_feature["depth"] = crop_image(augmented_feature["depth"], dy, dx, img_h - dy, img_w - dx)
        augmented_feature["crop_offset"][0] += dy
        augmented_feature["crop_offset"][1] += dx

        move_detections(augmented_label, -dy, -dx)

    if hyper_params.problem.augmentation.get("random_crop", None) is not None:
        img_h, img_w, img_c = augmented_feature["image"].shape
        target_w = hyper_params.problem.augmentation.random_crop.shape.width
        target_h = hyper_params.problem.augmentation.random_crop.shape.height

        delta_x = max(int(math.ceil((target_w + 1 - img_w) / 2)), 0)
        delta_y = max(int(math.ceil((target_h + 1 - img_h) / 2)), 0)
        move_detections(augmented_label, delta_y, delta_x)
        augmented_feature["image"] = cv2.copyMakeBorder(augmented_feature["image"],
                                                        delta_y, delta_y, delta_x, delta_x,
                                                        cv2.BORDER_CONSTANT)
        img_h, img_w, img_c = augmented_feature["image"].shape

        start_x = 0
        start_y = 0
        if len(augmented_label["detections_2d"]) != 0:
            idx = random.randint(0, len(augmented_label["detections_2d"]) - 1)
            detection = augmented_label["detections_2d"][idx]
            start_x = int(detection.cx - random.random() * (target_w - 20) / 2.0 - 10)
            start_y = int(detection.cy - random.random() * (target_h - 20) / 2.0 - 10)
        else:
            start_x = int(img_w * random.random())
            start_y = int(img_h * random.random())

        # Compute start point so that crop fit's into image and random crop contains detection
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if start_x >= img_w - target_w:
            start_x = img_w - target_w - 1
        if start_y >= img_h - target_h:
            start_y = img_h - target_h - 1

        # Crop image
        augmented_feature["image"] = crop_image(augmented_feature["image"], start_y, start_x, target_h, target_w)
        if "depth" in feature:
            augmented_feature["depth"] = crop_image(augmented_feature["depth"], start_y, start_x, target_h, target_w)
        augmented_feature["crop_offset"][0] += start_y
        augmented_feature["crop_offset"][1] += start_x

        # Crop labels
        move_detections(augmented_label, -start_y, -start_x)

    if hyper_params.problem.augmentation.get("enable_texture_augmentation", False):
        if random.random() < 0.5:
            augmented_feature["image"] = full_texture_augmentation(augmented_feature["image"])

    return augmented_feature, augmented_label
