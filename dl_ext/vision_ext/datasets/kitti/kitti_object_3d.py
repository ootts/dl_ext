from enum import Enum

KITTIObjectClass = Enum('KITTIObjectClass', ('Car', 'Van', 'Truck',
                                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                             'Misc', 'DontCare'))


class KITTIObject3D:
    cls: KITTIObjectClass
    truncated: float
    occluded: float
    alpha: float
    x1: float
    y1: float
    x2: float
    y2: float
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float

    def __init__(self, cls: KITTIObjectClass, truncated: float, occluded: float, alpha: float,
                 x1: float, y1: float, x2: float, y2: float,
                 h: float, w: float, l: float,
                 x: float, y: float, z: float, ry: float) -> None:
        super().__init__()
        self.ry = ry
        self.z = z
        self.y = y
        self.x = x
        self.l = l
        self.w = w
        self.h = h
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.alpha = alpha
        self.occluded = occluded
        self.truncated = truncated
        self.cls = cls
