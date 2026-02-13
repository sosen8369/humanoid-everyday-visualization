import math

import cv2
import numpy as np

class ShapeGroup:
    def __init__(self, x, y, size, colors, valids, angle=0, thickness=1):
        self.x = x
        self.y = y
        self.size = size
        self.colors = colors
        self.valids = valids
        self.angle = angle 
        self.thickness = thickness

    def rotate_point(self, px, py):
        theta = math.radians(self.angle)
        nx = math.cos(theta) * (px - self.x) - math.sin(theta) * (py - self.y) + self.x
        ny = math.sin(theta) * (px - self.x) + math.cos(theta) * (py - self.y) + self.y
        return [int(nx), int(ny)]
    
    def draw_null_pattern(self, img, pts_np):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, pts_np, 255)

        tile_size = 8
        pattern = np.full((tile_size*2, tile_size*2), 220, dtype=np.uint8)
        pattern[:tile_size, :tile_size] = 255
        pattern[tile_size:, tile_size:] = 255

        full_pattern = np.tile(pattern, (img.shape[0]//(tile_size*2)+1, img.shape[1]//(tile_size*2)+1))
        full_pattern = full_pattern[:img.shape[0], :img.shape[1]]
        full_pattern_bgr = cv2.cvtColor(full_pattern, cv2.COLOR_GRAY2BGR)
        
        img[mask == 255] = full_pattern_bgr[mask == 255]

    def draw_rotated_rect(self, img, x1, y1, x2, y2, color, isvalid):
        pts = [
            self.rotate_point(x1, y1),
            self.rotate_point(x2, y1),
            self.rotate_point(x2, y2),
            self.rotate_point(x1, y2)
        ]
        pts_np = np.array([pts], dtype=np.int32)

        if not color:
            self.draw_null_pattern(img, pts_np)
        else:
            cv2.fillPoly(img, pts_np, color) #type: ignore

        if self.thickness > 0:
            if isvalid:
                cv2.polylines(img, pts_np, True, (0, 0, 0), self.thickness, lineType=cv2.LINE_AA) #type: ignore
            else:
                cv2.polylines(img, pts_np, True, (0, 0, 200), self.thickness, lineType=cv2.LINE_AA) #type: ignore

class PyramidGroup(ShapeGroup):
    def draw(self, img):
        s = self.size
        self.draw_rotated_rect(img, self.x - s, self.y, self.x, self.y + s, self.colors[0], self.valids[0])
        self.draw_rotated_rect(img, self.x, self.y, self.x + s, self.y + s, self.colors[1], self.valids[1])
        self.draw_rotated_rect(img, self.x - s//2, self.y - s, self.x + s//2, self.y, self.colors[2], self.valids[2])

class GridGroup(ShapeGroup):
    def draw(self, img):
        s = self.size
        self.draw_rotated_rect(img, self.x - s, self.y - s, self.x, self.y, self.colors[0], self.valids[0])
        self.draw_rotated_rect(img, self.x, self.y - s, self.x + s, self.y, self.colors[1], self.valids[1])
        self.draw_rotated_rect(img, self.x - s, self.y, self.x, self.y + s, self.colors[2], self.valids[2])
        self.draw_rotated_rect(img, self.x, self.y, self.x + s, self.y + s, self.colors[3], self.valids[3])