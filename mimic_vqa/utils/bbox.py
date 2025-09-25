"""
Bounding box utilities for spatial grounding
"""
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box representation with coordinates (x, y, w, h)"""
    x: int  # x-coordinate of top-left corner
    y: int  # y-coordinate of top-left corner  
    w: int  # width
    h: int  # height
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate bounding box coordinates"""
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"Invalid bbox dimensions: w={self.w}, h={self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Invalid bbox position: x={self.x}, y={self.y}")
    
    @property
    def x2(self) -> int:
        """Right edge x-coordinate"""
        return self.x + self.w
    
    @property
    def y2(self) -> int:
        """Bottom edge y-coordinate"""
        return self.y + self.h
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates"""
        return (self.x + self.w / 2, self.y + self.h / 2)
    
    @property
    def area(self) -> int:
        """Bounding box area"""
        return self.w * self.h
    
    def intersection_over_union(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another bounding box"""
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate center-to-center distance"""
        c1_x, c1_y = self.center
        c2_x, c2_y = other.center
        return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside bounding box"""
        return self.x <= x < self.x2 and self.y <= y < self.y2
    
    def expand(self, margin: int) -> 'BoundingBox':
        """Expand bounding box by margin"""
        return BoundingBox(
            x=max(0, self.x - margin),
            y=max(0, self.y - margin),
            w=self.w + 2 * margin,
            h=self.h + 2 * margin,
            confidence=self.confidence
        )
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x2, self.y2)
    
    def to_cxcywh(self) -> Tuple[float, float, int, int]:
        """Convert to center coordinates format"""
        cx, cy = self.center
        return (cx, cy, self.w, self.h)
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int, 
                  confidence: float = 1.0) -> 'BoundingBox':
        """Create from (x1, y1, x2, y2) coordinates"""
        return cls(x=x1, y=y1, w=x2-x1, h=y2-y1, confidence=confidence)
    
    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: int, h: int,
                    confidence: float = 1.0) -> 'BoundingBox':
        """Create from center coordinates"""
        return cls(x=int(cx - w/2), y=int(cy - h/2), w=w, h=h, 
                  confidence=confidence)
    
    def __str__(self) -> str:
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"
    
    def __repr__(self) -> str:
        return self.__str__()


def parse_bbox_string(bbox_str: str) -> Optional[BoundingBox]:
    """
    Parse bounding box from string format like:
    - "[321, 133, 507, 153]"
    - "BBox: [321, 133, 507, 153]"
    - "x=321, y=133, w=186, h=20"
    """
    if not bbox_str:
        return None
    
    # Try different patterns
    patterns = [
        # [x, y, w, h] or [x1, y1, x2, y2]
        r'\[(\d+),?\s*(\d+),?\s*(\d+),?\s*(\d+)\]',
        # x=X, y=Y, w=W, h=H
        r'x=(\d+).*?y=(\d+).*?w=(\d+).*?h=(\d+)',
        # (x, y, w, h)
        r'\((\d+),?\s*(\d+),?\s*(\d+),?\s*(\d+)\)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, bbox_str)
        if match:
            coords = [int(x) for x in match.groups()]
            
            # Determine if it's xywh or xyxy format based on context
            if "x=" in bbox_str and "w=" in bbox_str:
                # Explicit xywh format
                return BoundingBox(*coords)
            else:
                # Assume xyxy format, convert to xywh
                x1, y1, x2, y2 = coords
                return BoundingBox.from_xyxy(x1, y1, x2, y2)
    
    return None


def format_bbox_string(bbox: BoundingBox, format_type: str = "xywh") -> str:
    """Format bounding box as string"""
    if format_type == "xywh":
        return f"[{bbox.x}, {bbox.y}, {bbox.w}, {bbox.h}]"
    elif format_type == "xyxy":
        return f"[{bbox.x}, {bbox.y}, {bbox.x2}, {bbox.y2}]"
    elif format_type == "cxcywh":
        cx, cy = bbox.center
        return f"[{cx:.1f}, {cy:.1f}, {bbox.w}, {bbox.h}]"
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def merge_bounding_boxes(boxes: List[BoundingBox], 
                        min_iou: float = 0.1) -> List[BoundingBox]:
    """Merge overlapping bounding boxes"""
    if not boxes:
        return []
    
    merged = []
    used = set()
    
    for i, box1 in enumerate(boxes):
        if i in used:
            continue
            
        # Find all boxes that overlap with this one
        group = [box1]
        used.add(i)
        
        for j, box2 in enumerate(boxes[i+1:], i+1):
            if j in used:
                continue
                
            if box1.intersection_over_union(box2) >= min_iou:
                group.append(box2)
                used.add(j)
        
        # Merge the group into a single box
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Calculate bounding box that encompasses all
            min_x = min(box.x for box in group)
            min_y = min(box.y for box in group)
            max_x2 = max(box.x2 for box in group)
            max_y2 = max(box.y2 for box in group)
            
            # Average confidence
            avg_confidence = sum(box.confidence for box in group) / len(group)
            
            merged_box = BoundingBox.from_xyxy(min_x, min_y, max_x2, max_y2, 
                                             avg_confidence)
            merged.append(merged_box)
    
    return merged
