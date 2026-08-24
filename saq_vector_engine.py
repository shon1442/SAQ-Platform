import io
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import shapely.geometry as sg
from shapely.ops import polygonize, unary_union
import ezdxf
from ezdxf.document import Drawing

class DXFVectorParser:
    def __init__(self, file_bytes_or_path, unit_scale_to_meter: float = 1.0):
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            stream = io.StringIO(file_bytes_or_path.decode('utf-8', errors='ignore'))
            self.doc: Drawing = ezdxf.read(stream)
        elif isinstance(file_bytes_or_path, io.BytesIO):
            content = file_bytes_or_path.getvalue().decode('utf-8', errors='ignore')
            stream = io.StringIO(content)
            self.doc: Drawing = ezdxf.read(stream)
        elif isinstance(file_bytes_or_path, str):
            self.doc: Drawing = ezdxf.readfile(file_bytes_or_path)
        else:
            raise ValueError("Unsupported format")
        self.msp = self.doc.modelspace()
        self.scale = unit_scale_to_meter
        self._auto_detect_unit()

    def _auto_detect_unit(self):
        insunits = self.doc.header.get('$INSUNITS', 0)
        if insunits == 4:
            self.scale = 0.001
        elif insunits == 5:
            self.scale = 0.01
        elif insunits == 6:
            self.scale = 1.0

    def get_layers_summary(self) -> List[Dict[str, Any]]:
        layers = {}
        for layer in self.doc.layers:
            layers[layer.dxf.name] = {"name": layer.dxf.name, "color": layer.dxf.color, "is_off": layer.is_off(), "entity_count": 0}
        for entity in self.msp:
            l_name = entity.dxf.layer
            if l_name not in layers:
                layers[l_name] = {"name": l_name, "color": 7, "is_off": False, "entity_count": 0}
            layers[l_name]["entity_count"] += 1
        return sorted(list(layers.values()), key=lambda x: x["entity_count"], reverse=True)

    def extract_lines_and_segments(self, target_layers: Optional[List[str]] = None) -> List[sg.LineString]:
        segments = []
        for entity in self.msp:
            if target_layers and entity.dxf.layer not in target_layers:
                continue
            dxftype = entity.dxftype()
            if dxftype == 'LINE':
                p1 = (entity.dxf.start.x * self.scale, entity.dxf.start.y * self.scale)
                p2 = (entity.dxf.end.x * self.scale, entity.dxf.end.y * self.scale)
                if p1 != p2:
                    segments.append(sg.LineString([p1, p2]))
            elif dxftype == 'LWPOLYLINE':
                points = [(p[0] * self.scale, p[1] * self.scale) for p in entity.get_points('xy')]
                if len(points) >= 2:
                    if entity.is_closed and points[0] != points[-1]:
                        points.append(points[0])
                    for i in range(len(points) - 1):
                        if points[i] != points[i+1]:
                            segments.append(sg.LineString([points[i], points[i+1]]))
        return segments

    def extract_closed_polygons(self, target_layers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        polygons = []
        for entity in self.msp:
            if target_layers and entity.dxf.layer not in target_layers:
                continue
            if entity.dxftype() == 'LWPOLYLINE' and entity.is_closed:
                pts = [(p[0] * self.scale, p[1] * self.scale) for p in entity.get_points('xy')]
                if len(pts) >= 3:
                    poly = sg.Polygon(pts)
                    if poly.is_valid and poly.area > 0.1:
                        polygons.append({"layer": entity.dxf.layer, "area_m2": round(poly.area, 3), "perimeter_m": round(poly.length, 3), "geometry": poly, "centroid": (poly.centroid.x, poly.centroid.y)})
        if not polygons:
            segments = self.extract_lines_and_segments(target_layers)
            if segments:
                noded = unary_union(segments)
                for poly in list(polygonize(noded)):
                    if poly.is_valid and 0.5 < poly.area < 500.0:
                        polygons.append({"layer": "polygonized", "area_m2": round(poly.area, 3), "perimeter_m": round(poly.length, 3), "geometry": poly, "centroid": (poly.centroid.x, poly.centroid.y)})
        return polygons

    def extract_blocks(self, target_layers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        blocks = []
        for entity in self.msp:
            if entity.dxftype() == 'INSERT':
                layer = entity.dxf.layer
                name = entity.dxf.name
                if target_layers and layer not in target_layers:
                    continue
                insert_pt = entity.dxf.insert
                raw_rot = entity.dxf.rotation % 360.0
                cardinal_angles = [0, 90, 180, 270, 360]
                closest = min(cardinal_angles, key=lambda a: abs(a - raw_rot))
                norm_rot = 0 if closest == 360 else closest
                blocks.append({"name": name, "layer": layer, "x": round(insert_pt.x * self.scale, 3), "y": round(insert_pt.y * self.scale, 3), "rotation_deg": round(raw_rot, 1), "cardinal_rotation": norm_rot})
        return blocks

    def calculate_wall_takeoff(self, wall_layers: List[str], wall_height_m: float = 2.80) -> Dict[str, Any]:
        segments = self.extract_lines_and_segments(wall_layers)
        total_len = sum(seg.length for seg in segments)
        center_len = total_len / 2.0
        return {"total_raw_linear_m": round(total_len, 2), "estimated_wall_centerline_m": round(center_len, 2), "wall_height_m": wall_height_m, "total_wall_area_m2": round(center_len * wall_height_m, 2)}

    def detect_wet_rooms_and_cladding(self, room_polygons: List[Dict[str, Any]], sanitary_blocks: List[Dict[str, Any]], cladding_height_m: float = 2.40) -> List[Dict[str, Any]]:
        if not room_polygons or not sanitary_blocks:
            return []
        sanitary_points = [sg.Point(b["x"], b["y"]) for b in sanitary_blocks]
        wet_rooms = []
        for i, room in enumerate(room_polygons):
            poly = room["geometry"]
            matching_pts = [pt for pt in sanitary_points if poly.contains(pt)]
            if matching_pts:
                perimeter = room["perimeter_m"]
                wet_rooms.append({"room_id": f"WetRoom_{i+1}", "floor_area_m2": room["area_m2"], "perimeter_m": perimeter, "cladding_height_m": cladding_height_m, "cladding_area_m2": round(perimeter * cladding_height_m, 2), "fixtures_count": len(matching_pts)})
        return wet_rooms

def compare_vector_delta(base_parser: DXFVectorParser, rev_parser: DXFVectorParser, tolerance_m: float = 0.15) -> Dict[str, Any]:
    base_blocks = base_parser.extract_blocks()
    rev_blocks = rev_parser.extract_blocks()
    added, removed, moved, unchanged = [], [], [], []
    matched_rev = set()
    for b in base_blocks:
        best_match, best_dist, best_idx = None, float("inf"), None
        for idx, r in enumerate(rev_blocks):
            if idx in matched_rev:
                continue
            if r["name"] == b["name"]:
                dist = math.hypot(r["x"] - b["x"], r["y"] - b["y"])
                if dist < best_dist:
                    best_dist, best_match, best_idx = dist, r, idx
        if best_match is not None:
            if best_dist <= tolerance_m:
                matched_rev.add(best_idx)
                unchanged.append({**b, "status": "Unchanged"})
            elif best_dist <= 3.0:
                matched_rev.add(best_idx)
                moved.append({"name": b["name"], "base_pos": (b["x"], b["y"]), "rev_pos": (best_match["x"], best_match["y"]), "move_distance_m": round(best_dist, 2), "rotation_delta_deg": round((best_match["rotation_deg"] - b["rotation_deg"]) % 360, 1), "status": "Moved"})
            else:
                removed.append({**b, "status": "Removed"})
        else:
            removed.append({**b, "status": "Removed"})
    for idx, r in enumerate(rev_blocks):
        if idx not in matched_rev:
            added.append({**r, "status": "Added"})
    base_lines = base_parser.extract_lines_and_segments()
    rev_lines = rev_parser.extract_lines_and_segments()
    envelope_breach = False
    if base_lines and rev_lines:
        b_bounds = sg.MultiLineString(base_lines).bounds
        r_bounds = sg.MultiLineString(rev_lines).bounds
        if max(abs(a - b) for a, b in zip(b_bounds, r_bounds)) > 0.20:
            envelope_breach = True
    return {"envelope_breach": envelope_breach, "added_blocks": added, "removed_blocks": removed, "moved_blocks": moved, "unchanged_blocks": unchanged, "summary": {"added_count": len(added), "removed_count": len(removed), "moved_count": len(moved), "unchanged_count": len(unchanged)}}