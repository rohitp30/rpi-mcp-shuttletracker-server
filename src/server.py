#!/usr/bin/env python3
import os
import math
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import requests
import numpy as np
from fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

ROUTES_JSON_PATH = os.environ.get("ROUTES_JSON_PATH", "data/routes.json")
SCHEDULE_JSON_PATH = os.environ.get("SCHEDULE_JSON_PATH", "data/schedule.json")
SHUTTLE_LOCATIONS_URL = os.environ.get("SHUTTLE_LOCATIONS_URL", "https://shuttles.rpi.edu/api/locations")

DEFAULT_SPEED_MPS = float(os.environ.get("DEFAULT_SPEED_MPS", "6"))  # ~13 mph
AT_STOP_THRESHOLD_M = float(os.environ.get("AT_STOP_THRESHOLD_M", "25"))
ROUTE_AMBIGUITY_THRESHOLD_M = float(os.environ.get("ROUTE_AMBIGUITY_THRESHOLD_M", "20"))

# -----------------------------------------------------------------------------
# MCP server
# -----------------------------------------------------------------------------

mcp = FastMCP("RPI Shuttle MCP Server")

# Keep your sample tools for sanity checks ------------------------------------------------

@mcp.tool(description="Greet a user by name with a welcome message from the MCP server")
def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to our RPI Shuttle MCP server"

@mcp.tool(description="Get information about the MCP server including name, version, environment, and Python version")
def get_server_info() -> dict:
    return {
        "server_name": "RPI Shuttle MCP Server",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "python_version": os.sys.version.split()[0]
    }

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

EARTH_R = 6371000.0  # meters

def to_rad(d: float) -> float:
    return d * math.pi / 180.0

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    lat1r, lat2r = to_rad(lat1), to_rad(lat2)
    s = math.sin(dlat/2)**2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(s))

def nearest_point_on_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]):
    # Treat lat/lon locally as planar (ok for small campus distances)
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    px, py = p[1], p[0]
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx*abx + aby*aby
    t = 0.0 if ab2 == 0 else (apx*abx + apy*aby) / ab2
    t = max(0.0, min(1.0, t))
    proj = (ay + t*aby, ax + t*abx)  # (lat, lon)
    dist = haversine_m(p, proj)
    return proj, t, dist

def cumulative_lengths(poly: List[Tuple[float, float]]) -> List[float]:
    acc = [0.0]
    for i in range(1, len(poly)):
        acc.append(acc[-1] + haversine_m(poly[i-1], poly[i]))
    return acc

# -----------------------------------------------------------------------------
# Route data loader & index
# -----------------------------------------------------------------------------

class RouteIndex:
    """
    Parses your routes.json (the big structure you pasted).
    Concatenates each route's ROUTES segments into one continuous polyline in loop order.
    Indexes stops by id, alias (NAME), and lowercase variants.
    """

    def __init__(self, routes_json: Dict[str, Any]):
        self.routes_raw = routes_json
        self.routes: Dict[str, Dict[str, Any]] = {}   # key: route_name ("NORTH"/"WEST")
        self.stop_alias: Dict[str, Tuple[str, str]] = {}  # normalized_name -> (route_name, stop_key)

        # detect active routes if schedule is available, otherwise include NORTH/WEST if present
        self.active = [k for k in routes_json.keys() if k in ("NORTH", "WEST")]

        for rname in self.active:
            r = routes_json[rname]
            # concat segments
            segments = r.get("ROUTES", [])
            points: List[Tuple[float, float]] = []
            for seg in segments:
                for lat, lon in seg:
                    points.append((float(lat), float(lon)))

            # build stop list (both STOPS and POLYLINE_STOPS are useful)
            stop_keys: List[str] = r.get("STOPS", [])
            polyline_stop_keys: List[str] = r.get("POLYLINE_STOPS", stop_keys)

            stops: Dict[str, Dict[str, Any]] = {}
            for key in set(stop_keys + polyline_stop_keys):
                if key in ("ROUTES", "STOPS", "POLYLINE_STOPS", "COLOR"):
                    continue
                node = r.get(key)
                if not isinstance(node, dict):
                    continue
                coords = node.get("COORDINATES")
                if not coords or len(coords) != 2:
                    continue
                lat, lon = float(coords[0]), float(coords[1])
                name = node.get("NAME", key.replace("_", " ").title())
                stops[key] = {"id": key, "name": name, "lat": lat, "lon": lon}

                # index aliases
                for alias in {key, name, name.replace(" (Return)", ""), key.replace("_", " ")}:
                    self.stop_alias[alias.lower()] = (rname, key)

            self.routes[rname] = {
                "name": rname,
                "color": r.get("COLOR"),
                "poly": points,   # continuous polyline
                "stops": stops,   # dict of stop_key -> {id,name,lat,lon}
            }

        # precompute accumulative lengths
        for rname, r in self.routes.items():
            r["acc"] = cumulative_lengths(r["poly"])
            r["length_m"] = r["acc"][-1] if r["acc"] else 0.0

    def all_routes(self) -> List[str]:
        return list(self.routes.keys())

    def nearest_on_route(self, rname: str, p: Tuple[float, float]) -> Tuple[float, int, float, Tuple[float, float]]:
        """Return (distance_m, seg_index, t, snapped_point) on a given route."""
        r = self.routes[rname]
        poly = r["poly"]
        best = (float("inf"), 0, 0.0, poly[0] if poly else p)
        for i in range(len(poly) - 1):
            proj, t, d = nearest_point_on_segment(p, poly[i], poly[i+1])
            if d < best[0]:
                best = (d, i, t, proj)
        return best

    def nearest_route(self, p: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Pick the closest route by snapped distance; if ambiguous within threshold, return None."""
        cand = []
        for rname in self.all_routes():
            d, i, t, proj = self.nearest_on_route(rname, p)
            cand.append((d, rname, i, t, proj))
        if not cand:
            return None
        cand.sort(key=lambda x: x[0])
        if len(cand) > 1:
            # ambiguity check: if the snapped points are within ROUTE_AMBIGUITY_THRESHOLD_M, say ambiguous
            p0 = cand[0][4]
            p1 = cand[1][4]
            if haversine_m(p0, p1) < ROUTE_AMBIGUITY_THRESHOLD_M:
                return None
        d, rname, i, t, proj = cand[0]
        return {"route": rname, "dist_m": d, "seg_index": i, "t": t, "snapped": proj}

    def distance_along(self, rname: str, seg_index: int, t: float) -> float:
        """Distance from start of the polyline to the snapped point."""
        r = self.routes[rname]
        poly = r["poly"]
        acc = r["acc"]
        base = acc[seg_index]
        seg_len = haversine_m(poly[seg_index], poly[seg_index+1])
        return base + t * seg_len

    def nearest_stop(self, rname: str, p: Tuple[float, float]) -> Tuple[str, Dict[str, Any], float]:
        """Return (stop_key, stop_dict, distance_m) for the nearest stop on route."""
        r = self.routes[rname]
        best_key, best_stop, best_d = None, None, float("inf")
        for key, s in r["stops"].items():
            d = haversine_m(p, (s["lat"], s["lon"]))
            if d < best_d:
                best_key, best_stop, best_d = key, s, d
        return best_key, best_stop, best_d

    def find_stop(self, user_text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        Resolve a stop by name/id across routes.
        Returns (route_name, stop_key, stop_dict) or None.
        """
        if not user_text:
            return None
        key = user_text.strip().lower()
        # direct alias hit
        if key in self.stop_alias:
            rname, skey = self.stop_alias[key]
            return rname, skey, self.routes[rname]["stops"][skey]
        # fuzzy-ish pass: startswith
        for alias, (rname, skey) in self.stop_alias.items():
            if alias.startswith(key):
                return rname, skey, self.routes[rname]["stops"][skey]
        return None

    def poly_index_of_stop(self, rname: str, stop_key: str) -> int:
        """
        Use the vertex of the polyline closest to the stop coordinate.
        """
        r = self.routes[rname]
        target = (r["stops"][stop_key]["lat"], r["stops"][stop_key]["lon"])
        best_i, best_d = 0, float("inf")
        for i, pt in enumerate(r["poly"]):
            d = haversine_m(pt, target)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def forward_distance(self, rname: str, from_dist: float, to_vertex_index: int) -> float:
        """Distance going forward along the loop from from_dist to the vertex index (wrap-around if needed)."""
        r = self.routes[rname]
        acc = r["acc"]
        loop_len = r["length_m"]
        to_dist = acc[to_vertex_index]
        forward = to_dist - from_dist
        if forward < 0:
            forward += loop_len
        return forward

# -----------------------------------------------------------------------------
# Data bootstrapping
# -----------------------------------------------------------------------------

def load_routes() -> RouteIndex:
    # Prefer file; if not present, allow embedding via env var ROUTES_JSON (stringified JSON)
    data: Dict[str, Any]
    if os.path.exists(ROUTES_JSON_PATH):
        with open(ROUTES_JSON_PATH, "r") as f:
            data = json.load(f)
    else:
        embedded = os.environ.get("ROUTES_JSON")
        if not embedded:
            raise RuntimeError("No routes.json found. Provide data/routes.json or set ROUTES_JSON env var.")
        data = json.loads(embedded)
    return RouteIndex(data)

ROUTES = load_routes()

# -----------------------------------------------------------------------------
# Shuttle API
# -----------------------------------------------------------------------------

@mcp.tool(description="Fetch current shuttle vehicle positions from the RPI shuttle API")
def get_vehicles() -> dict:
    """
    Returns:
      {
        'vehicles': [
           { 'id': '1', 'lat': 42.73, 'lon': -73.67, 'updated_at': '...' },
           ...
        ],
        'fetched_at': epoch_seconds
      }
    """
    try:
        r = requests.get(SHUTTLE_LOCATIONS_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"failed to fetch vehicles: {e}"}

    arr = data if isinstance(data, list) else (data.get("vehicles") or data.get("data") or [])
    vehicles = []
    for i, v in enumerate(arr):
        vehicles.append({
            "id": str(v.get("id") or v.get("vehicle_id") or v.get("name") or i),
            "lat": float(v.get("lat") or v.get("latitude")),
            "lon": float(v.get("lon") or v.get("lng") or v.get("long") or v.get("longitude")),
            "updated_at": v.get("updated_at") or v.get("timestamp") or v.get("last_seen")
        })
    vehicles = [v for v in vehicles if math.isfinite(v["lat"]) and math.isfinite(v["lon"])]
    return {"vehicles": vehicles, "fetched_at": int(time.time())}

# -----------------------------------------------------------------------------
# Core tools: route inference, stop check, ETA
# -----------------------------------------------------------------------------

@mcp.tool(description="Infer whether a coordinate lies on the NORTH or WEST route (with nearest stop)")
def infer_route(lat: float, lon: float) -> dict:
    p = (float(lat), float(lon))
    best = ROUTES.nearest_route(p)
    if not best:
        return {"ambiguous": True, "message": "Point is ambiguous between routes (too close to both)."}
    rname = best["route"]
    nearest_stop_key, nearest_stop, stop_d = ROUTES.nearest_stop(rname, best["snapped"])
    return {
        "ambiguous": False,
        "route": rname,
        "snapped_dist_m": round(best["dist_m"], 2),
        "nearest_stop": {
            "id": nearest_stop_key,
            "name": nearest_stop["name"],
            "distance_m": round(stop_d, 2)
        }
    }

@mcp.tool(description="Check if a coordinate is currently at a known stop (within threshold meters)")
def is_at_stop(lat: float, lon: float, threshold_m: float = AT_STOP_THRESHOLD_M) -> dict:
    p = (float(lat), float(lon))
    # fast pass: pick nearest route first (or scan both if ambiguous)
    routes_to_check = ROUTES.all_routes()
    result = []
    for rname in routes_to_check:
        key, stop, d = ROUTES.nearest_stop(rname, p)
        if d <= threshold_m:
            result.append({"route": rname, "stop_id": key, "stop_name": stop["name"], "distance_m": round(d, 2)})
    return {"at_any_stop": bool(result), "matches": result, "threshold_m": threshold_m}

@mcp.tool(description="Estimate ETA (seconds) from a coordinate to a named stop along the inferred route loop")
def eta_to_stop(stop_name: str, lat: float, lon: float, speed_mps: float = DEFAULT_SPEED_MPS) -> dict:
    if speed_mps <= 0:
        speed_mps = DEFAULT_SPEED_MPS

    p = (float(lat), float(lon))
    # Find the stop record (route + stop key)
    resolved = ROUTES.find_stop(stop_name)
    if not resolved:
        return {"error": f"Unknown stop '{stop_name}'."}
    target_route, stop_key, stop_data = resolved

    # Snap the point to the best route
    best = ROUTES.nearest_route(p)
    if not best:
        return {"ambiguous": True, "message": "Point is ambiguous between routes (too close to both)."}

    rname = best["route"]
    if rname != target_route:
        # If user asked for a stop on a different route, be explicit
        return {"error": f"Stop '{stop_data['name']}' is on {target_route}, but the point is on {rname}."}

    from_dist = ROUTES.distance_along(rname, best["seg_index"], best["t"])
    to_vertex = ROUTES.poly_index_of_stop(rname, stop_key)
    forward_m = ROUTES.forward_distance(rname, from_dist, to_vertex)
    eta_s = int(round(forward_m / float(speed_mps)))

    return {
        "ambiguous": False,
        "route": rname,
        "stop": {"id": stop_key, "name": stop_data["name"], "lat": stop_data["lat"], "lon": stop_data["lon"]},
        "distance_m": int(round(forward_m)),
        "speed_mps": float(speed_mps),
        "eta_seconds": eta_s,
        "eta_human": f"{eta_s//60} min {eta_s%60} sec"
    }

# -----------------------------------------------------------------------------
# Convenience: vehicle-aware wrappers (optional for Poke prompts)
# -----------------------------------------------------------------------------

def _first_vehicle_latlon() -> Optional[Tuple[float, float, str]]:
    data = get_vehicles()
    if "vehicles" not in data or not data["vehicles"]:
        return None
    v = data["vehicles"][0]
    return (v["lat"], v["lon"], v["id"])

@mcp.tool(description="Where is the shuttle? (uses the first available vehicle)")
def where_is_the_shuttle() -> dict:
    v = _first_vehicle_latlon()
    if not v:
        return {"error": "No vehicles online"}
    lat, lon, vid = v
    ir = infer_route(lat=lat, lon=lon)
    at = is_at_stop(lat=lat, lon=lon, threshold_m=AT_STOP_THRESHOLD_M)
    return {"vehicle_id": vid, "position": {"lat": lat, "lon": lon}, "route_info": ir, "at_stop": at}

@mcp.tool(description="ETA to a stop for the first available shuttle vehicle (fallback speed if needed)")
def eta_for_first_vehicle(stop_name: str, speed_mps: float = DEFAULT_SPEED_MPS) -> dict:
    v = _first_vehicle_latlon()
    if not v:
        return {"error": "No vehicles online"}
    lat, lon, vid = v
    res = eta_to_stop(stop_name=stop_name, lat=lat, lon=lon, speed_mps=speed_mps)
    res["vehicle_id"] = vid
    return res

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    print(f"Starting FastMCP server on {host}:{port}")

    mcp.run(
        transport="http",
        host=host,
        port=port,
        stateless_http=True
    )
