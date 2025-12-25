from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import osmnx as ox
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to store cached maps
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)


@app.get("/{district_name}")
def read_root(district_name: str):
    try:
        # Added .title() here too for consistency
        lat, lon = ox.geocode(f"{district_name.title()}, Nepal")
        return {
            "id": "1",
            "name": "saimon",
            "latitude": lat,
            "longitude": lon,
            "description": district_name,
            "category": "",
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Location not found: {str(e)}")


@app.get("/roads/{district_name}")
def get_roads(district_name: str):
    clean_name = district_name.title()
    # Define a filename for this district
    cache_file = os.path.join(CACHE_DIR, f"{clean_name}.geojson")

    try:
        if os.path.exists(cache_file):
            print(f"Loading {clean_name} from cache...")
            with open(cache_file, "r") as f:
                return json.load(f)

        print(f"Downloading {clean_name} from OSM (this may take time)...")
        place_query = f"{clean_name}, Nepal"

        G = ox.graph_from_place(place_query, network_type="drive", simplify=True)

        gdf_edges = ox.graph_to_gdfs(G, nodes=False)

        columns_to_keep = ["geometry", "name", "highway", "length"]
        available_cols = [c for c in columns_to_keep if c in gdf_edges.columns]
        gdf_edges = gdf_edges[available_cols]

        geojson_str = gdf_edges.to_json()
        with open(cache_file, "w") as f:
            f.write(geojson_str)

        return json.loads(geojson_str)

    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"District '{district_name}' not found on OpenStreetMap.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
