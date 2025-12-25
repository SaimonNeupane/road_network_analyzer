from fastapi import FastAPI, HTTPException
import subprocess
from fastapi.middleware.cors import CORSMiddleware
import osmnx as ox
import json
import os
import sys
from pathlib import Path

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


@app.get("/roads/{place_name}/{district_name}")
def proposed_road(place_name: str, district_name: str):
    current_api_dir = Path(__file__).resolve().parent

    # Go one level up to find the project root
    # e.g., /home/saimon/Desktop/road_network_analyzer
    project_root = current_api_dir.parent

    # Define the exact location of run.py
    script_path = project_root / "run.py"

    # Define where the output file will be (inside the root's 'outputs' folder)
    output_file_path = project_root / "outputs" / "recommendations.json"

    # ---------------------------------------------------------
    # 2. SAFETY CHECKS
    # ---------------------------------------------------------
    if not script_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Server Config Error: Cannot find script at {script_path}",
        )

    # ---------------------------------------------------------
    # 3. RUN THE SCRIPT
    # ---------------------------------------------------------
    try:
        # We use cwd=project_root so the script finds 'nepal_rural_data.csv', 'artifacts', etc.
        result = subprocess.run(
            [sys.executable, str(script_path), district_name, place_name],
            cwd=project_root,  # <--- CRITICAL: Runs the command from the project root
            capture_output=True,
            text=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Subprocess failed to start: {str(e)}"
        )

    # ---------------------------------------------------------
    # 4. HANDLE ERRORS (Script Crash)
    # ---------------------------------------------------------
    if result.returncode != 0:
        # Log the full error to your console for debugging
        print(f"SCRIPT ERROR LOG:\n{result.stderr}")

        # Return the error in the API response
        raise HTTPException(
            status_code=500, detail=f"Inference Script Crash: {result.stderr.strip()}"
        )

    # ---------------------------------------------------------
    # 5. READ & RETURN DATA
    # ---------------------------------------------------------
    if not output_file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Script finished successfully, but 'outputs/recommendations.json' was not generated.",
        )

    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail="The generated JSON file is corrupted or empty."
        )
