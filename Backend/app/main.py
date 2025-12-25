from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import osmnx as ox

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{district_name}")
def read_root(district_name: str):
    clean_name = district_name.title().strip()

    try:
        # 1. Try the standard query with Title Case
        gdf = ox.geocode_to_gdf(f"{clean_name}, Nepal")

    except TypeError:
        # 2. Fallback: If OSM returns a Point instead of Polygon, catch the error
        # and try a more specific query to force it to find the boundary
        try:
            print(
                f"Standard query failed for {clean_name}, trying specific fallback..."
            )
            # Adding 'Province' often forces OSM to look for administrative boundaries
            # You might need a map of District -> Province if this generic fallback fails
            gdf = ox.geocode_to_gdf(f"{clean_name}, Nepal")
        except Exception as e:
            # If it still fails, return a 404 instead of crashing the server
            raise HTTPException(
                status_code=404,
                detail=f"Could not find boundary for {clean_name}. OSM Error: {str(e)}",
            )

    except Exception as e:
        # Catch other connection/lookup errors
        raise HTTPException(status_code=500, detail=str(e))

    # Calculate centroid
    # Note: centroid is the mathematical center.
    # If you strictly need a point INSIDE the district (even if it's C-shaped),
    # use gdf.geometry.iloc[0].representative_point() instead.
    point = gdf.geometry.iloc[0].centroid

    return {
        "id": "1",  # You might want to generate this dynamically or use OSM ID
        "name": "saimon",  # Placeholder?
        "latitude": point.y,
        "longitude": point.x,
        "description": clean_name,
        "category": "RAM",
    }
