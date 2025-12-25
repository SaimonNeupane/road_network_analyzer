from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{district_name}")
def read_root(district_name):
    longitude = 87.8942
    lattitude = 26.5455
    district = district_name
    dictt = {"longitude": longitude, "latitude": lattitude, "district": district}
    return dictt
