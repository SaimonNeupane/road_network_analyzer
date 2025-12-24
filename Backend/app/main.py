from fastapi import FastAPI

app = FastAPI()


@app.get("/{district_name}")
def read_root():
    return {"Message": "Hello world "}
