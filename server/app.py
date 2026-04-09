from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Pandemic RL Server Running"}


# ✅ REQUIRED main function
def main():
    return app


# ✅ REQUIRED entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)