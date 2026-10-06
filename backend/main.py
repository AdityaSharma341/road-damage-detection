from backend.database import engine, SessionLocal
from backend.models import Base, ImageRecord
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI,UploadFile,File
from ultralytics import YOLO

app=FastAPI()
Base.metadata.create_all(bind=engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model=YOLO("runs/detect/runs/road_damage_detection_v2-2/weights/best.pt")

@app.get("/")
def home():
    return FileResponse("frontend.html")

## Upload Image

@app.post("/upload-image")
def upload_image(file:UploadFile=File(...)):
    with open("backend/Uploads/"+file.filename,"wb") as f:
        f.write(file.file.read())
        
    return {
        "message": "Image Uploaded Successfully",
        "filename": file.filename
    }

## Predict
@app.post("/predict")
def predict(filename: str):

    image_path = "backend/Uploads/" + filename

    output_path = "backend/outputs/prediction/" + filename

    model.predict(
        source=image_path,
        save=True,
        project=r"D:\Damage Road Detection\data\backend\outputs",
        name="prediction",
        exist_ok=True
    )

    db = SessionLocal()

    record = ImageRecord(
        input_image=image_path,
        output_image=output_path
    )

    db.add(record)
    db.commit()
    db.close()

    return {
        "message": "Prediction Completed",
        "output_image": output_path
    }


## save image
@app.get("/output")
def output(filename: str):
    file_path = Path("backend/outputs/prediction/" + filename)

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found."
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )
    
@app.get("/show-image")
def show_image(filename: str):

    file_path = Path("backend/outputs/prediction/" + filename)

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found."
        )

    return FileResponse(
        path=file_path,
        media_type="image/jpeg"
    )

@app.get("/show_all")
def show_predictions():

    db = SessionLocal()

    records = db.query(ImageRecord).all()

    db.close()

    result = []

    for record in records:

        result.append({
            "id": record.id,
            "input_image": record.input_image,
            "output_image": record.output_image
        })

    return {
        "total_images": len(result),
        "images": result
    }

@app.get("/history")
def history():

    db = SessionLocal()

    records = db.query(ImageRecord).all()

    result = []

    for row in records:

        result.append({
            "id": row.id,
            "input_image": row.input_image,
            "output_image": row.output_image
        })

    db.close()

    return result