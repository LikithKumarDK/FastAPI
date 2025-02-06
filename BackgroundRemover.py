import os
import time
import shutil
import torch
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from models.birefnet import BiRefNet
from utils import check_state_dict

# Define BASE_DIR for handling relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths and Constants
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_folder")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "BiRefNet-general-epoch_244.pth")
TEMP_FOLDER = os.path.join(BASE_DIR, "temp_files")  # Temporary folder for input files

# Ensure required directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# FastAPI application
app = FastAPI()

# Mount output folder for serving processed images
app.mount("/output_folder", StaticFiles(directory=OUTPUT_FOLDER), name="output_folder")

# Load Model (Ensure model path exists)
if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
birefnet = BiRefNet(bb_pretrained=False)

# Load model weights
try:
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)
    birefnet.to(device)
    birefnet.eval()
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Preprocessing pipeline
MODEL_INPUT_SIZE = (1024, 1024)
preprocess = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Image Processing Function
def process_image(image_path: str, output_image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        input_images = preprocess(image).unsqueeze(0).to(device)

        # Predict and generate the mask
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(original_size)

        # Add alpha channel (mask) to the image
        image.putalpha(mask)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        image.save(output_image_path, "PNG")
        print(f" Image processed and saved at: {output_image_path}")

    except Exception as e:
        print(f" Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


# Background Removal API Endpoint
@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    start_process_time = time.time()

    # Ensure filename is safe
    filename = os.path.basename(file.filename)
    input_image_path = os.path.join(TEMP_FOLDER, f"temp_{filename}")
    output_image_path = os.path.join(OUTPUT_FOLDER, f"no_bg_{filename}")

    try:
        # Save the uploaded file temporarily
        with open(input_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image
        process_image(input_image_path, output_image_path)

        processing_time = round(time.time() - start_process_time, 2)
        print(f" Processing time: {processing_time} seconds")

        # Construct the URL for the output image
        output_image_url = f"/output_folder/no_bg_{filename}"

        return JSONResponse({
            "image_url": output_image_url,
            "time_taken": processing_time
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        # Clean up the temporary input file
        if os.path.exists(input_image_path):
            os.remove(input_image_path)


# Serve the index.html file from the root folder
@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Index file not found")


# Serve the style.css file from the root folder
@app.get("/styles.css", response_class=FileResponse)
async def serve_css():
    css_path = os.path.join(BASE_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


# # Run the Application
# if __name__ == "__main__":
#     import uvicorn
#     print(" Starting FastAPI Server...")
#     port = int(os.getenv("PORT", 10000)) 
#     uvicorn.run(app, host="0.0.0.0", port=port)
