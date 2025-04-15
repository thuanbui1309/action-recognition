from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
import pose_estimation

app = FastAPI()

@app.post("/actions/detect")
async def count_people(video: UploadFile = File(...)):
    # Path for the temporary directory
    filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join("temp_videos", filename)
    os.makedirs("temp_videos", exist_ok=True)

    # Save the uploaded video file to the temporary directory
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Detect actions
        try:
            result = pose_estimation.main(cam_input=temp_path)

        except Exception as e:
            return {"error": f"Error in pose estimation: {str(e)}"}

    except Exception as e:
        return {"error": f"Error saving video file: {str(e)}"}

    # Cleanup
    try:
        os.remove(temp_path)
    except Exception as e:
        return {"error": f"Error deleting temp file: {str(e)}"}

    return {"action_detections": result}
