from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
import pose_estimation
from utils.active_marking import active_marking


app = FastAPI()

@app.post("/actions")
async def actions_detections(video: UploadFile = File(...)):
    # Path for the temporary directory
    filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join("temp_videos", filename)
    os.makedirs("temp_videos", exist_ok=True)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Detect actions
        try:
            actions = pose_estimation.main(cam_input=temp_path)

            state = active_marking(actions)
        except Exception as e:
            return {"error": f"Error in pose estimation: {str(e)}"}

    except Exception as e:
        return {"error": f"Error saving video file: {str(e)}"}

    # Cleanup
    try:
        os.remove(temp_path)
    except Exception as e:
        return {"error": f"Error deleting temp file: {str(e)}"}

    return {
        "state": state,
        "action_detections": actions
    }
