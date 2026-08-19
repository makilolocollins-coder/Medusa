from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, UnidentifiedImageError

from model import predict_image


# ============================================================
# MEDUSA API
# ============================================================

app = FastAPI(
    title="Medusa AI API",
    description="AI-assisted medical image analysis API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "app": "Medusa AI",
        "status": "online",
        "model": "MammoSense V2",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "MammoSense V2",
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Use JPEG, PNG or WEBP."
            ),
        )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        contents = await file.read()

        if len(contents) == 0:

            raise HTTPException(
                status_code=400,
                detail="Empty image file.",
            )


        # 10 MB upload limit
        if len(contents) > 10 * 1024 * 1024:

            raise HTTPException(
                status_code=413,
                detail="Image is larger than 10 MB.",
            )


        image = Image.open(
            BytesIO(contents)
        )


        # Force actual decoding
        image.load()


    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {str(e)}",
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = predict_image(image)


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "success": True,

        "model": "MammoSense V2",

        "filename": file.filename,

        "prediction": result["prediction"],

        "confidence": result["confidence"],

        "probabilities": result["probabilities"],
    }
