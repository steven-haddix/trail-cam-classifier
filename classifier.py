from __future__ import annotations

import os
from pathlib import Path
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(
        description="The type of animal or object detected (e.g., 'deer', 'fox', 'person', 'coyote', 'nothing')."
    )
    count: int = Field(
        description="The number of individuals of this type seen in the frames."
    )
    certainty: str = Field(description="Confidence level: 'high', 'medium', or 'low'.")


class ClassificationResult(BaseModel):
    detections: List[Detection] = Field(
        description="A list of distinct detections found in the images."
    )
    summary: str = Field(
        description="A brief explanation of what was seen across the 4 frames."
    )


def classify_tile(image_path: Path, api_key: str | None = None) -> ClassificationResult:
    """
    Classifies a 2x2 hero tile using Google Gemini 2.0 Flash.
    """
    client = genai.Client(api_key=api_key)

    # Read image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # System instruction for trail cam context
    prompt = """
    This image is a 2x2 grid (4 frames) from a wildlife trail camera. 
    Analyze the frames and identify any animals, people, or vehicles. 
    Be specific about animal species if possible (e.g., 'white-tailed deer' instead of just 'animal').
    If nothing of interest is found, return an empty list of detections with the label 'nothing'.
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificationResult,
            temperature=0,
        ),
    )

    # Gemini SDK with response_schema populated response.parsed
    if response.parsed:
        return response.parsed

    # Fallback if parsing failed
    try:
        # Pydantic 2.x
        return ClassificationResult.model_validate_json(response.text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return ClassificationResult(
            detections=[], summary=f"Failed to classify. Error: {e}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <path_to_tile>")
        sys.exit(1)

    tile_path = Path(sys.argv[1])
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    result = classify_tile(tile_path, api_key=api_key)
    print(result.model_dump_json(indent=2))
