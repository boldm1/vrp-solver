import os
from typing import List

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_distance_matrix(origins: List[str], destinations: List[str]) -> dict:
    """
    Fetch matrix containing distance and travel time between between each origin and all
    requested destinations using Google Distance Matrix API.

    Args:
        origins (List[str])
        destinations (List[str])

    Returns:
        dict: Dictionary with 'distance_text', 'distance_value', 'duration_text', 'duration_value'
    """

    # Enforce usage limits
    # https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing
    if len(origins) > 25:
        raise ValueError("Too many origins! Max 25 per request.")
    if len(destinations) > 25:
        raise ValueError("Too many destinations! Max 25 per request.")
    if len(origins) * len(destinations) > 100:
        raise ValueError(
            "Max 100 elements per request! (1 element = 1 origin * 1 destination)"
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": ", ".join(origins),
        "destinations": ", ".join(destinations),
        "units": "metric",
        "key": api_key,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise error if request failed

    data = response.json()

    try:
        element = data["rows"][0]["elements"][0]
        return {
            "distance_text": element["distance"]["text"],
            "distance_value": element["distance"]["value"],  # in meters
            "duration_text": element["duration"]["text"],
            "duration_value": element["duration"]["value"],  # in seconds
        }

    except (KeyError, IndexError):
        return {"error": "Invalid response format", "raw": data}
