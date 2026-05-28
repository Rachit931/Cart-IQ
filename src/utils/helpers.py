import json

import joblib


def save_artifact(
    artifact,
    path: str,
):
    """
    Save artifact using joblib.
    """

    joblib.dump(
        artifact,
        path,
    )

    print(f"Saved artifact: {path}")


def save_metadata(
    metadata: dict,
    path: str,
):
    """
    Save metadata/configuration
    as JSON file.
    """

    with open(
        path,
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"Saved Metadata: {path}")
