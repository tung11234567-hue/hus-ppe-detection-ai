from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


def main() -> None:
    """Optional helper: download a Roboflow dataset in YOLOv8 format.

    Fill .env from .env.example first.
    """
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project_name = os.getenv("ROBOFLOW_PROJECT")
    version = os.getenv("ROBOFLOW_VERSION", "1")

    missing = [k for k, v in {
        "ROBOFLOW_API_KEY": api_key,
        "ROBOFLOW_WORKSPACE": workspace,
        "ROBOFLOW_PROJECT": project_name,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env variables: {', '.join(missing)}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(int(version)).download("yolov8", location="datasets/ppe")
    print(f"Downloaded dataset to: {Path(dataset.location).resolve()}")
    print("Update data/ppe.yaml if the folder structure is different.")


if __name__ == "__main__":
    main()
