import yaml
import numpy as np

from src.data_models import GoupedPointCloud

def load_grouped_point_cloud(yaml_file: str) -> GoupedPointCloud:
    with open(yaml_file, "r", encoding="utf-8") as f:
        data: dict = yaml.safe_load(f)

    gp = GoupedPointCloud(
        positions = np.array(data.get("positions")),
        groups= data.get("groups")
    )

    if (s := data.get("scale")) is not None:
        gp.scale(np.array(s))

    return gp