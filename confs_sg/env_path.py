HOTDOG_LOG_DIR = r"neus_pretrain/nerf-synthesis/hotdog-neus"
TRUCK_LOG_PATH = r"neus_pretrain/blender-render/truck-neus"

HOTDOG_MESH_PATH = r"neus_pretrain/nerf-synthesis/hotdog-neus/clean.ply"
TRUCK_MESH_PATH = r"neus_pretrain/blender-render/truck-neus/clean.ply"


def set_path(CONF):
    global NEUS_LOG_DIR, MESH_PATH, ENCODING
    if "hotdog" in CONF:
        NEUS_LOG_DIR = HOTDOG_LOG_DIR
        MESH_PATH = HOTDOG_MESH_PATH
        ENCODING = "PE"
    elif CONF == "truck":
        NEUS_LOG_DIR = TRUCK_LOG_PATH
        MESH_PATH = TRUCK_MESH_PATH
        ENCODING = "PE"
    else:
        raise NotImplementedError