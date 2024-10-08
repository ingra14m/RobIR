import os

def set_path(path, iter=200000):
    global NEUS_LOG_DIR, MESH_PATH, ENCODING
    ENCODING = "PE"
    NEUS_LOG_DIR = path
    MESH_PATH = os.path.join(path, "meshes/mesh_{:06d}.ply".format(iter))

    print("Load Mesh from {}, using {} Encoding".format(MESH_PATH, ENCODING))
    # if "hotdog" in CONF:
    #     NEUS_LOG_DIR = HOTDOG_LOG_DIR
    #     MESH_PATH = HOTDOG_MESH_PATH
    #     ENCODING = "PE"
    # elif CONF == "truck":
    #     NEUS_LOG_DIR = TRUCK_LOG_PATH
    #     MESH_PATH = TRUCK_MESH_PATH
    #     ENCODING = "PE"
    # else:
    #     raise NotImplementedError