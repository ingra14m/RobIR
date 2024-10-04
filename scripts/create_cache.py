from model.texture_model import TextureCache

# The mesh path, we use it to generate the cache we need for HDR-Factor. We have already provided the result
mesh_list = [
    # r"D:\Git_Project\NeuS-Pytorch\logs\cup-neus\meshes\clean.ply",
]

for MESH_PATH in mesh_list:
    tex_cache = TextureCache(MESH_PATH)
    tex_cache.render_basics(2048)

if __name__ == '__main__':
    pass
