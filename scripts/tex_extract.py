import glob
import os
import cv2
import numpy as np
import trimesh
from PIL import Image


def slice_at(img, vis, res=1024):
    s = (res + 2) * vis + 2
    img = img[s:s + res, 2:res + 2]
    return img / 255


def extract_tex(log_dir):
    plot_dir = log_dir + r"\plots"
    renderings = os.listdir(plot_dir)
    rendering = sorted(renderings)[-1]
    rendering = plot_dir + rf"\{rendering}"
    rendering = cv2.imread(rendering)

    normal = slice_at(rendering, 0, rendering.shape[-2] - 4)
    albedo = slice_at(rendering, 2, rendering.shape[-2] - 4)
    roughness = slice_at(rendering, 3, rendering.shape[-2] - 4)
    metallic = slice_at(rendering, 4, rendering.shape[-2] - 4)  # no metallic now, -> 0

    return albedo, roughness, metallic, normal


def try_on_mesh(albedo, roughness, normal, mesh_file):
    mesh = trimesh.load(mesh_file)

    material = trimesh.visual.material.PBRMaterial(baseColorTexture=albedo,
                                                   metallicRoughnessTexture=roughness, normalTexture=normal)
    tex_visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, material=material)
    mesh.visual = tex_visual
    mesh.show()


def export_obj(mesh_file, log_dir, target_dir, target_name=None):
    mesh = trimesh.load(mesh_file)
    albedo, roughness, metallic, normal = extract_tex(log_dir)

    albedo = np.flip(albedo, -1) * 255
    albedo = Image.fromarray(albedo.astype(np.uint8))
    roughness = np.flip(roughness, -1) * 255
    roughness = Image.fromarray(roughness.astype(np.uint8))
    normal = np.flip(normal, -1) * 255
    normal = Image.fromarray(normal.astype(np.uint8))

    material = trimesh.visual.material.PBRMaterial(normalTexture=normal)
    tex_visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, material=material)
    mesh.visual = tex_visual

    obj, tex = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_color=True, include_texture=True,
                                               return_texture=True, write_texture=False, resolver=None, digits=8)

    if target_name is None:
        target_name = os.path.basename(mesh_file)
    if target_name[-4:] != ".obj":
        target_name = target_name + ".obj"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(os.path.join(target_dir, target_name), "w") as fp:
        fp.write(obj)

    for k in tex:
        with open(os.path.join(target_dir, k), "wb") as fp:
            fp.write(tex[k])

    with open(os.path.join(target_dir, "albedo.png"), "wb") as fp:
        albedo.save(fp)
    with open(os.path.join(target_dir, "roughness.png"), "wb") as fp:
        roughness.save(fp)
    with open(os.path.join(target_dir, "normal.png"), "wb") as fp:
        normal.save(fp)
    return
