import os
import cv2
import numpy as np


def slice_at(img, vis, res=1024):
    s = (res + 2) * vis + 2
    img = img[s:s + res, 2:res + 2]
    return img / 255


def edge_detect(img):
    sobel_gradx = cv2.Sobel(img, -1, 1, 0)
    sobel_grady = cv2.Sobel(img, -1, 0, 1)
    sobel_grad = cv2.addWeighted(sobel_gradx, 0.5, sobel_grady, 0.5, 0)
    return sobel_grad


def mutual_info(log_dir):
    plot_dir = log_dir + r"\plots"
    renderings = os.listdir(plot_dir)
    rendering = sorted(renderings)[-1]
    rendering = plot_dir + rf"\{rendering}"
    rendering = cv2.imread(rendering)

    normal = slice_at(rendering, 0, rendering.shape[-2] - 4)
    light = slice_at(rendering, 1, rendering.shape[-2] - 4)
    albedo = slice_at(rendering, 2, rendering.shape[-2] - 4)

    normal_edge = edge_detect(normal)
    light_edge = edge_detect(light)
    albedo_edge = edge_detect(albedo)

    albedo_edge[normal_edge > 0.1] = 0
    res = (light_edge * albedo_edge).mean()

    mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

    return mse2psnr(res)
    # cv2.imshow("", light_edge)
    # cv2.waitKey()
