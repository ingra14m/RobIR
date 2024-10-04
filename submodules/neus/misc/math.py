import numpy as np


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh_arr, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    ... Can be 0 or more batch dimensions.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param sh: SH coeffs (..., C, (max degree + 1) ** 2)
    :param dirs: unit directions (..., 3)
    :return: (..., C)
    """

    sh = sh_arr

    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == len(sh)
    assert sh[0].shape[-1] == 3  # 3 color channels

    result = C0 * sh[0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[1] +
                  C1 * z * sh[2] -
                  C1 * x * sh[3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[4] +
                      C2[1] * yz * sh[5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[6] +
                      C2[3] * xz * sh[7] +
                      C2[4] * (xx - yy) * sh[8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[9] +
                          C3[1] * xy * z * sh[10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[13] +
                          C3[5] * z * (xx - yy) * sh[14] +
                          C3[6] * x * (xx - 3 * yy) * sh[15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[16] +
                              C4[1] * yz * (3 * xx - yy) * sh[17] +
                              C4[2] * xy * (7 * zz - 1) * sh[18] +
                              C4[3] * yz * (7 * zz - 3) * sh[19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[20] +
                              C4[5] * xz * (7 * zz - 3) * sh[21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[24])
    return result


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
    """Continuous learning rate decay function.

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    Args:
        step: int, the current optimization step.
        lr_init: float, the initial learning rate.
        lr_final: float, the final learning rate.
        max_steps: int, the number of steps during optimization.
        lr_delay_steps: int, the number of steps to delay the full learning rate.
        lr_delay_mult: float, the multiplier on the rate when delaying it.

    Returns:
        lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * np.log(mse)


def psnr_to_mse(psnr):
    """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
    return np.exp(-0.1 * np.log(10.) * psnr)
