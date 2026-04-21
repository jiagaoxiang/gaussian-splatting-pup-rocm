import numpy as np
import pytest
import torch

from scene.cameras import Camera
from utils import general_utils as gu


def _cuda_only():
    return not torch.cuda.is_available()


def test_pick_default_device_prefers_current_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.1.0", raising=False)

    assert str(gu.pick_default_device()) == "cuda:0"


def test_pick_default_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert str(gu.pick_default_device()) == "cpu"


def test_resolve_tensor_device_uses_reference_tensor():
    reference = torch.zeros(1)

    assert gu.resolve_tensor_device(reference) == reference.device


def test_make_background_tensor_uses_resolved_device():
    background = gu.make_background_tensor([0, 0, 0], reference=torch.zeros(1))

    assert background.device == torch.device("cpu")
    assert background.dtype == torch.float32
    torch.testing.assert_close(background, torch.tensor([0.0, 0.0, 0.0]))


def test_strip_lowerdiag_preserves_input_device():
    lower = torch.ones((2, 3, 3))

    uncertainty = gu.strip_lowerdiag(lower)

    assert uncertainty.device == lower.device


def test_build_rotation_preserves_input_device():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    rotation = gu.build_rotation(quat)

    assert rotation.device == quat.device


def test_build_scaling_rotation_preserves_input_device():
    scales = torch.tensor([[1.0, 2.0, 3.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    transform = gu.build_scaling_rotation(scales, quat)

    assert transform.device == scales.device


def test_camera_respects_requested_cpu_device(monkeypatch):
    def fail_cuda_call(self, *args, **kwargs):
        raise AssertionError("Camera should not call Tensor.cuda() for CPU mode")

    monkeypatch.setattr(torch.Tensor, "cuda", fail_cuda_call, raising=False)

    image = torch.ones((3, 2, 2), dtype=torch.float32)
    camera = Camera(
        colmap_id=0,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
        FoVx=0.5,
        FoVy=0.5,
        image=image,
        gt_alpha_mask=None,
        image_name="test",
        uid=0,
        data_device="cpu",
    )

    assert camera.original_image.device == torch.device("cpu")
    assert camera.world_view_transform.device == torch.device("cpu")
    assert camera.projection_matrix.device == torch.device("cpu")


@pytest.mark.skipif(_cuda_only(), reason="Requires ROCm/CUDA runtime")
def test_compress_rasterizer_zero_gaussians_smoke():
    from compress_diff_gaussian_rasterization import (
        GaussianRasterizationSettings as CountSettings,
        GaussianRasterizer as CountRasterizer,
    )

    device = torch.device("cuda:0")
    zero3 = torch.zeros((0, 3), device=device)
    zero2 = torch.zeros((0, 3), device=device)
    zero1 = torch.zeros((0, 1), device=device)
    scales = torch.zeros((0, 3), device=device)
    rots = torch.zeros((0, 4), device=device)
    colors = torch.zeros((0, 3), device=device)
    mat = torch.eye(4, device=device)
    campos = torch.zeros((3,), device=device)
    bg = torch.zeros((3,), device=device)

    settings = CountSettings(
        image_height=4,
        image_width=4,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=mat,
        projmatrix=mat,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        f_count=True,
    )
    rasterizer = CountRasterizer(settings)
    output = rasterizer.forward_count(
        means3D=zero3,
        means2D=zero2,
        opacities=zero1,
        colors_precomp=colors,
        scales=scales,
        rotations=rots,
    )

    assert isinstance(output, tuple)
    assert len(output) == 4


@pytest.mark.skipif(_cuda_only(), reason="Requires ROCm/CUDA runtime")
def test_fisher_rasterizer_zero_gaussians_smoke():
    from rasterization_and_pup_fisher import (
        GaussianRasterizationSettings as FisherSettings,
        GaussianRasterizer as FisherRasterizer,
    )

    device = torch.device("cuda:0")
    zero3 = torch.zeros((0, 3), device=device)
    zero2 = torch.zeros((0, 3), device=device)
    zero1 = torch.zeros((0, 1), device=device)
    scales = torch.zeros((0, 3), device=device)
    rots = torch.zeros((0, 4), device=device)
    colors = torch.zeros((0, 3), device=device)
    mat = torch.eye(4, device=device)
    campos = torch.zeros((3,), device=device)
    bg = torch.zeros((3,), device=device)

    settings = FisherSettings(
        image_height=4,
        image_width=4,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=mat,
        projmatrix=mat,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = FisherRasterizer(settings)
    output = rasterizer(
        means3D=zero3,
        means2D=zero2,
        opacities=zero1,
        colors_precomp=colors,
        scales=scales,
        rotations=rots,
        fishers=torch.zeros((0, 21), device=device),
    )

    assert isinstance(output, tuple)
    assert len(output) == 2
