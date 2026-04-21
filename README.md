# PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting 🐶

[Alex Hanson*](https://www.cs.umd.edu/~hanson/), [Allen Tu*](https://tuallen.github.io), [Vasu Singla](https://vasusingla.github.io/), [Mayuka Jayawardhana](https://www.cs.umd.edu/people/mayukaj), [Matthias Zwicker](https://www.cs.umd.edu/~zwicker/), [Tom Goldstein](https://www.cs.umd.edu/~tomg/) 

[arXiv](https://arxiv.org/abs/2406.10219) | [Website](https://pup3dgs.github.io/)

*\* indicates equal contribution*

<img src="assets/teaser_figure.png" alt="Teaser Figure" />

This repository is the official code for the paper "PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting". It is built on top of the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) and [LightGaussian](https://github.com/VITA-Group/LightGaussian) codebases. Fisher computation in the included submodule is built using code from [FisherRF](https://github.com/JiangWenPL/FisherRF). If you make use of this code in your research, please consider citing these works in addition to ours.


**Abstract:** *Recent advances in novel view synthesis have enabled real-time rendering speeds with high reconstruction accuracy.
3D Gaussian Splatting (3D-GS), a foundational point-based parametric 3D scene representation, models scenes as large sets of 3D Gaussians.
However, complex scenes can consist of millions of Gaussians, resulting in high storage and memory requirements that limit the viability of 3D-GS on devices with limited resources. Current techniques for compressing these pretrained models by pruning Gaussians rely on combining heuristics to determine which Gaussians to remove. At high compression ratios, these pruned scenes suffer from heavy degradation of visual fidelity and loss of foreground details. In this paper, we propose a principled sensitivity pruning score that preserves visual fidelity and foreground details at significantly higher compression ratios than existing approaches. It is computed as a second-order approximation of the reconstruction error on the training views with respect to the spatial parameters of each Gaussian. Additionally, we propose a multi-round prune-refine pipeline that can be applied to any pretrained 3D-GS model without changing its training pipeline. After pruning 90% of Gaussians, a substantially higher percentage than previous methods, our PUP 3D-GS pipeline increases average rendering speed by 3.56× while retaining more salient foreground information and achieving higher image quality metrics than existing techniques on scenes from the Mip-NeRF 360, Tanks \& Temples, and Deep Blending datasets.*

## Setup
Follow the setup instruction for the original [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) codebase, including recursively cloning the repository submodules. 

```shell
conda env create --file environment.yml
conda activate gaussian_splatting_pup
```

We created an additional submodule for CUDA Fisher computation: `rasterization_and_pup_fisher`. Ensure that it is also cloned and installed.

## ROCm / Bayesian_3DGS integration

This fork is validated for use inside the `Bayesian_3DGS` ROCm container environment (`bhm_3dgs`).

- The shared GraphDECO-style dependencies `diff_gaussian_rasterization` and `simple_knn` are expected to come from the parent ROCm environment.
- The PUP-specific extensions `compress_diff_gaussian_rasterization` and `rasterization_and_pup_fisher` are built locally from this fork under ROCm.
- The goal of this port is to preserve PUP's pruning and Fisher logic while adapting the build/runtime layer for AMD GPUs. It is an integrated ROCm port for this repository workflow, not a claim of fresh-machine standalone portability.

## Running

### Full Pruning Pipeline

To prune and evaluate a scene, set the environment variables `SCENE_DATA_PATH`, `SCENE_MODEL_PATH`, and `SCENE_NAME`, then run:

```shell
bash scripts/full_pruning_pipeline.sh
```

* `SCENE_DATA_PATH` is the path to the COLMAP or NeRF Synthetic dataset.

* `SCENE_MODEL_PATH` is the path to the pretrained 3D-GS model.

* `SCENE_NAME` is unique name for the scene.

This script will create the `./experiments/<SCENE_NAME>_80` and `./experiments/<SCENE_NAME>_80_50` directories for one-round 80% and two-round 90% pruning, respectively. Then, it will save the prune-refined models inside the `point_cloud/iteration_35000` subdirectories. It will then render the test views using the 3D-GS `render.py` script, which has been updated to also output the average FPS in `fps.txt`. Finally, the image quality metrics of the test renders will be computed and saved in `results.json` using the 3D-GS `metrics.py` script.


### Generate Fishers

To create Fisher matrices of a scene, set the environment variables `SCENE_DATA_PATH` and `SCENE_MODEL_PATH`, then run:

```shell
bash scripts/create_fishers.sh
```

* `SCENE_DATA_PATH` is the path to the COLMAP or NeRF Synthetic dataset.

* `SCENE_MODEL_PATH` is the path to the pretrained 3D-GS model.

Fishers will be saved in the `fisher-pool-<X>/fishers_xyz_scaling` subdirectory of the pretrained model directory, where `X` is the training iteration.

## Results
<img src="assets/experiments_figure.png" alt="Experiments Figure"/>
<img src="assets/appendix_figure.png" alt="Appendix Figure"/>



<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@InProceedings{HansonTuPUP3DGS,
    author    = {Hanson, Alex and Tu, Allen and Singla, Vasu and Jayawardhana, Mayuka and Zwicker, Matthias and Goldstein, Tom},
    title     = {PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5949-5958},
    url       = {https://pup3dgs.github.io/}
}</code></pre>
  </div>
</section>

<h2 class="title">Related Work</h2>
<p>For additional papers on efficient 3D Gaussian Splatting, see our group’s related work below. If your
    research builds on ours, we encourage you to cite these papers.</p>
<ol>
    <li>
        <a href="https://speedysplat.github.io/" target="_blank" rel="noopener">
            <strong>Speedy-Splat</strong>
        </a>
        <em>(CVPR 2025)</em>
        <small><a href="https://speedysplat.github.io/#BibTeX" target="_blank">[BibTeX]</a></small>
        — Accelerate 3D Gaussian Splatting rendering speed by over 6× and reduce model size by over 90%
        through accurately localizing primitives during rasterization and pruning the scene during
        training, providing a significantly higher speedup than existing techniques while maintaining
        competitive image quality.
    </li>
    <li>
        <a href="https://speede3dgs.github.io/" target="_blank" rel="noopener">
            <strong>SpeeDe3DGS</strong>
        </a>
        <small><a href="https://speede3dgs.github.io/#BibTeX" target="_blank">[BibTeX]</a></small>
        — Boost DeformableGS rendering speed from 20 to 276 FPS using temporal sensitivity pruning and
        groupwise SE(3) motion distillation, all while preserving the superior image quality of per-Gaussian
        neural motion.
    </li>
</ol>

## Funding and Acknowledgments

This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA R\&D Contract No. 140D0423C0076. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Additional support was provided by ONR MURI program and the AFOSR MURI program. Commercial support was provided by Capital One Bank, the Amazon Research Award program, and Open Philanthropy. Zwicker was additionally supported by the National Science Foundation (IIS-2126407). Goldstein was additionally supported by the National Science Foundation (IIS-2212182) and by the NSF TRAILS Institute (2229885).
