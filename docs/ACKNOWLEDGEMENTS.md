# Acknowledgements

SemanticVLA is built on a long chain of excellent open-source work. We are grateful to the authors of the following projects.

## Foundational code

- [**StarVLA**](https://github.com/StarVLA) — the Qwen-VL + DiT action backbone implementation that we use as the foundation of our model. The `baseframework` loader, the DiT-B flow-matching action head, and the multi-stream Qwen-VL dataloader all derive from StarVLA.
- [**UniVLA**](https://github.com/OpenDriveLab/UniVLA) — the latent-action token formulation that motivates our trace-conditioned LAM. Our LAM head structure and the language-side latent action token prediction are inspired by UniVLA.

## Data and benchmarks

- [**LeRobot**](https://github.com/huggingface/lerobot) — the v3 chunked Parquet/video data format used by all four released TraceX 240K datasets.
- [**Open X-Embodiment**](https://robotics-transformer-x.github.io/) — provides BridgeData V2, RT-1/Fractal, and BC-Z in a unified RLDS format.
- [**BridgeData V2**](https://rail-berkeley.github.io/bridgedata/) — Bridge WidowX trajectories used for both LAM and Bridge VLA training.
- [**RT-1 / Fractal**](https://robotics-transformer1.github.io/) — Google Robot trajectories used in the unified OXE LAM.
- [**BC-Z**](https://sites.google.com/view/bc-z/home) — xArm trajectories used in the unified OXE LAM.
- [**DROID**](https://droid-dataset.github.io/) — Franka Panda trajectories included in the TraceX 240K release.
- [**LIBERO**](https://github.com/Lifelong-Robot-Learning/LIBERO) — the LIBERO suite used for evaluation and finetuning.
- [**SimplerEnv**](https://github.com/simpler-env/SimplerEnv) — the SimplerEnv WidowX evaluation harness used for our Bridge checkpoint.

## Vision backbones

- [**DINOv2**](https://github.com/facebookresearch/dinov2) — frozen visual encoder used inside the LAM.
- [**Qwen-VL**](https://github.com/QwenLM/Qwen3-VL) — Qwen3-VL-4B-Instruct used as the VLA's VLM backbone.

## Trace annotation pipeline

- [**Molmo**](https://huggingface.co/allenai/Molmo-72B-0924) — open VLM used in Stage 1 of the trace annotation pipeline to point at the robot gripper on sparse keyframes.
- [**CoTracker**](https://github.com/facebookresearch/co-tracker) — dense point tracker used in Stage 2 of the trace annotation pipeline to propagate the Molmo keyframes into per-frame dense traces.

## Tools

- [**Hugging Face Hub**](https://huggingface.co/) — model and dataset hosting.

If we have inadvertently omitted a project that you believe should be credited here, please open an issue on the [GitHub repo](https://github.com/Fei-Ni/SemanticVLA_Offcial) and we will update this page.
