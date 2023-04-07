# Open-vocabulary-Segment-Anything
An interesting demo by combining [OWL-ViT](https://arxiv.org/abs/2205.06230) of Google and [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/) of Meta!

![Detect Result](./outputs/owlvit_box.jpg)
![Segment Anything Result](./outputs/owlvit_segment_anything_output.jpg)

## Highlight
- Detect and Segment everything with Language!
- Detect objects in more details (small objects)
- Image-conditioned detection and Text-condition detection


## Catelog
- [x] OWL-ViT + Segment-Anything Demo
- [ ] Image-conditioned detection
- [ ] Huggingface Demo

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install OWL-ViT (the OWL-ViT is included in transformer library):

```bash
pip install transformer
```

More details can be found in [installation segment anything](https://github.com/facebookresearch/segment-anything#installation)

## Run Demo

- download segment-anything checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

- Run demo
```bash
bash run_demo.sh
```

## Reference
Please give applause for [IDEA-Research](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/segment_anything)