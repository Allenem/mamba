# 1. Mambar-related code

Contents:
- [1. Mambar-related code](#1-mambar-related-code)
  - [1.1. NLP mission](#11-nlp-mission)
  - [1.2. Vision mission](#12-vision-mission)
    - [1.2.1. Mamba Installation](#121-mamba-installation)
    - [1.2.2. Comparison of different models](#122-comparison-of-different-models)
      - [1.2.2.1. Unet](#1221-unet)
      - [1.2.2.2. SwinUnet](#1222-swinunet)
      - [1.2.2.3. VM-Unet](#1223-vm-unet)

## 1.1. NLP mission

- [NLP mission model(./NLPMission/model.py)](./NLPMission/model.py)

neglect

## 1.2. Vision mission

- [Vision mission model(./VisionMission/vmunet2d.py)](./VisionMission/vmunet2d.py)

### 1.2.1. Mamba Installation

**Ubuntu**

https://blog.csdn.net/weixin_43328758/article/details/137269102

```shell
conda create -n <your_env_name> python=3.10.13
conda activate <your_env_name>
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install causal-conv1d
pip install mamba-ssm
```

**Windows**

https://blog.csdn.net/m0_59115667/article/details/137794459

https://blog.csdn.net/weixin_45659168/article/details/137862487

### 1.2.2. Comparison of different models

|                  | Unet     | SwinUnet | VMUnet   |
| ---------------- | -------- | -------- | -------- |
| Parameters       | 31.04M   | 41.6M    | 44.27M   |
| Time<sup>1</sup> | 0.00563s | 0.01432s | 0.02219s |

<sup>1</sup> Time per epoch

---

<details open>
<summary>View contents</summary>

#### 1.2.2.1. Unet

**Model**

<details>
<summary>View contents</summary>

```
Unet(
  (down_blocks): ModuleList(
    (0): UnetDownBlock(
      (convs): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): UnetDownBlock(
      (convs): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): UnetDownBlock(
      (convs): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (3): UnetDownBlock(
      (convs): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (4): UnetDownBlock(
      (convs): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
      )
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (up_blocks): ModuleList(
    (0): UnetUpBlock(
      (upsample): ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
      (convs): Sequential(
        (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): UnetUpBlock(
      (upsample): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
      (convs): Sequential(
        (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): UnetUpBlock(
      (upsample): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
      (convs): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): UnetUpBlock(
      (upsample): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
      (convs): Sequential(
        (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (final_block): Sequential(
    (0): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

</details>

**Parameters**

Windows 1 epoch forward
```
Number of parameters: 31.04M
input size: torch.Size([2, 3, 256, 256])
Used time: 0.17s
output size: torch.Size([2, 1, 256, 256])
```

Linux 1000 epoches forward
```
Number of parameters: 31.04M
input size: torch.Size([2, 3, 256, 256])
Used time: 5.63s
output size: torch.Size([2, 1, 256, 256])
```

1 epoch: 5.63s / 1000 = 0.00563s

---

#### 1.2.2.2. SwinUnet

**Model**

<details>
<summary>View contents</summary>

```
SwinUnet(
  (swin_unet): SwinTransformerSys(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): BasicLayer(
        dim=96, input_resolution=(64, 64), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=96, input_resolution=(64, 64), num_heads=3, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(8, 8), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=96, input_resolution=(64, 64), num_heads=3, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(8, 8), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.009)
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(64, 64), dim=96
          (reduction): Linear(in_features=384, out_features=192, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): BasicLayer(
        dim=192, input_resolution=(32, 32), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=192, input_resolution=(32, 32), num_heads=6, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(8, 8), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.018)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=192, input_resolution=(32, 32), num_heads=6, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(8, 8), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.027)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(32, 32), dim=192
          (reduction): Linear(in_features=768, out_features=384, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer(
        dim=384, input_resolution=(16, 16), depth=6
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.036)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.045)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (2): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.055)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (3): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.064)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (4): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.073)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (5): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.082)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(16, 16), dim=384
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer(
        dim=768, input_resolution=(8, 8), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=768, input_resolution=(8, 8), num_heads=24, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=768, window_size=(8, 8), num_heads=24
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.091)
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=768, input_resolution=(8, 8), num_heads=24, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=768, window_size=(8, 8), num_heads=24
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.100)
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (layers_up): ModuleList(
      (0): PatchExpand(
        (expand): Linear(in_features=768, out_features=1536, bias=False)
        (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
      (1): BasicLayer_up(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.036)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.045)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (2): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.055)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (3): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.064)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (4): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.073)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (5): SwinTransformerBlock(
            dim=384, input_resolution=(16, 16), num_heads=12, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(8, 8), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.082)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (upsample): PatchExpand(
          (expand): Linear(in_features=384, out_features=768, bias=False)
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer_up(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=192, input_resolution=(32, 32), num_heads=6, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(8, 8), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.018)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=192, input_resolution=(32, 32), num_heads=6, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(8, 8), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.027)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (upsample): PatchExpand(
          (expand): Linear(in_features=192, out_features=384, bias=False)
          (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer_up(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=96, input_resolution=(64, 64), num_heads=3, window_size=8, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(8, 8), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=96, input_resolution=(64, 64), num_heads=3, window_size=8, shift_size=4, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(8, 8), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.009)
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (concat_back_dim): ModuleList(
      (0): Identity()
      (1): Linear(in_features=768, out_features=384, bias=True)
      (2): Linear(in_features=384, out_features=192, bias=True)
      (3): Linear(in_features=192, out_features=96, bias=True)
    )
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (norm_up): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    (up): FinalPatchExpand_X4(
      (expand): Linear(in_features=96, out_features=1536, bias=False)
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (output): Conv2d(96, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (activation): Sigmoid()
  )
)
```

</details>

**Parameters**

Windows 1 epoch forward
```
Number of parameters: 41.39M
Input shape: torch.Size([2, 3, 256, 256])
Used time: 0.15s
Output shape: torch.Size([2, 1, 256, 256])
```

Linux 1000 epoches forward
```
Number of parameters: 41.39M
Input shape: torch.Size([2, 3, 256, 256])
Used time: 14.32s
Output shape: torch.Size([2, 1, 256, 256])
```

1 epoch: 14.32s / 1000 = 0.01432s

---

#### 1.2.2.3. VM-Unet

**Model**

<details>
<summary>View contents</summary>

```
VMUNet(
  (vmunet): VSSM(
    (patch_embed): PatchEmbed2D(
      (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.0)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.014285714365541935)
          )
        )
        (downsample): PatchMerging2D(
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (reduction): Linear(in_features=384, out_features=192, bias=False)
        )
      )
      (1): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.02857142873108387)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.04285714402794838)
          )
        )
        (downsample): PatchMerging2D(
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (reduction): Linear(in_features=768, out_features=384, bias=False)
        )
      )
      (2): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.05714285746216774)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.0714285746216774)
          )
          (2): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.08571428805589676)
          )
          (3): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.10000000149011612)
          )
          (4): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.11428571492433548)
          )
          (5): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.12857142090797424)
          )
          (6): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.1428571492433548)
          )
          (7): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.15714286267757416)
          )
          (8): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.17142857611179352)
          )
        )
        (downsample): PatchMerging2D(
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
        )
      )
      (3): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.18571428954601288)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.20000000298023224)
          )
        )
      )
    )
    (layers_up): ModuleList(
      (0): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.20000000298023224)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.18571428954601288)
          )
        )
      )
      (1): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.17142857611179352)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.15714286267757416)
          )
          (2): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.1428571492433548)
          )
          (3): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.12857142090797424)
          )
          (4): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.11428571492433548)
          )
          (5): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.10000000149011612)
          )
          (6): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.08571428805589676)
          )
          (7): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.0714285746216774)
          )
          (8): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.05714285746216774)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=768, out_features=1536, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.04285714402794838)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.02857142873108387)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=384, out_features=768, bias=False)
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.014285714365541935)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.0)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=192, out_features=384, bias=False)
          (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_up): Final_PatchExpand2D(
      (expand): Linear(in_features=96, out_features=384, bias=False)
      (norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
    )
    (final_conv): Conv2d(24, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

</details>

**Parameters**

Windows 1 epoch forward
```
Number of parameters: 44.27M
input shape: torch.Size([2, 3, 256, 256])
Forward time: 3.71 s
output shape: torch.Size([2, 1, 256, 256])
```

Linux 1000 epoches forward
```
Number of parameters: 44.27M
input shape: torch.Size([2, 3, 256, 256])
Forward time: 22.19 s
output shape: torch.Size([2, 1, 256, 256])
```

1 epoch: 22.19 / 1000 = 0.02219s

</details>
