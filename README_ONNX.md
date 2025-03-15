# Real-ESRGAN with ONNX Runtime on CPU

This extension to the Real-ESRGAN project allows you to run the models on CPU using ONNX Runtime, which can provide better performance compared to running PyTorch models directly on CPU.

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_onnx.txt
```

2. (Optional) If you want to use CUDA with ONNX Runtime, install the CUDA version:

```bash
pip install onnxruntime-gpu
```

## Converting PyTorch Models to ONNX

You can convert any of the Real-ESRGAN models to ONNX format using the provided script:

```bash
python convert_to_onnx.py --model_name realesr-animevideov3
```

Available model names:
- realesr-animevideov3 (default)
- RealESRGAN_x4plus_anime_6B
- RealESRGAN_x4plus
- RealESRNet_x4plus
- RealESRGAN_x2plus
- realesr-general-x4v3

By default, the script will:
1. Download the PyTorch model if it's not already in the `weights` directory
2. Convert it to ONNX format with dynamic input/output shapes
3. Save the ONNX model to the `weights` directory with the same name but `.onnx` extension

Additional options:
- `--model_path`: Specify a custom path to the PyTorch model
- `--output_path`: Specify a custom path to save the ONNX model
- `--static`: Use static input/output shapes instead of dynamic axes (not recommended for processing images/videos of different sizes)

## Running Video Inference with ONNX Runtime on CPU

You can use the provided script to run video inference with ONNX Runtime on CPU:

```bash
python inference_realesrgan_cpu_onnx.py -i input_video.mp4 -n realesr-animevideov3 -o results
```

The script will:
1. Convert the model to ONNX format if it doesn't exist yet
2. Run inference on the input video using ONNX Runtime on CPU
3. Save the upscaled video to the specified output directory

Additional options:
- `--onnx_model_path`: Specify a custom path to the ONNX model
- `-s` or `--outscale`: The final upsampling scale (default: 4)
- `-t` or `--tile`: Tile size for processing large images (0 for no tiling)
- `--tile_pad`: Padding size for each tile (default: 10)
- `--pre_pad`: Pre-padding size at each border (default: 0)
- `--fps`: FPS of the output video (default: same as input)
- `--ffmpeg_bin`: Path to the ffmpeg binary (default: 'ffmpeg')

## Performance Considerations

- For large videos, use the tiling option (`-t`) to avoid memory issues
- Adjust the tile size based on your available memory
- ONNX Runtime is generally faster than PyTorch on CPU, especially with optimizations

## Comparison with GPU Inference

While CPU inference with ONNX Runtime is more accessible for users without GPUs, it's still significantly slower than GPU inference. Here's a rough comparison:

- GPU (CUDA): Fastest option, recommended for processing large videos
- CPU (ONNX Runtime): Moderate speed, good for users without GPUs
- CPU (PyTorch): Slowest option, not recommended for video processing

## Troubleshooting

If you encounter any issues:

1. Make sure you have installed all the required dependencies
2. Check that the model was correctly converted to ONNX format
3. For memory issues, try reducing the tile size or using a smaller model
4. For compatibility issues, try using an older version of ONNX Runtime 