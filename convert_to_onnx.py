import argparse
import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def convert_to_onnx(model_name, model_path, onnx_path, dynamic=True):
    """Convert PyTorch model to ONNX format.
    
    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the PyTorch model.
        onnx_path (str): Path to save the ONNX model.
        dynamic (bool): Whether to use dynamic axes for input and output.
    """
    print(f'Converting {model_name} to ONNX format...')
    
    # Load model
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    
    # Load model weights
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))
    
    # prefer to use params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Export to ONNX
    if dynamic:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test ONNX model with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Compare PyTorch and ONNX Runtime outputs
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Check if the outputs are close
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-5)
    
    print(f'ONNX model saved to {onnx_path}')
    print(f'PyTorch and ONNX Runtime outputs match with {netscale}x upscaling')
    
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Convert Real-ESRGAN models to ONNX format')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-animevideov3',
        help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
              ' RealESRGAN_x2plus | realesr-general-x4v3'
              'Default:realesr-animevideov3'))
    parser.add_argument(
        '-p',
        '--model_path',
        type=str,
        default=None,
        help='Path to the PyTorch model. If not provided, it will be downloaded automatically.')
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        default=None,
        help='Path to save the ONNX model. If not provided, it will be saved in the weights directory.')
    parser.add_argument(
        '--static',
        action='store_true',
        help='Use static input and output shapes instead of dynamic axes.')
    
    args = parser.parse_args()
    
    # Get model path
    if args.model_path is None:
        model_name = args.model_name.split('.pth')[0]
        model_path = os.path.join('weights', f'{model_name}.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            if model_name == 'RealESRGAN_x4plus':
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            elif model_name == 'RealESRNet_x4plus':
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
            elif model_name == 'RealESRGAN_x4plus_anime_6B':
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
            elif model_name == 'RealESRGAN_x2plus':
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            elif model_name == 'realesr-animevideov3':
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
            elif model_name == 'realesr-general-x4v3':
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    else:
        model_name = args.model_name
    
    # Get output path
    if args.output_path is None:
        os.makedirs('weights', exist_ok=True)
        output_path = os.path.join('weights', f'{model_name.split(".pth")[0]}.onnx')
    else:
        output_path = args.output_path
    
    # Convert to ONNX
    convert_to_onnx(model_name, model_path, output_path, dynamic=not args.static)


if __name__ == '__main__':
    main() 