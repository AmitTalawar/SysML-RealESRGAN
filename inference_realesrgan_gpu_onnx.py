import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
import onnx
import onnxruntime as ort
import math
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg


def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret


class RealESRGANerONNX:
    """A helper class for upsampling images with RealESRGAN using ONNX Runtime with GPU acceleration.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the ONNX model.
        tile (int): As too large images result in memory issues, this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        gpu_id (int): GPU device ID to use. Default: 0.
    """

    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, gpu_id=0):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.gpu_id = gpu_id

        # Setup providers for ONNX Runtime
        # Always try to use CUDA first, then fall back to CPU if not available
        providers = []

        # Check if CUDA is available in ONNX Runtime
        cuda_available = 'CUDAExecutionProvider' in ort.get_available_providers()
        if cuda_available:
            providers.append(('CUDAExecutionProvider', {'device_id': self.gpu_id}))
        providers.append('CPUExecutionProvider')

        # Print available providers for debugging
        print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
        print(f"Using providers: {providers}")

        # Load ONNX model
        try:
            self.ort_session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name

            # Check if GPU is being used
            session_providers = self.ort_session.get_providers()
            print(f"Active providers: {session_providers}")
            if 'CUDAExecutionProvider' in session_providers:
                print(f"Using GPU acceleration with CUDA (device ID: {gpu_id})")
            else:
                print("WARNING: CUDA is not available. Falling back to CPU execution.")
                if cuda_available:
                    print("CUDA provider was available but not used. This might be due to CUDA compatibility issues.")
                    print("Try installing the correct CUDA version for your GPU.")
        except Exception as e:
            print(f"Error initializing ONNX session: {e}")
            print("Trying to load with CPU provider only...")
            self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print("Successfully loaded model with CPU provider.")

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        # Debug input image
        print(f"Pre-process input shape: {img.shape}, dtype: {img.dtype}, min: {np.min(img)}, max: {np.max(img)}")

        # Check for NaN or invalid values
        if np.isnan(img).any():
            print("WARNING: Input image contains NaN values!")
            img = np.nan_to_num(img)

        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        self.img = img[np.newaxis, ...]

        # pre_pad
        if self.pre_pad != 0:
            self.img = np.pad(self.img, ((0, 0), (0, 0), (0, self.pre_pad), (0, self.pre_pad)), mode='reflect')

        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        else:
            self.mod_scale = self.scale  # Default to scale value

        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.shape
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = np.pad(self.img, ((0, 0), (0, 0), (0, self.mod_pad_h), (0, self.mod_pad_w)), mode='reflect')

        # Debug processed input
        print(f"Processed input shape: {self.img.shape}, min: {np.min(self.img)}, max: {np.max(self.img)}")

    def process(self):
        # model inference
        try:
            # Debug input to model
            print(f"Model input shape: {self.img.shape}, min: {np.min(self.img)}, max: {np.max(self.img)}")

            self.output = self.ort_session.run([self.output_name], {self.input_name: self.img})[0]

            # Debug model output
            print(f"Model output shape: {self.output.shape}, min: {np.min(self.output)}, max: {np.max(self.output)}")

            # Check for NaN or zero values in output
            if np.isnan(self.output).any():
                print("WARNING: Model output contains NaN values!")
                self.output = np.nan_to_num(self.output)

            if np.max(self.output) == 0:
                print("WARNING: Model output is all zeros!")
        except Exception as e:
            print(f"Error during model inference: {e}")
            # Create a dummy output as fallback
            _, _, h, w = self.img.shape
            self.output = np.zeros((1, 3, h * self.scale, w * self.scale), dtype=np.float32)
            print("Created fallback output due to inference error.")

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = np.zeros(output_shape, dtype=np.float32)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    output_tile = self.ort_session.run([self.output_name], {self.input_name: input_tile})[0]
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.shape
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.shape
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = np.clip(output_img[0], 0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = np.clip(output_alpha[0], 0, 1)
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = args.input
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        print(f"Creating output video with dimensions: {out_width}x{out_height}, FPS: {fps}")

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        # Ensure frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Empty frame detected, skipping")
            return

        # Ensure frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            print(f"Warning: Converting frame from {frame.dtype} to uint8")
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Check if frame has any content
        if np.max(frame) == 0:
            print("Warning: Black frame detected")

        try:
            frame_bytes = frame.tobytes()
            self.stream_writer.stdin.write(frame_bytes)
        except Exception as e:
            print(f"Error writing frame: {e}")

    def close(self):
        try:
            self.stream_writer.stdin.close()
            self.stream_writer.wait()
            print("Video writer closed successfully")
        except Exception as e:
            print(f"Error closing video writer: {e}")


def convert_to_onnx(model_name, model_path, onnx_path):
    """Convert PyTorch model to ONNX format."""
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

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f'ONNX model saved to {onnx_path}')
    return onnx_path


def inference_video(args, video_save_path):
    # Convert model to ONNX if needed
    if not os.path.exists(args.onnx_model_path):
        # Get original model path
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

        # Convert to ONNX
        convert_to_onnx(model_name, model_path, args.onnx_model_path)

    # Initialize ONNX upsampler
    if args.model_name == 'RealESRGAN_x4plus':
        scale = 4
    elif args.model_name == 'RealESRNet_x4plus':
        scale = 4
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        scale = 4
    elif args.model_name == 'RealESRGAN_x2plus':
        scale = 2
    elif args.model_name == 'realesr-animevideov3':
        scale = 4
    elif args.model_name == 'realesr-general-x4v3':
        scale = 4
    else:
        scale = 4

    upsampler = RealESRGANerONNX(
        scale=scale,
        model_path=args.onnx_model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        gpu_id=args.gpu_id
    )

    reader = Reader(args)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # Create debug directory if needed
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(args.output, 'debug_frames')
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug frames will be saved to: {debug_dir}")

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    frame_count = 0

    # Process first few frames with extra debugging
    verbose_debug = args.verbose_debug

    while True:
        img = reader.get_frame()
        if img is None:
            break

        # Extra debugging for first few frames
        if verbose_debug and frame_count < 3:
            print(f"\n==== DETAILED DEBUG FOR FRAME {frame_count} ====")

        # Debug: Print input frame info
        if args.debug and frame_count == 0:
            print(f"Input frame shape: {img.shape}, dtype: {img.dtype}, min: {np.min(img)}, max: {np.max(img)}")
            input_debug_path = os.path.join(debug_dir, f"input_frame_{frame_count}.png")
            cv2.imwrite(input_debug_path, img)
            print(f"Saved input debug frame to: {input_debug_path}")

            # Save a simple color test pattern to verify file writing works
            test_pattern = np.zeros((height, width, 3), dtype=np.uint8)
            # Red, Green, Blue squares
            h_third, w_third = height // 3, width // 3
            test_pattern[0:h_third, 0:w_third, 2] = 255  # Red
            test_pattern[h_third:2*h_third, w_third:2*w_third, 1] = 255  # Green
            test_pattern[2*h_third:, 2*w_third:, 0] = 255  # Blue
            test_pattern_path = os.path.join(debug_dir, "test_pattern.png")
            cv2.imwrite(test_pattern_path, test_pattern)
            print(f"Saved test pattern to: {test_pattern_path}")

        try:
            if verbose_debug and frame_count < 3:
                print(f"Processing frame {frame_count} with enhanced debugging")

            output, img_mode = upsampler.enhance(img, outscale=args.outscale)

            # Debug: Print output frame info
            if args.debug:
                print(f"Output frame shape: {output.shape}, dtype: {output.dtype}, min: {np.min(output)}, max: {np.max(output)}")

                # Save output frame
                if frame_count % 10 == 0 or (verbose_debug and frame_count < 3):
                    output_debug_path = os.path.join(debug_dir, f"output_frame_{frame_count}.png")
                    # Make a copy to avoid modifying the original
                    output_copy = output.copy()

                    # If output is all black, add a visible marker
                    if np.max(output_copy) == 0:
                        h, w = output_copy.shape[:2]
                        cv2.putText(output_copy, "BLACK FRAME", (w//4, h//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    cv2.imwrite(output_debug_path, output_copy)
                    print(f"Saved output debug frame to: {output_debug_path}")

                    # Also save a resized version of the original for comparison
                    resized_img = cv2.resize(img, (output.shape[1], output.shape[0]),
                                           interpolation=cv2.INTER_LANCZOS4)
                    resized_path = os.path.join(debug_dir, f"resized_original_{frame_count}.png")
                    cv2.imwrite(resized_path, resized_img)
                    print(f"Saved resized original to: {resized_path}")

            # Ensure output is valid
            if np.isnan(output).any() or np.max(output) == 0:
                print(f"Warning: Frame {frame_count} has invalid values. Using original frame instead.")
                # Resize original frame as fallback
                output = cv2.resize(img, (int(width * args.outscale), int(height * args.outscale)),
                                   interpolation=cv2.INTER_LANCZOS4)

            writer.write_frame(output)
        except RuntimeError as error:
            print(f'Error processing frame {frame_count}:', error)
            print('If you encounter memory issues, try to set --tile with a smaller number.')
            # Resize original frame as fallback
            output = cv2.resize(img, (int(width * args.outscale), int(height * args.outscale)),
                               interpolation=cv2.INTER_LANCZOS4)
            writer.write_frame(output)
        except Exception as e:
            print(f'Unexpected error processing frame {frame_count}:', e)
            # Resize original frame as fallback
            output = cv2.resize(img, (int(width * args.outscale), int(height * args.outscale)),
                               interpolation=cv2.INTER_LANCZOS4)
            writer.write_frame(output)

        pbar.update(1)
        frame_count += 1

        # End verbose debugging after a few frames
        if verbose_debug and frame_count >= 3:
            verbose_debug = False
            print("Ending verbose debugging after 3 frames")

    reader.close()
    writer.close()
    print(f"Video processing complete. Processed {frame_count} frames.")
    print(f"Output saved to: {video_save_path}")


def main():
    """Inference demo for Real-ESRGAN with GPU and ONNX Runtime.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-animevideov3',
        help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
              ' RealESRGAN_x2plus | realesr-general-x4v3'
              'Default:realesr-animevideov3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '--onnx_model_path',
        type=str,
        default=None,
        help='Path to the ONNX model. If not provided, it will be generated from the PyTorch model.')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save intermediate frames')
    parser.add_argument('--verbose_debug', action='store_true', help='Enable verbose debugging for the first few frames')
    args = parser.parse_args()

    # Set default ONNX model path if not provided
    if args.onnx_model_path is None:
        args.onnx_model_path = os.path.join('weights', f'{args.model_name.split(".pth")[0]}.onnx')

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    inference_video(args, video_save_path)


if __name__ == '__main__':
    main()