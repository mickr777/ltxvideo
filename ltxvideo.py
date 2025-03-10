# pip install diffusers==0.32.2

from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    LTXImageToVideoPipeline,
    LTXPipeline,
)
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)
from invokeai.app.invocations.upscale import ESRGANInvocation
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    StringOutput,
    UIComponent,
    invocation,
)
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer


@invocation(
    "ltx_video_generation",
    title="LTX Video Generation",
    tags=["video", "LTX", "generation"],
    category="video",
    version="0.5.5",
    use_cache=False,
)
class LTXVideoInvocation(BaseInvocation):
    """Generates videos using LTX-Video pipeline from Hugging Face Diffusers."""

    task_type: Literal["text-to-video", "image-to-video"] = InputField(
        description="Select the generation task type", default="text-to-video"
    )
    prompt: str = InputField(
        description="Text prompt for the video", ui_component=UIComponent.Textarea
    )
    negative_prompt: str = InputField(
        description="Negative prompt to avoid unwanted artifacts",
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        ui_component=UIComponent.Textarea,
    )
    input_image: ImageField = InputField(
        description="Input image for image-to-video task", default=None
    )
    width: Literal[
        "128",
        "160",
        "192",
        "224",
        "256",
        "288",
        "320",
        "352",
        "384",
        "416",
        "448",
        "480",
        "512",
        "544",
        "576",
        "608",
        "640",
        "672",
        "704",
        "736",
        "768",
        "800",
        "832",
        "864",
        "896",
        "928",
        "960",
        "992",
        "1024",
        "1056",
        "1088",
        "1120",
        "1152",
        "1184",
        "1216",
        "1248",
        "1280",
    ] = InputField(description="Width of the generated video", default="704")
    height: Literal[
        "128",
        "160",
        "192",
        "224",
        "256",
        "288",
        "320",
        "352",
        "384",
        "416",
        "448",
        "480",
        "512",
        "544",
        "576",
        "608",
        "640",
        "672",
        "704",
        "736",
        "768",
        "800",
        "832",
        "864",
        "896",
        "928",
        "960",
        "992",
        "1024",
        "1056",
        "1088",
        "1120",
        "1152",
        "1184",
        "1216",
        "1248",
        "1280",
    ] = InputField(description="Height of the generated video", default="512")

    num_frames: Literal[
        "9",
        "17",
        "25",
        "33",
        "41",
        "49",
        "57",
        "65",
        "73",
        "81",
        "89",
        "97",
        "105",
        "113",
        "121",
        "129",
        "137",
        "145",
        "153",
        "161",
        "169",
        "177",
        "185",
        "193",
        "201",
        "209",
        "217",
        "225",
        "233",
        "241",
        "249",
        "257",
    ] = InputField(description="Number of frames in the video", default="161")

    fps: int = InputField(
        description="Frames per second for the generated video", default=24
    )
    num_inference_steps: int = InputField(
        description="Number of inference steps for video generation", default=50
    )
    guidance_scale: float = InputField(
        description="Guidance scale for classifier-free diffusion. Higher values = stronger prompt adherence, lower values = better image quality.",
        default=3.0,
    )
    seed: int = InputField(
        description="seed for reproducibility. Set -1 for random behavior.", default=42
    )
    max_length: Literal["128", "256", "512", "1024"] = InputField(
        description="Maximum length of the input prompt in tokens. (Higher values may result in longer encoding times)",
        default="256",
    )
    output_path: str = InputField(
        description="Path to save the generated video",
        default=str(Path(__file__).parent / "generated_videos"),
    )
    save_last_frame: bool = InputField(
        description="Save the last frame of the video as an uncompressed PNG file",
        default=False,
    )

    apply_compression: bool = InputField(
        description="Apply compression artifacts to simulate video-like input",
        default=False,
    )
    compression_intensity: int = InputField(
        description="Compression intensity (higher = more compression artifacts, 0 = none)",
        default=20,
    )

    upscale_frames: bool = InputField(
        description="Enable upscaling of video frames after generation", default=False
    )
    upscale_model: Literal[
        "RealESRGAN_x4plus.pth",
        "ealESRGAN_x4plus_anime_6B.pth",
        "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        "RealESRGAN_x2plus.pth",
    ] = InputField(
        description="Upscale Model Selection", default="RealESRGAN_x2plus.pth"
    )

    def initialize_pipeline(
        self, context: InvocationContext
    ) -> LTXPipeline | LTXImageToVideoPipeline:
        try:
            context.util.signal_progress("Loading transformer model...")

            ltx_model_path = context.models.download_and_cache_model(
                source="Lightricks/LTX-Video::/ltx-video-2b-v0.9.1.safetensors"
            )

            context.util.signal_progress("Loading text encoder...")
            text_encoder_path = context.models.download_and_cache_model(
                source="Lightricks/LTX-Video::/text_encoder"
            )
            text_encoder = T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path=str(text_encoder_path),
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            )
            text_encoder.config.max_length = 1024
            text_encoder.model_max_length = 1024

            context.util.signal_progress("Loading tokenizer...")
            tokenizer_path = context.models.download_and_cache_model(
                source="Lightricks/LTX-Video::/tokenizer"
            )
            tokenizer = T5Tokenizer.from_pretrained(
                pretrained_model_name_or_path=str(tokenizer_path),
            )
            tokenizer.model_max_length = 1024
            tokenizer.max_length = 1024

            vae = AutoencoderKLLTXVideo.from_single_file(
                ltx_model_path,
                torch_dtype=torch.bfloat16,
            )
            vae.enable_tiling()

            if self.task_type == "text-to-video":
                pipeline = LTXPipeline.from_single_file(
                    ltx_model_path,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    torch_dtype=torch.bfloat16,
                )
            elif self.task_type == "image-to-video":
                pipeline = LTXImageToVideoPipeline.from_single_file(
                    ltx_model_path,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    torch_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            context.util.signal_progress("Optimizing pipeline...")
            pipeline.enable_model_cpu_offload()
            context.util.signal_progress("Pipeline optimization complete.")

            return pipeline

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise

    def load_image(self, context):
        try:
            if not self.input_image:
                raise ValueError("No input image provided.")

            image = context.images.get_pil(self.input_image.image_name)
            if image is None:
                raise ValueError("Image retrieval failed. Image is None.")

            if image.mode != "RGB":
                image = image.convert("RGB")

            original_width, original_height = image.size
            scale = min(
                int(self.width) / original_width, int(self.height) / original_height
            )
            new_width = (int(original_width * scale) // 32) * 32
            new_height = (int(original_height * scale) // 32) * 32
            image = image.resize((new_width, new_height), Image.LANCZOS)

            if self.apply_compression:
                image = self.add_compression_artifacts(
                    image, self.compression_intensity
                )

            return image

        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def add_compression_artifacts(self, image, intensity):
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 - intensity]
            _, encoded_image = cv2.imencode(".jpg", image_cv, encode_param)
            decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

            compressed_image = Image.fromarray(
                cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
            )
            return compressed_image

        except Exception as e:
            print(f"Error adding compression artifacts: {e}")
        return image

    def upscale_video_frames(
        self,
        video_frames,
        esrgan_invocation: ESRGANInvocation,
        model_name: str,
        tile_size: int,
        context: InvocationContext,
    ):
        upscaled_frames = []
        for frame in video_frames:
            try:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                esrgan_invocation.image = ImageField(image_name="temp_frame")
                esrgan_invocation.model_name = model_name
                esrgan_invocation.tile_size = tile_size

                temp_image_dto = context.images.save(image=pil_frame)

                esrgan_invocation.image.image_name = temp_image_dto.image_name

                upscaled_output = esrgan_invocation.invoke(context)

                upscaled_image = context.images.get_pil(
                    upscaled_output.image.image_name
                )

                upscaled_frames.append(
                    cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)
                )

            except Exception as e:
                print(f"Error during frame upscaling: {e}")

        return upscaled_frames

    def generate_video(
        self,
        pipeline: LTXPipeline | LTXImageToVideoPipeline,
        image,
        prompt,
        negative_prompt,
        context: InvocationContext,
    ) -> StringOutput:
        """
        Generates video frames using the LTX pipeline and optionally upscales them.
        """
        try:
            print(f"Task type: {self.task_type}")

            if self.task_type == "image-to-video" and image is not None:
                output_width, output_height = image.size
            else:
                output_width, output_height = int(self.width), int(self.height)

            if self.seed > 0:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(self.seed)
            else:
                generator = None

            context.util.signal_progress(f"Generating Video With Prompt: {prompt}")

            def callback_on_step_end(
                pipeline: LTXPipeline | LTXImageToVideoPipeline,
                step: int,
                timestep: int,
                callback_kwargs: dict,
            ):
                progress = min((step + 1) / self.num_inference_steps, 1.0)
                context.util.signal_progress(
                    f"Step {step + 1}/{self.num_inference_steps}", progress
                )
                return callback_kwargs

            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": output_width,
                "height": output_height,
                "num_frames": int(self.num_frames),
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "generator": generator,
                "max_sequence_length": int(self.max_length),
                "callback_on_step_end": callback_on_step_end,
            }

            if self.task_type == "image-to-video":
                pipeline_kwargs["image"] = image

            video_output = pipeline(**pipeline_kwargs)

            if video_output.frames:
                video_frames = [
                    cv2.cvtColor(
                        np.clip(np.array(frame, dtype=np.uint8), 0, 255),
                        cv2.COLOR_RGB2BGR,
                    )
                    for sublist in video_output.frames
                    for frame in (sublist if isinstance(sublist, list) else [sublist])
                    if isinstance(frame, (Image.Image, np.ndarray))
                ]

                print(f"Total frames collected: {len(video_frames)}")

                Path(self.output_path).mkdir(parents=True, exist_ok=True)
                original_video_file_name = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_original.mp4"
                )
                original_video_path = Path(self.output_path) / original_video_file_name

                frame_height, frame_width = video_frames[0].shape[:2]
                out = cv2.VideoWriter(
                    str(original_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (frame_width, frame_height),
                )

                for frame in video_frames:
                    out.write(frame)
                out.release()
                print(f"Original video successfully saved to: {original_video_path}")

                if self.save_last_frame:
                    last_video_frame_path = (
                        Path(self.output_path)
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_original.png"
                    )
                    cv2.imwrite(str(last_video_frame_path), video_frames[-1])
                    print(f"Last upscaled frame saved to: {last_video_frame_path}")

            if self.upscale_frames:
                model_name = self.upscale_model

                if model_name in [
                    "RealESRGAN_x4plus.pth",
                    "ealESRGAN_x4plus_anime_6B.pth",
                    "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
                ]:
                    scale_factor = 4
                elif model_name == "RealESRGAN_x2plus.pth":
                    scale_factor = 2
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

                esrgan_invocation = ESRGANInvocation()

                upscaled_frames = self.upscale_video_frames(
                    video_frames,
                    esrgan_invocation,
                    model_name=model_name,
                    tile_size=400,
                    context=context,
                )

                if upscaled_frames:
                    upscale_video_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_upscaled.mp4"
                    upscale_video_path = (
                        Path(self.output_path) / upscale_video_file_name
                    )

                    upscale_height, upscale_width = (
                        frame_height * scale_factor,
                        frame_width * scale_factor,
                    )
                    upscale_out = cv2.VideoWriter(
                        str(upscale_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (upscale_width, upscale_height),
                    )
                    for frame in upscaled_frames:
                        upscale_out.write(frame)
                    upscale_out.release()
                    print(f"Upscaled video successfully saved to: {upscale_video_path}")

                if self.save_last_frame:
                    last_upscaled_frame_path = (
                        Path(self.output_path)
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_upscaled.png"
                    )
                    cv2.imwrite(str(last_upscaled_frame_path), video_frames[-1])
                    print(f"Last upscaled frame saved to: {last_upscaled_frame_path}")

                return StringOutput(
                    value=f"Original video saved to: {original_video_path}"
                )

            return StringOutput(value="No valid frames generated.")

        except Exception as e:
            print(f"Error during video generation: {e}")
            return StringOutput(value=f"Error during video generation: {str(e)}")

    def invoke(self, context: InvocationContext) -> StringOutput:
        try:
            context.util.signal_progress("Initializing the pipeline...")
            pipeline = self.initialize_pipeline(context)
            if not pipeline:
                return StringOutput(value="Pipeline initialization failed.")

            if self.task_type == "image-to-video" and not self.input_image:
                return StringOutput(
                    value="Input image is required for image-to-video task."
                )

            image = (
                self.load_image(context) if self.task_type == "image-to-video" else None
            )
            if not image and self.task_type == "image-to-video":
                return StringOutput(value="Failed to load input image.")

            return self.generate_video(
                pipeline, image, self.prompt, self.negative_prompt, context
            )

        except Exception as e:
            return StringOutput(value=f"Error: {str(e)}")
