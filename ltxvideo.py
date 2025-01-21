# pip install diffusers==0.32.2

from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    GGUFQuantizationConfig,
    LTXImageToVideoPipeline,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FlowMatchEulerDiscreteScheduler
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    StringOutput,
    invocation,
)
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer


@invocation(
    "ltx_video_generation",
    title="LTX Video Generation",
    tags=["video", "LTX", "generation"],
    category="video",
    version="0.4.0",
    use_cache=False,
)
class LTXVideoInvocation(BaseInvocation):
    """Generates videos using LTX-Video pipeline from Hugging Face Diffusers."""

    task_type: Literal["text-to-video", "image-to-video"] = InputField(
        description="Select the generation task type", default="text-to-video"
    )
    prompt: str = InputField(description="Text prompt for the video")
    negative_prompt: str = InputField(
        description="Negative prompt to avoid unwanted artifacts", 
        default="low quality, blurry, distorted, watermark, artifacts"
    )
    input_image: ImageField = InputField(
        description="Input image for image-to-video task", default=None
    )
    width: Literal[
        "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", 
        "576", "608", "640", "672", "704", "736", "768", "800", "832", "864", "896", "928", "960", "992", 
        "1024", "1056", "1088", "1120", "1152", "1184", "1216", "1248", "1280"
    ] = InputField(description="Width of the generated video", default="640")
    height: Literal[
        "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544",
        "576", "608", "640", "672", "704", "736", "768", "800", "832", "864", "896", "928", "960", "992",
          "1024", "1056", "1088", "1120", "1152", "1184", "1216", "1248", "1280"
    ] = InputField(description="Height of the generated video", default="640")
    
    num_frames: Literal[
        "9", "17", "25", "33", "41", "49", "57", "65", "73", "81", "89", "97", "105", "113", "121", "129",
        "137", "145", "153", "161", "169", "177", "185", "193", "201", "209", "217", "225", "233", "241", "249", "257",
    ] = InputField(description="Number of frames in the video", default="105")
    fps: int = InputField(
        description="Frames per second for the generated video", default=24
    )
    num_inference_steps: int = InputField(
        description="Number of inference steps for video generation", default=30
    )
    guidance_scale: float = InputField(
        description="Guidance scale for classifier-free diffusion. Higher values = stronger prompt adherence, lower values = better image quality.",
        default=3.0,
    )
    seed: int = InputField(
    description="seed for reproducibility. Set -1 for random behavior.", default=42
    )
    shift: float = InputField(
        description="Timestep shift parameter to control noise schedule dynamics", 
        default=8.0
    )
    max_length: Literal["128", "256", "512", "1024"] = InputField(
        description="Maximum length of the input prompt in tokens. (Higher values may result in longer encoding times)", default="256"
    )
    output_path: str = InputField(
        description="Path to save the generated video",
        default=str(Path(__file__).parent / "generated_videos"),
    )
    save_last_frame: bool = InputField(
    description="Save the last frame of the video as an uncompressed PNG file",
    default=False
    )

    def initialize_pipeline(self, context: InvocationContext) -> LTXPipeline | LTXImageToVideoPipeline:
        """Initializes the correct pipeline with quantized models and manual progress updates."""
        try:
            context.util.signal_progress("Loading transformer model...")
            ckpt_path = "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q8_0.gguf"
            transformer = LTXVideoTransformer3DModel.from_single_file(
                ckpt_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                torch_dtype=torch.float16,
            )

            double_quant_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            context.util.signal_progress("Loading text encoder...")
            text_encoder = T5EncoderModel.from_pretrained(
                "Lightricks/LTX-Video",
                subfolder="text_encoder",
                quantization_config=double_quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            text_encoder.config.max_length = 1024
            text_encoder.model_max_length = 1024

            context.util.signal_progress("Loading tokenizer...")
            tokenizer = T5Tokenizer.from_pretrained(
                "Lightricks/LTX-Video",
                subfolder="tokenizer",
            )
            tokenizer.model_max_length = 1024
            tokenizer.max_length = 1024

            context.util.signal_progress("Loading VAE...")
            vae = AutoencoderKLLTXVideo.from_pretrained(
                "Lightricks/LTX-Video",
                subfolder="vae",
                torch_dtype=torch.float16,
            )
            vae.enable_tiling()
            
            if self.task_type == "image-to-video":
                scheduler = FlowMatchEulerDiscreteScheduler(
                    base_shift=0.6,
                    max_shift=2.5,
                    shift_terminal=0.15,
                    use_dynamic_shifting=True,
                )
            else:
                scheduler = FlowMatchEulerDiscreteScheduler(
                    shift=float(self.shift),
                )

            context.util.signal_progress("Initializing pipeline...")
            if self.task_type == "text-to-video":
                pipeline = LTXPipeline(
                    transformer=transformer,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    scheduler=scheduler,
                )
            elif self.task_type == "image-to-video":
                pipeline = LTXImageToVideoPipeline(
                    transformer=transformer,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    scheduler=scheduler,
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            context.util.signal_progress("Optimizing pipeline...")
            pipeline.enable_model_cpu_offload()
            context.util.signal_progress("Pipeline optimization complete.")

            return pipeline

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return None


    def load_image(self, context):
        """Loads and preprocesses the input image while preserving the aspect ratio."""
        try:
            if not self.input_image:
                raise ValueError("No input image provided.")

            image = context.images.get_pil(self.input_image.image_name)
            if image is None:
                raise ValueError("Image retrieval failed. Image is None.")

            if image.mode != "RGB":
                image = image.convert("RGB")
                
            original_width, original_height = image.size
            scale = min(int(self.width) / original_width, int(self.height) / original_height)
            new_width = (int(original_width * scale) // 32) * 32
            new_height = (int(original_height * scale) // 32) * 32
            image = image.resize((new_width, new_height), Image.LANCZOS)

            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def generate_video(
        self,
        pipeline: LTXPipeline | LTXImageToVideoPipeline,
        image,
        prompt,
        negative_prompt,
        context: InvocationContext,
    ) -> StringOutput:
        """Generates video frames using the specified pipeline and directly exports to video."""
        try:
            print(f"Task type: {self.task_type}")

            if self.task_type == "image-to-video" and image is not None:
                output_width, output_height = image.size
            else:
                output_width, output_height = int(self.width), int(self.height)

            generator = torch.manual_seed(self.seed) if self.seed > 0 else None

            context.util.signal_progress(f"Generating Video With Prompt: {prompt}")

            def callback_on_step_end(
                pipeline: LTXPipeline | LTXImageToVideoPipeline,
                step: int,
                timestep: int,
                callback_kwargs: dict,
            ):
                progress = min((step + 1) / self.num_inference_steps, 1.0)
                context.util.signal_progress(f"Step {step + 1}/{self.num_inference_steps}", progress)
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
                        np.clip(np.array(frame, dtype=np.uint8), 0, 255), cv2.COLOR_RGB2BGR
                    )
                    for sublist in video_output.frames
                    for frame in (sublist if isinstance(sublist, list) else [sublist])
                    if isinstance(frame, (Image.Image, np.ndarray))
                ]

                print(f"Total frames collected: {len(video_frames)}")

                Path(self.output_path).mkdir(parents=True, exist_ok=True)

                video_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                full_video_path = Path(self.output_path) / video_file_name

                frame_height, frame_width = video_frames[0].shape[:2]
                out = cv2.VideoWriter(
                    str(full_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (frame_width, frame_height),
                )

                for frame in video_frames:
                    out.write(frame)

                out.release()
                print(f"Video successfully saved to: {full_video_path}")

                if self.save_last_frame:
                    try:
                        last_frame = video_frames[-1]
                        last_frame_path = full_video_path.with_suffix(".png")
                        cv2.imwrite(str(last_frame_path), last_frame)
                        print(f"Last frame successfully saved as PNG to: {last_frame_path}")
                        return StringOutput(
                            value=(
                                f"Video successfully saved to: {full_video_path}\n"
                                f"Last frame saved as PNG to: {last_frame_path}"
                            )
                        )
                    except Exception as e:
                        print(f"Error saving the last frame as PNG: {e}")
                        
                return StringOutput(value=f"Video successfully saved to: {full_video_path}")

            return StringOutput(value="No valid frames generated.")

        except Exception as e:
            print(f"Error during video generation: {e}")
            return StringOutput(value=f"Error during video generation: {str(e)}")


    def invoke(self, context: InvocationContext) -> StringOutput:
        """Handles the invocation of video generation."""
        try:
            context.util.signal_progress("Initializing the pipeline...")
            pipeline = self.initialize_pipeline(context)
            if not pipeline:
                return StringOutput(value="Pipeline initialization failed.")

            if self.task_type == "image-to-video" and not self.input_image:
                return StringOutput(value="Input image is required for image-to-video task.")

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
