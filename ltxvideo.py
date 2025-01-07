# pip install git+https://github.com/huggingface/diffusers

import torch
import cv2
from diffusers import (
    LTXPipeline,
    LTXImageToVideoPipeline,
    LTXVideoTransformer3DModel,
    GGUFQuantizationConfig,
)
from invokeai.invocation_api import (
    BaseInvocation,
    InvocationContext,
    invocation,
    InputField,
    StringOutput,
    ImageField,
)
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Literal


@invocation(
    "ltx_video_generation",
    title="LTX Video Generation",
    tags=["video", "LTX", "generation"],
    category="video",
    version="0.2.0",
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
        default="low quality, blurry, distorted, watermark, artifacts, text artifacts, noisy, oversaturated, underexposed, overexposed, low resolution, jpeg artifacts, bad anatomy, malformed limbs, extra limbs, missing limbs, bad proportions, unnatural lighting, unrealistic details, unnatural skin texture, duplicate elements"
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
    description="seed for reproducibility. Set -1 for random behavior.", default=-1
    )
    output_path: str = InputField(
        description="Path to save the generated video",
        default=str(Path(__file__).parent / "generated_videos"),
    )

    def initialize_pipeline(self):
        """Initializes the correct pipeline with quantized models."""
        try:
            ckpt_path = "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q8_0.gguf"

            transformer = LTXVideoTransformer3DModel.from_single_file(
                ckpt_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                torch_dtype=torch.float16,
            )

            if self.task_type == "text-to-video":
                pipeline = LTXPipeline.from_pretrained(
                    "Lightricks/LTX-Video",
                    transformer=transformer,
                    torch_dtype=torch.float16,
                    device_map="balanced",
                )
            elif self.task_type == "image-to-video":
                pipeline = LTXImageToVideoPipeline.from_pretrained(
                    "Lightricks/LTX-Video",
                    transformer=transformer,
                    torch_dtype=torch.float16,
                    device_map="balanced",
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            pipeline.reset_device_map()
            pipeline.enable_model_cpu_offload()
            pipeline.enable_attention_slicing()

            return pipeline

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return None

    def load_image(self, context):
        """Loads and preprocesses the input image while preserving the aspect ratio."""
        try:
            if self.input_image:
                image = context.images.get_pil(self.input_image.image_name)

                if image is None:
                    raise ValueError("Image retrieval failed. Image is None.")

                original_width, original_height = image.size

                width = int(self.width)
                height = int(self.height)

                scale = min(width / original_width, height / original_height)

                new_width = (int(original_width * scale) + 31) // 32 * 32
                new_height = (int(original_height * scale) + 31) // 32 * 32

                image = image.resize((new_width, new_height))
                print(f"Resized image dimensions: {new_width}x{new_height}")
                return image

            raise ValueError("No input image provided.")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def generate_video(self, pipeline, image, prompt, negative_prompt):
        """Generates video frames using the specified pipeline and directly exports to video."""
        try:
            print(f"Task type: {self.task_type}")

            if self.task_type == "image-to-video" and image is not None:
                output_width, output_height = image.size
            else:
                output_width, output_height = int(self.width), int(self.height)

            generator = torch.manual_seed(self.seed) if self.seed > 0 else None

            if self.task_type == "text-to-video":
                print(f"Generating video with prompt: {prompt}")
                video_output = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=output_width,
                    height=output_height,
                    num_frames=int(self.num_frames),
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    max_sequence_length=250,
                )
            elif self.task_type == "image-to-video":
                print(f"Generating video with image and prompt: {prompt}")
                video_output = pipeline(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=output_width,
                    height=output_height,
                    num_frames=int(self.num_frames),
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    max_sequence_length=250,
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            if video_output.frames:
                video_frames = []

                all_frames = []
                for frame in video_output.frames:
                    if isinstance(frame, list):
                        all_frames.extend(frame)
                    else:
                        all_frames.append(frame)

                for frame in all_frames:
                    if isinstance(frame, (Image.Image, np.ndarray)):
                        frame_array = np.array(frame, dtype=np.uint8)
                        frame_array = np.clip(frame_array, 0, 255)
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                        video_frames.append(frame_array)
                    else:
                        print(f"Skipping frame of unsupported type: {type(frame)}")

                print(f"Total frames collected: {len(video_frames)}")

                Path(self.output_path).mkdir(parents=True, exist_ok=True)

                video_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                full_video_path = Path(self.output_path) / video_file_name

                if video_frames:
                    frame_height, frame_width = video_frames[0].shape[:2]

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        str(full_video_path),
                        fourcc,
                        self.fps,
                        (frame_width, frame_height), 
                    )


                    for frame in video_frames:
                        out.write(frame)

                    out.release()

                    return StringOutput(
                        value=f"Video successfully saved to: {full_video_path}"
                    )
                else:
                    print("No valid frames to export.")
                    return StringOutput(value="No valid frames generated.")

        except Exception as e:
            print(f"Error during video generation: {e}")
            return StringOutput(value=f"Error during video generation: {str(e)}")

    def invoke(self, context: InvocationContext) -> StringOutput:
        """Handles the invocation of video generation."""
        try:
            pipeline = self.initialize_pipeline()
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

            # Generate video
            return self.generate_video(
                pipeline, image, self.prompt, self.negative_prompt
            )

        except Exception as e:
            return StringOutput(value=f"Error: {str(e)}")
