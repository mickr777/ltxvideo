# pip install git+https://github.com/huggingface/diffusers

from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXConditionPipeline,
    LTXVideoCondition,
)
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


@invocation(
    "ltx_video_generation",
    title="LTX Video Generation",
    tags=["video", "LTX", "generation"],
    category="video",
    version="0.9.5",
    use_cache=False,
)
class LTXVideoInvocation(BaseInvocation):
    """Generates videos using LTX-Video v0.9.5 pipeline with condition support."""

    task_type: Literal["text-to-video", "image-to-video"] = InputField(
        description="Select the generation task type",
        default="text-to-video"
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

    fps: int = InputField(description="Frames per second for the video", default=24)
    num_inference_steps: int = InputField(
        description="Number of inference steps", default=40
    )
    guidance_scale: float = InputField(
        description="Guidance scale for classifier-free diffusion", default=3.0
    )
    seed: int = InputField(
        description="Seed for reproducibility. Set -1 for random.", default=42
    )
    output_path: str = InputField(
        description="Path to save generated video",
        default=str(Path(__file__).parent / "generated_videos"),
    )
    save_last_frame: bool = InputField(
        description="Save the last frame of the video as PNG", default=False
    )
    apply_compression: bool = InputField(
        description="Apply compression artifacts to input image", default=False
    )
    compression_intensity: int = InputField(
        description="Compression intensity (0-100)", default=20
    )

    def initialize_pipeline(self, context: InvocationContext) -> LTXConditionPipeline:
        try:
            context.util.signal_progress("Loading LTX-Video v0.9.5 pipeline...")
            pipeline = LTXConditionPipeline.from_pretrained(
                "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
            )
            pipeline.vae.enable_tiling()
            pipeline.enable_sequential_cpu_offload()
            return pipeline
        except Exception as e:
            print(f"Pipeline initialization error: {e}")
            raise
        
    def load_image(self, context):
        try:
            if not self.input_image:
                raise ValueError("No input image provided.")

            image = context.images.get_pil(self.input_image.image_name)
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
            return Image.fromarray(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Compression error: {e}")
            return image

    def generate_video(
        self,
        pipeline: LTXConditionPipeline,
        image,
        prompt,
        negative_prompt,
        context: InvocationContext,
    ) -> StringOutput:
        try:
            output_width, output_height = int(self.width), int(self.height)

            if self.task_type == "image-to-video" and image is not None:
                output_width, output_height = image.size

            def callback_on_step_end(pipeline, step: int, timestep: int, callback_kwargs: dict):
                progress = min((step + 1) / self.num_inference_steps, 1.0)
                context.util.signal_progress(f"Step {step + 1}/{self.num_inference_steps}", progress)
                return callback_kwargs

            generator = torch.Generator(device="cuda").manual_seed(self.seed) if self.seed > 0 else None

            conditions = []
            if self.task_type == "image-to-video" and image:
                conditions.append(LTXVideoCondition(image=image, frame_index=0))

            video_output = pipeline(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=output_width,
                height=output_height,
                num_frames=int(self.num_frames),
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )

            if not video_output.frames:
                return StringOutput(value="No video frames generated.")

            video_frames = [
                cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                for sublist in video_output.frames
                for frame in (sublist if isinstance(sublist, list) else [sublist])
                if isinstance(frame, (Image.Image, np.ndarray))
            ]

            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            video_file = (
                Path(self.output_path)
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_video.mp4"
            )
            frame_height, frame_width = video_frames[0].shape[:2]
            out = cv2.VideoWriter(
                str(video_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame_width, frame_height),
            )
            for frame in video_frames:
                out.write(frame)
            out.release()

            if self.save_last_frame:
                last_frame_file = (
                    Path(self.output_path)
                    / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_last_frame.png"
                )
                cv2.imwrite(str(last_frame_file), video_frames[-1])

            return StringOutput(value=f"Video saved to: {video_file}")

        except Exception as e:
            print(f"Video generation error: {e}")
            return StringOutput(value=f"Error: {str(e)}")

    def invoke(self, context: InvocationContext) -> StringOutput:
        try:
            pipeline = self.initialize_pipeline(context)
            image = (
                self.load_image(context) if self.task_type == "image-to-video" else None
            )
            if self.task_type == "image-to-video" and not image:
                return StringOutput(value="Failed to load input image.")

            return self.generate_video(
                pipeline, image, self.prompt, self.negative_prompt, context
            )

        except Exception as e:
            return StringOutput(value=f"Error: {str(e)}")
