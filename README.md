# LTX Video Generation Node
## Community Node for InvokeAI

The **LTX Video Generation Node** for InvokeAI enables users to generate dynamic videos using the LTX-Video pipeline from Hugging Face Diffusers. This node supports both **text-to-video** and **image-to-video** generation

### ⚠️ Important Warning
This node requires `diffusers` version **0.32.2** to work correctly. to Install it using the following command:  
pip install diffusers==0.32.2  
Please note that this may cause other issues in invoke (so far I have not noticed any)

### Fields and Descriptions

| Field                   | Description                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| task_type               | Select the generation task type: `text-to-video` or `image-to-video`. Default is `text-to-video`. |
| prompt                  | Text prompt for the video.                                                                     |
| negative_prompt         | Negative prompt to avoid unwanted artifacts, such as "worst quality, inconsistent motion, blurry, jittery, distorted." |
| input_image             | Input image for the `image-to-video` task (ignored for `text-to-video`). Default is `None`.     |
| width                   | Width of the generated video in pixels (selectable from predefined values, e.g., 128 to 1280). Default is `640`. |
| height                  | Height of the generated video in pixels (selectable from predefined values, e.g., 128 to 1280). Default is `640`. |
| num_frames              | Number of frames in the video (selectable from predefined values, e.g., 9 to 257). Default is `105`. |
| fps                     | Frames per second for the generated video. Default is `24`.                                     |
| num_inference_steps     | Number of inference steps for video generation. Default is `30`.                                |
| guidance_scale          | Guidance scale for classifier-free diffusion. Higher values = stronger prompt adherence, lower values = better image quality. Default is `3.0`. |
| seed                    | Seed for reproducibility. Use `-1` for random behavior. Default is `42`.                        |
| max_length              | Maximum length of the input prompt in tokens (higher values may result in longer encoding times). Default is `256`. |
| output_path             | Path to save the generated video. Default is a `generated_videos` directory in the current file's path. |
| save_last_frame         | Option to save the last frame of the video as an uncompressed PNG file. Default is `False`.      |
| apply_compression       | Enable compression artifacts to simulate video-like input. Default is `False`.                  |
| compression_intensity   | Compression intensity level (higher = more compression artifacts, 0 = none). Default is `20`.    |
| upscale_frames          | Enable upscaling of video frames after generation. Default is `False`.                          |
| upscale_model           | Upscale model selection. Default is `RealESRGAN_x2plus.pth`.                                    |

