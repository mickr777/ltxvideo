# LTX Video Generation Node
## Community Node for InvokeAI

The **LTX Video Generation Node** for InvokeAI enables users to generate dynamic videos using the LTX-Video pipeline from Hugging Face Diffusers. This node supports both **text-to-video** and **image-to-video** generation

### ⚠️ Important Warning
This node requires `diffusers` version **0.32.1** to work correctly. to Install it using the following command:
pip install diffusers==0.32.1
Please note that this may cuase other issues in invoke (so far I have not noticed any)

### Fields and Descriptions

| Field                  | Description                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| task_type              | Specifies the generation task type: `text-to-video` or `image-to-video`.                         |
| prompt                 | Text prompt guiding the video generation.                                                       |
| negative_prompt        | Negative prompt to avoid unwanted artifacts in the video (e.g., blurry, noisy, low quality).     |
| input_image            | Input image for the `image-to-video` task (ignored for `text-to-video`).                         |
| width                  | Width of the generated video in pixels.                                                         |
| height                 | Height of the generated video in pixels.                                                        |
| num_frames             | Number of frames to generate for the video.                                                     |
| fps                    | Frames per second for the output video.                                                         |
| num_inference_steps    | Number of denoising steps during video generation (higher values improve quality).               |
| guidance_scale         | Controls adherence to the prompt (higher values = stronger adherence, lower = better quality).   |
| seed                   | Seed for reproducibility (use -1 for random behavior).                                           |
| output_path            | Path to save the generated video.                                                               |
| save_last_frame        | Option to save the last frame of the video as an uncompressed PNG file.                          |
