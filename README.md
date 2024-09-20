# FacePop

![FacePop Logo](resources/logo.png)
*<!-- Note: Add a logo image here, e.g., `images/logo.png` -->*
![FacePop Logo](resources/logo2.png)
*<!-- Note: Add a logo image here, e.g., `images/logo.png` -->*

<img src="resources/face_crop.png" alt="Face Crop + Background Removal" width="300"/>
An example of the low resolution image cropped, background removed, angle set for best generative img2img processing and first processing pass.

<img src="resources/face_alignment.png" alt="Landmark Detection and Facial Upright Alignment" width="300"/>
The red dots indicate the original position of the landmark detection. The image behind it is the face tilted to the most ideal upright angle for generative img2img processing.

<img src="resources/face_mask.png" alt="Final pass masking using Inpainting mask, combines with Inpainting" width="300"/>
For final image composite a robust mask inpainting mask is applied so it doesn't receive a second generative process, but the rest of the image does. This allows for very good results of generative blending with the rest of the image.

**FacePop** is a robust extension for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that enhances image processing by detecting, enhancing, and managing faces within images. Leveraging advanced technologies like [Mediapipe](https://mediapipe.dev/), [MODNet](https://github.com/ZHKKKe/MODNet), and [ControlNet](https://github.com/lllyasviel/ControlNet), FacePop streamlines tasks such as face detection, background removal, and image enhancement directly within your Stable Diffusion workflow.

## Motivation

FacePop was developed as a solution to the limitations encountered with existing tools like **Zoom Enhancer**, eliminating the need for additional extensions such as **Unprompted**. While **Zoom Enhancer** provides basic facial zoom capabilities, FacePop offers more control and flexibility over facial detection and processing features. This extension not only enhances the zoom functionality but also integrates seamlessly with other popular plugins like **ControlNet**, **ReActor**, and **After Detailer**.

By leveraging these integrations, FacePop allows users to customize their image processing workflows extensively, ensuring that each facial enhancement task can be fine-tuned to meet specific requirements. Whether it's for facial feature refinement, background removal, or advanced image manipulation techniques, FacePop provides the necessary tools to achieve better results. This approach ensures that users have greater control over their image enhancements, making FacePop a valuable tool for anyone looking to improve their Stable Diffusion experience.

## What it Does

**FacePop** simplifies facial image processing within the Stable Diffusion workflow by offering a straightforward and customizable solution. Here's how FacePop works:

1. **Face Detection**:
    - **Accurate Identification**: Uses [Mediapipe](https://mediapipe.dev/) to accurately detect and locate all faces within an image.
    - **Landmark Detection**: Identifies key facial landmarks to ensure proper cropping and alignment.

2. **Face Cropping and Upscaling**:
    - **Automatic Cropping**: Each detected face is cropped from the main image based on user-defined settings for width, height, and padding.
    - **Upscaling**: The cropped faces are upscaled to the specified dimensions, enhancing their resolution and detail.

3. **Separate Processing of Faces**:
    - **Individual Enhancements**: Each cropped face is processed separately, allowing for targeted enhancements like color correction, sharpening, and background removal using tools like [MODNet](https://github.com/ZHKKKe/MODNet).
    - **Custom Settings**: Users can adjust processing parameters to suit specific needs for each face.

4. **Mask Creation and Integration**:
    - **Creating Masks**: After processing, FacePop generates masks around each enhanced face.
    - **Blending**: These masks help seamlessly blend the enhanced faces back into the original image without visible seams or artifacts.

5. **Final Image Composite and Processing**:
    - **Reintegration**: The processed faces are placed back onto their original locations in the main image using the created masks.
    - **Final Enhancements**: The entire composite image is then subjected to final processing, which may include additional enhancements or integration with other plugins like [ControlNet](https://github.com/lllyasviel/ControlNet), [ReActor](https://github.com/your-repo/ReActor), and [After Detailer](https://github.com/your-repo/After-Detailer).
    - **Polished Output**: Ensures that all elements of the image are harmoniously combined for a professional-grade final image.

6. **Integration with Other Plugins**:
    - **Compatibility**: Designed to work seamlessly with popular extensions like **ControlNet**, **ReActor**, and **After Detailer**, allowing users to leverage multiple tools within a single workflow.
    - **Enhanced Features**: Provides additional capabilities without the need for extra dependencies like the **Unprompted** extension.

7. **User-Friendly Controls and Settings**:
    - **Easy Configuration**: Offers an intuitive interface within the Stable Diffusion Web UI for configuring face detection, cropping dimensions, padding, and processing parameters.
    - **Adjustable Parameters**: Users can fine-tune various settings to control the processing workflow and achieve desired results.

8. **Performance Optimization**:
    - **Efficient Processing**: Handles face detection and processing separately to optimize performance, enabling faster image enhancements even with multiple faces.
    - **Resource Management**: Manages system resources effectively to ensure smooth operation without significant slowdowns, even when integrating with other resource-intensive plugins.

**Benefits of Using FacePop**:

- **Greater Control**: Provides more precise control over facial zoom enhancements compared to tools like **Zoom Enhancer**, allowing for tailored adjustments.
- **Higher Quality**: Delivers better quality results through advanced processing techniques and seamless plugin integrations.
- **Flexible Usage**: Suitable for a wide range of use cases, from simple face enhancements to complex image manipulations involving multiple plugins.
- **Simplified Workflow**: Combines multiple processing steps into a single, cohesive workflow, reducing the need for multiple extensions and simplifying the overall image enhancement process.
- **Scalable**: Efficiently handles images with numerous faces, making it ideal for both individual and bulk processing tasks.

By addressing the limitations of existing tools and offering a more versatile approach to facial image processing, FacePop becomes an essential extension for users seeking advanced control and improved results within the Stable Diffusion ecosystem. Whether you're enhancing portraits, refining facial features, or integrating with other image manipulation tools, FacePop provides the necessary capabilities to achieve exceptional outcomes.

## Table of Contents

- [Top](#facepop)
- [Motivation](#motivation)
- [What it Does](#what-it-does)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install via Extensions Manager](#install-via-extensions-manager)
- [Usage](#usage)
- [Deny Scripts File](#deny-scripts-list)
- [Screenshots](#screenshots)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Face Detection**: Utilizes Mediapipe's Face Mesh and Face Detection for precise localization of faces.
- **Background Removal**: Integrates MODNet for effective background removal, focusing on portrait matting.
- **Image Enhancement**: Offers multiple enhancement methods including Unsharp Mask, Bilateral Filter, Median Filter, and a Hybrid approach.
- **Auto-Alignment**: Automatically aligns faces based on detected landmarks to ensure consistent processing.
- **Batch Processing**: Efficiently handles multiple faces with configurable batch settings.
- **ControlNet Integration**: Supports ControlNet for advanced image manipulation if installed.
- **Customizable Settings**: Provides a comprehensive UI with adjustable parameters for tailored processing.
- **Debugging Tools**: Includes debugging features to help troubleshoot and optimize processing steps.

## Installation

### Prerequisites

Before installing FacePop, ensure you have the following:

- [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) installed and set up.
- **Git** installed on your system to enable extension installation via URL.

### Install via Extensions Manager

1. **Open Stable Diffusion Web UI**:
   Launch your Stable Diffusion Web UI as you normally would.

2. **Navigate to Extensions**:
   Click on the `Extensions` tab in the sidebar.

3. **Install from URL**:
   - Click on the `Install from URL` button.
   - Enter the repository URL: `https://github.com/TheCodeSlinger/FacePop.git`
   - Click `Install`.

   ![Install from URL](resources/install_from_url.png)
   *<!-- Note: Add an image showing the "Install from URL" button and where to input the URL -->*

4. **Restart Web UI**:
   After installation, you may be prompted to restart the Web UI. Follow the prompt to apply changes.

5. **Verify Installation**:
   - Once restarted, navigate to the `Extensions` tab to ensure `FacePop` is listed.
   - Alternatively, check within the Img2Img interface for the `FacePop` panel.

## Usage

1. **Open Img2Img Interface**:
   Navigate to the `Img2Img` tab within the Stable Diffusion Web UI.

2. **Locate FacePop Panel**:
   Scroll down to find the `FacePop` accordion or panel.

3. **Configure Settings**:
   - **Enable FacePop**: Toggle the extension on.
   - **Use MODNet**: Enable background removal.
   - **Enable Aggressive Face Detection**: Enhance face detection capabilities.
   - **Adjust Parameters**: Set face width, height, padding, detection confidence, maximum faces, and more as per your requirements.

   ![FacePop Settings](resources/facepop_settings.png)
   *<!-- Note: Add an image showing the FacePop settings within the Img2Img interface -->*

4. **Process Image**:
   - Upload or generate an image with faces.
   - Configure the desired settings in the FacePop panel.
   - Click `Generate` to process the image with FacePop enhancements.

5. **Review Results**:
   - Processed images will display enhanced faces.
   - Check the output directory for individual processed face images and the final composite image.

## Deny Scripts List

`deny_scripts_list.txt` is a crucial configuration file in the FacePop extension that allows users to specify which scripts should be ignored during specific processing stages. This ensures seamless integration and prevents potential conflicts between FacePop and other extensions or scripts within the Stable Diffusion Web UI.

### Purpose

The primary purpose of the `deny_scripts_list.txt` file is to **control the activation of certain scripts** during different phases of the image processing workflow. By specifying scripts to be ignored, FacePop can operate without interference, ensuring optimal performance and stability.

### File Structure

The `deny_scripts_list.txt` file is organized into sections, each corresponding to a different processing stage. Within each section, you can list the names of scripts that should be disabled during that particular stage.

```ini
; ignore any of these scripts when faces are being processed
[faces]
ADetailer

; ignore any of these scripts during final image composite
[final]
ReActor
```
## Screenshots

![FacePop UI Panel](resources/ui_screenshot.png)
*Figure 1: FacePop panel within the Img2Img interface.*

![Before and After](resources/before_after.png)
*Figure 2: Comparison of an image before and after FacePop processing.*

*<!-- Note: Replace the above image paths with actual image file paths once added to the repository -->*

## Dependencies

FacePop relies on several third-party libraries and tools to function effectively:

- [Mediapipe](https://mediapipe.dev/): For face detection and landmark recognition.
- [MODNet](https://github.com/ZHKKKe/MODNet): For background removal and matting.
- [ControlNet](https://github.com/lllyasviel/ControlNet): (Optional) For advanced image manipulation.
- [PyTorch](https://pytorch.org/): Required for MODNet and ControlNet.
- [OpenCV](https://opencv.org/): For image processing tasks.
- [Gradio](https://gradio.app/): For building the UI components.
- [Torchvision](https://pytorch.org/vision/stable/index.html): For image transformations.

These dependencies are automatically handled during installation via the Extensions Manager. If you encounter issues, you can manually install them using `pip`:

```bash
pip install mediapipe modnet torch torchvision opencv-python gradio
```
## **Additional Resources**

### **Model Files**

FacePop utilizes pre-trained models for efficient and accurate face detection. Below are the details of the essential model files used, including their descriptions and licensing information.

#### **1. deploy.prototxt**

- **Description**:
  
  The `deploy.prototxt` file defines the architecture of the Single Shot Multibox Detector (SSD) model used for face detection. It specifies the layers, parameters, and configurations required to deploy the model using OpenCV's Deep Neural Network (DNN) module.

- **Download Link**:
  
  You can download the `deploy.prototxt` file from the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt).

- **License**:
  
  The `deploy.prototxt` file is part of the OpenCV project and is licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause). Below is a summary of the license:

  > **BSD 3-Clause License**
  >
  > Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  >
  > 1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
  > 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
  > 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
  >
  > **Disclaimer**:
  >
  > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

#### **2. res10_300x300_ssd_iter_140000.caffemodel**

- **Description**:
  
  The `res10_300x300_ssd_iter_140000.caffemodel` is the pre-trained weights file for the SSD (Single Shot Multibox Detector) model with a ResNet-10 backbone. This model is optimized for face detection and is widely used in conjunction with the `deploy.prototxt` configuration file to perform real-time face detection tasks.

- **Download Link**:
  
  You can download the `res10_300x300_ssd_iter_140000.caffemodel` file from the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel).

- **License**:
  
  Similar to the `deploy.prototxt`, the `res10_300x300_ssd_iter_140000.caffemodel` is part of the OpenCV project and is licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause). Below is a summary of the license:

  > **BSD 3-Clause License**
  >
  > Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  >
  > 1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
  > 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
  > 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
  >
  > **Disclaimer**:
  >
  > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

#### **3. How to Obtain and Integrate the Model Files**

1. **Download the Files**:
   
   - **deploy.prototxt**: [Download Here](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   - **res10_300x300_ssd_iter_140000.caffemodel**: [Download Here](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel)

2. **Place the Files in \stable-diffusion-webui\extensions\FacePop\scripts\**:
   
   - Create a `models` directory within your `FacePop` project folder:
     ```bash
     mkdir models
     ```
   - Move the downloaded files into the `models` directory:
     ```bash
     mv \stable-diffusion-webui\extensions\FacePop\scripts\deploy.prototxt models/
     mv \stable-diffusion-webui\extensions\FacePop\scripts\res10_300x300_ssd_iter_140000.caffemodel models/
     ```



---
