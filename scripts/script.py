import os
import modules.scripts as scripts
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import importlib
import copy
import math
import datetime
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from modnet.models.modnet import MODNet
#import pprint # Only needed if you uncomment the p dump code for debugging output

# Import the process_images function
from modules.processing import process_images, StableDiffusionProcessingImg2Img

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.9)

class FacePopState:
    """
    Maintains the state for the FacePop extension, storing information about face detection,
    processing parameters, and model references.
    """
    # Static variables
    original_image = np.zeros((512, 512, 3), dtype=np.uint8)  # The original input image being processed
    is_processing_faces = False  # Flag indicating if face processing is currently ongoing
    first_p = None  # The first processing object, possibly storing pipeline parameters
    first_cn_list = None  # List related to ControlNet processing
    processed_faces = []  # List of processed face images and their metadata
    faces = []  # List of detected face coordinates
    controlNetModule = None  # ControlNet module instance, if loaded
    total_faces = 0  # Total number of detected faces
    scale_width = 0  # Width to scale face images
    scale_height = 0  # Height to scale face images
    proc_width = 0  # Processing image width
    proc_height = 0  # Processing image height
    padding = 0  # Padding around detected faces
    max_faces = 0  # Maximum number of faces to process
    enhancement_method = "Unsharp Mask"  # Default enhancement method is now Unsharp Mask
    confidence_threshold = 0.4  # Updated default to match Mediapipe's default
    batch_size = 0  # Number of faces to process per batch
    batch_count = 0  # Total number of batches
    batch_total = -1  # Total number of faces to process
    batch_i = 0  # Current batch index
    countdown = 0  # Countdown timer for processing
    started = False  # Flag indicating if processing has started
    enabled = True  # Flag indicating if the extension is enabled
    face_index = 0  # Index of the current face being processed
    output_path = None  # Path to save output images
    timestamp = None  # Timestamp for naming output files
    auto_align_faces = True  # Flag to auto-align faces based on landmarks
    rotation_angle_threshold = 10  # New default rotation threshold (degrees)
    keep_ratio = True  # Flag to maintain aspect ratio when resizing faces
    modnet_model = None  # Placeholder for MODNet model
    modnet_ref = None  # Reference to the MODNet model
    debug_path = "debug"  # Path for debug logs
    output_faces = True  # Flag to save individual processed face images
    aggressive_detection = False  # Flag to enable aggressive face detection (image rotation)
    combined_mask = None  # Combined mask for inpainting and face processing
    final_image_pass = False  # Flag indicating if the final image pass has been completed
    preview_images = []  # List of preview images generated during processing
    scripts = []  # List of scripts to manage
    face_detection = None  # Static variable for Mediapipe Face Detection
    deny_scripts_dict = {}  # Dictionary to manage script enabling/disabling
    upscale_to_original = False # Final Composite image will be saved to scale (after processing) of original source
    face_seperate_proc = False # Use different processing settings for faces than what is used for final composite image
    face_samping_steps = 40 # seperate face sampling steps
    face_cfg_scale = 7.0 # seperate face CFG scale
    face_denoising = 0.4 # seperate face denoising strength
    ssd_model_loaded = False

    @staticmethod
    def reset():
        #print('[FacePop Debug] [[[[[[[ RESET() ]]]]]]]')
        """Reset all static variables to their default state."""
        FacePopState.original_image = np.zeros((512, 512, 3), dtype=np.uint8)
        FacePopState.is_processing_faces = False
        FacePopState.first_p = None
        FacePopState.first_cn_list = None
        FacePopState.processed_faces = []
        FacePopState.faces = []
        FacePopState.total_faces = 0
        FacePopState.scale_width = 0
        FacePopState.scale_height = 0
        FacePopState.proc_width = 0
        FacePopState.proc_height = 0
        FacePopState.padding = 0
        FacePopState.max_faces = 0
        FacePopState.enhancement_method = "Unsharp Mask"
        FacePopState.confidence_threshold = 0.5  # Reset to default
        FacePopState.batch_count = 0
        FacePopState.batch_size = 0
        FacePopState.batch_total = -1
        FacePopState.batch_i = 0
        FacePopState.countdown = 0
        FacePopState.enabled = True
        FacePopState.face_index = 0
        FacePopState.output_path = None
        FacePopState.timestamp = None
        FacePopState.auto_align_faces = True  # Correct initialization
        FacePopState.rotation_angle_threshold = 10  # Reset rotation threshold
        FacePopState.keep_ratio = True  # Reset Keep Ratio to default (checked)
        FacePopState.modnet_model = None
        FacePopState.output_faces = True
        FacePopState.aggressive_detection = False
        FacePopState.combined_mask = None
        FacePopState.final_image_pass = False
        FacePopState.scripts = []
        FacePopState.deny_scripts_dict = {}
        FacePopState.upscale_to_original = False
        FacePopState.face_seperate_proc = False
        FacePopState.face_samping_steps = 40
        FacePopState.face_cfg_scale = 7.0
        FacePopState.face_denoising = 0.4
        # FacePopState.preview_images = []
        # ControlNetModule should not be reset as it's likely loaded once and reused.
        # FacePopState.ssd_model_loaded = False
        # Reset and initialize Face Detection with default confidence_threshold
        #FacePopState.initialize_face_detection(min_confidence=FacePopState.confidence_threshold)

    @staticmethod
    def initialize_face_detection(min_confidence=0.3):
        """
        Initialize or update the Mediapipe Face Detection with the given confidence threshold.

        :param min_confidence: Float between 0.0 and 1.0 representing the minimum detection confidence.
        """
        if FacePopState.face_detection is not None:
            FacePopState.face_detection.close()  # Close previous instance if exists
            FacePopState.face_detection = None

        # Initialize Mediapipe Face Detection with updated confidence
        FacePopState.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0: Short-range, 1: Full-range
            min_detection_confidence=min_confidence
        )
        if FacePopState.face_detection:
            print(f"[FacePop Debug] Initialized Mediapipe Face Detection.")# with min_detection_confidence={min_confidence}")

    @classmethod
    def load_modnet_model(cls, model_filename='modnet_photographic_portrait_matting.ckpt'):
        """
        Loads the MODNet model for background removal and stores it in the FacePopState for later use.

        This class method performs the following operations:
        - Constructs the model path based on the current script directory and the provided filename.
        - Checks if the model file exists at the specified path.
        - Initializes the MODNet model and wraps it with DataParallel for multi-GPU support if available.
        - Loads the model weights from the checkpoint file.
        - Transfers the model to GPU if CUDA is available; otherwise, it remains on CPU.
        - Sets the model to evaluation mode to disable training-specific layers like dropout.
        - Updates the FacePopState with the loaded MODNet model for use in image processing tasks.

        :param model_filename: 
            A string specifying the filename of the MODNet model checkpoint. 
            Defaults to 'modnet_photographic_portrait_matting.ckpt'.

        :return: 
            None. The loaded MODNet model is stored in `FacePopState.modnet_model`.
        """
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, model_filename)
    
        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"MODNet model checkpoint not found at: {model_path}")
            cls.modnet_model = None
            return
    
        # Initialize MODNet model
        cls.modnet_model = MODNet(backbone_pretrained=False)

        # Wrap the model with DataParallel if using multiple GPUs
        cls.modnet_model = nn.DataParallel(cls.modnet_model)
    
        # Check for GPU availability
        if torch.cuda.is_available():
            print('Use GPU...')
            cls.modnet_model = cls.modnet_model.cuda()
            cls.modnet_model.load_state_dict(torch.load(model_path))
        else:
            print('Use CPU...')
            cls.modnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
        # Set the model to evaluation mode
        cls.modnet_model.eval()
        print("MODNet model loaded successfully.")

class FacePopScript(scripts.Script):

    def __init__(self):
        super().__init__()
        self.debug = False

    # Extension title in the menu UI
    def title(self):
        return "FacePop"

    # Show in Img2Img only
    def show(self, is_img2img):
        return scripts.AlwaysVisible
        
    def load_ssd_model(self):
        """
        Initializes the FacePopScript class by loading essential models, including ControlNet if available.

        This constructor performs the following operations:
        - Sets up initial state variables and configurations.
        - Attempts to load the ControlNet model for advanced image processing capabilities.
        - If ControlNet is successfully loaded, it is stored within FacePopState for later use during face processing tasks.

        :return: 
            None. This method initializes the class instance and prepares necessary resources for face processing.
        """
        if FacePopState.ssd_model_loaded:
            return

        FacePopState.ssd_model_loaded = True

        # Define the path to the model files
        script_dir = os.path.dirname(__file__)
        model_prototxt = os.path.join(script_dir, "deploy.prototxt")
        model_caffemodel = os.path.join(script_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        # Load the SSD model with the full path
        try:
            self.net = cv2.dnn.readNetFromCaffe(model_prototxt, model_caffemodel)
            print("[FacePop Debug] Loaded OpenCV SSD model successfully.")
        except Exception as e:
            print(f"[FacePop Debug] Failed to load OpenCV SSD model: {e}")
            self.net = None

        # Dynamically import ControlNet and store it in the static state
        if FacePopState.controlNetModule is None:
            try:
                FacePopState.controlNetModule = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
                print("[FacePop Debug] EXTERNAL CODE LOADED")
            except ImportError:
                FacePopState.controlNetModule = None
                print("[FacePop Debug] NO EXTERNAL CODE FOUND")

    def ui(self, is_img2img):
        """
        Constructs the user interface for the FacePop extension within AUTOMATIC1111's stable-diffusion-webui.
        The UI is organized into distinct sections for basic features, debugging, advanced settings, and separate
        face processing to facilitate future development and enhance user experience.
    
        :param is_img2img:
            A boolean flag indicating whether the UI is being constructed for the Img2Img interface. If `True`,
            certain UI components may be adjusted or displayed differently to better suit the Img2Img workflow.
    
        :return:
            A list of Gradio components corresponding to the parameters of the `process` method, maintaining the
            order required for proper functionality.
        """
        if not is_img2img:
            return None

        with gr.Accordion("FacePop", open=False):
            # --------------------- Basic Features ---------------------
            with gr.Group():
                # Enable FacePop, Use MODNet, Aggressive Face Detection in one row
                with gr.Row():
                    enabled = gr.Checkbox(label="Enable FacePop", value=False, interactive=True)  # Enable/Disable extension
                    modnet_checkbox = gr.Checkbox(label="Use MODNet", value=True, interactive=True)  # Enable MODNet
                    aggressive_detection_checkbox = gr.Checkbox(label="Enable Aggressive Face Detection", value=True, interactive=True)  # Aggressive detection
    
                # Face Width and Height sliders in one row
                with gr.Row():
                    scale_width_slider = gr.Slider(minimum=64, maximum=2048, step=1, label="Face Width", value=720, interactive=True)  # Scale width
                    scale_height_slider = gr.Slider(minimum=64, maximum=2048, step=1, label="Face Height", value=720, interactive=True)  # Scale height
    
                # Padding and Maintain Aspect Ratio in one row
                with gr.Row():
                    padding_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Padding %", value=35, interactive=True)  # Padding
                    keep_ratio_checkbox = gr.Checkbox(label="Maintain Aspect Ratio", value=True, interactive=True)  # Keep aspect ratio

                # Detection Confidence Threshold and Maximum Faces to Process in one row
                with gr.Row():
                    confidence_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, label="Detection Confidence Threshold", value=0.5, interactive=True)  # Confidence threshold
                    max_faces_slider = gr.Slider(minimum=0, maximum=32, step=1, label="Maximum Faces to Process", value=5, interactive=True)  # Max faces
    
            # --------------------- Separate Face Processing Section ---------------------
            with gr.Accordion("Separate Face Processing", open=False):
                with gr.Group():
                    # Separate Face Processing Checkbox and Sampling Steps just below
                    separate_face_processing_checkbox = gr.Checkbox(label="Separate Face Processing", value=False, interactive=True)  # Enable separate face processing
                    sampling_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Sampling Steps for Face Processing", value=40, interactive=True)  # Sampling steps for faces
    
                    # CFG Scale and Denoising Strength in one row
                    with gr.Row():
                        cfg_scale_slider = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, label="CFG Scale for Face Processing", value=7.0, interactive=True)  # CFG Scale for faces
                        denoising_strength_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Denoising Strength for Face Processing", value=0.4, interactive=True)  # Denoising strength for faces
    
            # --------------------- Advanced Settings Section ---------------------
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Group():
                    # Auto Align Faces and Upscale To Original Source on the same line
                    with gr.Row():
                        auto_align_faces_checkbox = gr.Checkbox(label="Auto-Align Faces Based on Landmarks", value=True, interactive=True)  # Auto-align faces
                        rotation_angle_slider = gr.Slider(minimum=0, maximum=45, step=1, label="Rotation Angle Threshold (degrees)", value=10, interactive=True)  # Rotation angle threshold

                    # Image Enhancement Method and Rotation Angle Threshold on the same line
                    with gr.Row():
                        enhancement_dropdown = gr.Dropdown(choices=["Unsharp Mask", "Bilateral Filter", "Median Filter", "Hybrid"], label="Image Enhancement Method", value="Unsharp Mask", interactive=True)  # Enhancement method
                        upscale_to_original_source_checkbox = gr.Checkbox(label="Upscale To Original Source", value=False, interactive=True)  # Upscale to original source

                    # Custom Output Path
                    output_path = gr.Textbox(label="Custom Output Path", value="[date]", placeholder="Enter a custom path (use [date] for current date)", interactive=True)  # Custom output path

            # --------------------- Debugging Section ---------------------
            with gr.Accordion("Debugging", open=False):
                with gr.Group():
                    # Enable Debugging and Output Faces on the same line
                    with gr.Row():
                        debug = gr.Checkbox(label="Enable Debugging", value=False, interactive=True)  # Enable debugging
                        output_faces_checkbox = gr.Checkbox(label="Output Faces", value=True, interactive=True)  # Output individual faces
    
        return [
            enabled,  # Enable FacePop
            modnet_checkbox,  # Use MODNet
            aggressive_detection_checkbox,  # Aggressive Face Detection
            scale_width_slider,  # Face Width
            scale_height_slider,  # Face Height
            padding_slider,  # Padding
            keep_ratio_checkbox,  # Maintain Aspect Ratio
            confidence_slider,  # Confidence Threshold
            max_faces_slider,  # Max Faces
            debug,  # Enable Debugging
            output_faces_checkbox,  # Output Faces
            auto_align_faces_checkbox,  # Auto-Align Faces
            upscale_to_original_source_checkbox,  # Upscale To Original Source
            enhancement_dropdown,  # Enhancement Method
            rotation_angle_slider,  # Rotation Angle Threshold
            output_path,  # Custom Output Path
            separate_face_processing_checkbox,  # Separate Face Processing
            sampling_steps_slider,  # Sampling Steps for Face Processing
            cfg_scale_slider,  # CFG Scale for Face Processing
            denoising_strength_slider  # Denoising Strength for Face Processing
        ]


    def process(self, p, enabled=True, modnet_checkbox=True, aggressive_detection_checkbox=True,
                scale_width_slider=1.0, scale_height_slider=1.0, padding_slider=0,
                keep_ratio_checkbox=True, confidence_slider=0.5, max_faces_slider=5,
                debug=False, output_faces_checkbox=True, auto_align_faces_checkbox=True,
                upscale_to_original_source_checkbox=True, enhancement_dropdown="None",
                rotation_angle_slider=0.0, output_path="", separate_face_processing_checkbox=False,
                sampling_steps_slider=20, cfg_scale_slider=7.0, denoising_strength_slider=0.75):
    # Method implementation
        """
        Main processing method that handles face detection, enhancement, and other operations.
        It also initializes batch processing parameters and manages the overall workflow of the extension.

        This method performs the following tasks:
        - Initializes debug settings and checks if the extension is enabled.
        - Sets up the output directory based on user input, supporting dynamic date-based paths.
        - Configures processing parameters such as scaling, padding, confidence thresholds, and enhancement methods.
        - Detects faces in the input image using Mediapipe and optionally falls back to OpenCV's SSD face detection.
        - Crops, resizes, and enhances detected faces.
        - Supports both batch and non-batch processing modes.
        - Handles post-processing tasks, including overlaying processed faces onto the base image and saving the final output.
        - Manages the state of the extension to prevent redundant processing and ensure proper cleanup after operations.

        :return: 
            None. This method modifies the `processed.images` attribute by replacing it with the final 
            composite images stored in `FacePopState.preview_images`.
        """
        if not isinstance(p, StableDiffusionProcessingImg2Img):
            return

        # Initialize the debug attribute
        self.debug = debug

        if self.debug:
            print("[FacePop Debug] Entered 'process' method")

        if FacePopState.final_image_pass == True:
            return
            
        if FacePopState.started == True:
            if self.debug:
                print(f"[FacePop Debug] Stopping Process from running again.")
            return

        # If the extension is not enabled, do nothing
        if not enabled:
            #if self.debug:
            #    print(f"[FacePop Debug] Extension not enabled. Exiting 'process' method.")
            FacePopState.enabled = False
            return
        if not FacePopState.enabled:
            return

        if FacePopState.started == False:
            FacePopState.started = True
            if self.debug:
                print("[FacePop Debug] ------------------------- STARTING FACEPOP -------------------------")

                self.load_ssd_model() # Load SSD Model, if not already loaded

            FacePopState.reset() # reset on startup just to be sure
            ## Load deny scripts dictionary
            scripts_deny_file = os.path.join(os.path.dirname(__file__), "deny_scripts_list.txt")
            if os.path.exists(scripts_deny_file):
                FacePopState.deny_scripts_dict = self.load_scripts_from_file(scripts_deny_file)
                if self.debug:
                    print("[FacePop Debug] Loaded deny scripts list.")
            else:
                if self.debug:
                    print(f"[FacePop Debug] Could not load deny scripts list from {scripts_deny_file}")


        # **New Addition: Prevent Automatic1111 from saving images**
        p.do_not_save_samples = True
        #p.do_not_save_grid = True
        if self.debug:
            print("[FacePop Debug] Set 'do_not_save_samples' and 'do_not_save_grid' to True to prevent default saving.")


        # Check if the modnet_checkbox is checked and if the model is not already loaded
        if modnet_checkbox:
            # If the model is not loaded, attempt to load it
            if FacePopState.modnet_model is None:
                if self.debug:
                    print("[FacePop Debug] MODNet model is not loaded. Attempting to load...")

                try:
                    # Check if modnet_ref exists and assign it to modnet_model, otherwise load the model
                    if FacePopState.modnet_ref is not None:
                        FacePopState.modnet_model = FacePopState.modnet_ref
                        if self.debug:
                            print("[FacePop Debug] MODNet model reference restored from modnet_ref.")
                    else:
                        # Load MODNet model and set both modnet_model and modnet_ref
                        FacePopState.load_modnet_model()  # Load the model
                        FacePopState.modnet_ref = FacePopState.modnet_model
                        if self.debug:
                            print("[FacePop Debug] MODNet model loaded and reference saved.")
                except FileNotFoundError as e:
                    print(f"[FacePop Debug] {str(e)}")
                    FacePopState.enabled = False
                    return  # If the model loading fails, exit the method
            else:
                if self.debug:
                    print("[FacePop Debug] MODNet model is already loaded.")

        # Get the output path using the new method
        FacePopState.output_path = self.get_output_path(output_path, p.outpath_samples)
        # Set the debug path for logging if needed
        FacePopState.debug_path = os.path.join(FacePopState.output_path, "debug")
        if self.debug:
            os.makedirs(FacePopState.debug_path, exist_ok=True)
            print(f"[FacePop Debug] Output directory set to: {FacePopState.output_path}")

        # Generate and store a timestamp for file naming (only once per process)
        if FacePopState.timestamp is None:
            FacePopState.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.debug:
                print(f"[FacePop Debug] Timestamp for file names: {FacePopState.timestamp}")

        # Dumping 'p' object to a .txt file if debug is enabled
        dump_debugger = False
        #if self.debug and dump_debugger:
        #    # Create the file path with the timestamp
        #    dump_file_path = os.path.join(self.debug_path(), f"p_dump_{FacePopState.timestamp}.txt")
        #    # Dump the attributes to a text file with UTF-8 encoding
        #    with open(dump_file_path, 'w', encoding='utf-8') as file:
        #        pp = pprint.PrettyPrinter(indent=4)  # Pretty printer with 4-space indentation
        #        p_attributes = vars(p)  # Get the __dict__ of the 'p' object (all its attributes)
        #        file.write("[FacePop Debug] Dumping 'p' object attributes:\n\n")
        #        file.write(pp.pformat(p_attributes))  # Write formatted attributes to the file
            # Optional: Log the file location
        #    print(f"[FacePop Debug] 'p' object attributes dumped to {dump_file_path}")

        # Assign the slider values to the instance
        FacePopState.scale_width = int(scale_width_slider)
        FacePopState.scale_height = int(scale_height_slider)
        FacePopState.padding = int(padding_slider)
        FacePopState.confidence_threshold = float(confidence_slider)
        FacePopState.max_faces = int(max_faces_slider)
        FacePopState.enhancement_method = enhancement_dropdown
        FacePopState.auto_align_faces = auto_align_faces_checkbox
        FacePopState.rotation_angle_threshold = int(rotation_angle_slider)
        FacePopState.keep_ratio = keep_ratio_checkbox
        FacePopState.output_faces = output_faces_checkbox
        FacePopState.aggressive_detection = aggressive_detection_checkbox
        FacePopState.upscale_to_original = upscale_to_original_source_checkbox
        FacePopState.face_seperate_proc = separate_face_processing_checkbox
        FacePopState.face_samping_steps = sampling_steps_slider
        FacePopState.face_cfg_scale = cfg_scale_slider
        FacePopState.face_denoising = denoising_strength_slider

        # Capture the actual original image dimensions before any processing
        if FacePopState.proc_width == 0:
            FacePopState.original_image = p.init_images[0].copy()  # Make a true copy of the original image
            FacePopState.original_image_size = FacePopState.original_image.size  # Store original size (width, height)
            FacePopState.proc_width = p.width
            FacePopState.proc_height = p.height
            FacePopState.preview_images = [] # clear previews
            if self.debug:
                print(f"[FacePop Debug] Original image dimensions: {FacePopState.original_image_size} proc {FacePopState.proc_width} x {FacePopState.proc_height}")

        if FacePopState.batch_total == -1:
            FacePopState.batch_total = p.n_iter * p.batch_size
            FacePopState.batch_count = p.n_iter
            FacePopState.batch_size = p.batch_size
            if self.debug:
                print(f"[FacePop Debug] batch count (n_iter): {FacePopState.batch_count}")
                print(f"[FacePop Debug] batch size: {FacePopState.batch_size}")
                print(f"[FacePop Debug] Batching Total: {FacePopState.batch_total}")

        if FacePopState.total_faces != 0:
            return
        else:  # detect faces; this is called only on the first process
            # Convert PIL image to NumPy array
            init_image = np.array(p.init_images[0])
            # Perform face detection
            FacePopState.faces = self.detect_faces(init_image, p)  # Pass 'p' here
            # Set the total number of faces to process
            FacePopState.total_faces = len(FacePopState.faces)
            print(f"[FacePop Debug] Number of faces detected: {FacePopState.total_faces}")
            FacePopState.countdown = FacePopState.total_faces# + 1
            if self.debug:
                print(f"[FacePop Debug] Batching Postprocess Countdown Starting at: {FacePopState.countdown}")
            if FacePopState.total_faces == 0:
                if self.debug:
                    print("[FacePop Debug] No faces found, resetting state.")
                FacePopState.enabled = False  # Allow the extension to start fresh in the next run
                return

            # Prevent re-entrance if already processing faces
            if FacePopState.is_processing_faces:
                if self.debug:
                    print(f"[FacePop Debug] Already processing faces, skipping.")
                return

            # Set the flag to indicate face processing is in progress
            FacePopState.is_processing_faces = True

            try:
                # Store the original p for future use (if it's the first call)
                if FacePopState.first_p is None:
                    FacePopState.first_p = copy.copy(p)
                    if FacePopState.controlNetModule:
                        FacePopState.first_cn_list = FacePopState.controlNetModule.get_all_units_in_processing(FacePopState.first_p)

                if FacePopState.faces is not None and FacePopState.total_faces > 0:
                    # Process the detected faces and accumulate processed faces
                    FacePopState.processed_faces = self.crop_upscale_and_process_faces(init_image, p)
                    #if self.debug:
                    #    print(f"[FacePop Debug] Processed faces count: {len(FacePopState.processed_faces)}")

            finally:
                # Reset the flag, but we don't reset the whole state here to avoid losing face info
                FacePopState.is_processing_faces = False
                # Trash P..can't stop it from running so just make it as short as possible
                #p.init_images = []  # Clear the input images # ERRORS
                p.width = 100
                p.height = 100

            if self.debug:
                print("[FacePop Debug] Exiting 'process' method after processing faces.")

        return

    def crop_upscale_and_process_faces(self, image, p):
        """
        Crops, resizes, and applies image enhancement to detected faces within the input image.
        Supports both batch and non-batch processing modes.

        This method processes each detected face by:
        - Applying padding to ensure the face fits within the image bounds.
        - Resizing the face while maintaining aspect ratio if `keep_ratio` is enabled.
        - Optionally auto-aligning faces based on detected landmarks.
        - Enhancing the face image using the selected enhancement method.
        - Processing the face image through the Img2Img pipeline.
        - Optionally removing the background using MODNet.
        - Blending the processed face back into the original image.

        :param image: The input image as a NumPy array (e.g., BGR format). Represents the original image before any processing.
        :param p: The processing object from AUTOMATIC1111's pipeline, containing image data and various parameters.
        :return: A list of lists, where each sublist corresponds to a batch and contains tuples with:
                 - The processed face image (PIL Image or NumPy array).
                 - The coordinates of the face in the original image as a tuple (x, y, width, height).
                 - A status string indicating whether the face was 'processed' or 'skipped'.
                 Example:
                 [
                     [  # Batch 1
                         (processed_face_image1, (x1, y1, w1, h1), 'processed'),
                         (processed_face_image2, (x2, y2, w2, h2), 'skipped'),
                         ...
                     ],
                 ]
        """
        if self.debug:
            print(f"[FacePop Debug] Entered 'crop_upscale_and_process_faces' method with {len(FacePopState.faces)} faces to process")

        processed_faces = []  # Local list to store processed faces and their metadata

        if len(FacePopState.faces) == 0:
            print("[FacePop Debug] No faces detected, exiting 'crop_upscale_and_process_faces'")
            return processed_faces

        # Convert the inpainting mask to a NumPy array (assuming p.image_mask is a PIL image)
        mask = np.array(p.image_mask) if p.image_mask else None

        # Create a dummy placeholder (a blank image with the same size as the face crop)
        def create_dummy_image(width, height):
            return Image.new('RGB', (width, height), color='gray')  # Dummy placeholder image

        cropped_mask = None
        for iface, face_data in enumerate(FacePopState.faces):
            FacePopState.countdown -= 1

            x, y, w, h = face_data['bbox']

            padding = int(w * (FacePopState.padding * .01))

            # Apply padding and ensure the crop stays within image bounds
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, image.shape[1])
            y_end = min(y + h + padding, image.shape[0])

            face = image[y_start:y_end, x_start:x_end]

            if mask is not None:
                if self.debug:
                    print(f"[FacePop Debug] Inpainting mask detected.")

                # Check for face overlap with mask
                inpaint_region, cropped_mask = self.is_face_in_inpaint_region((x_start, y_start, x_end, y_end), mask)

                # Correct logic to check if face is in the inpainting region
                if not inpaint_region:  # Skip face if it doesn't meet the 20% threshold
                    if self.debug:
                        print(f"[FacePop Debug] Skipping face #{iface+1} as it's outside the inpainting region.")

                    # Add a dummy placeholder to maintain the number of processed faces
                    dummy_face = create_dummy_image(w, h)

                    # Ensure processed_faces is initialized properly
                    #if len(processed_faces) == 0:
                    #    processed_faces.append([])  # Initialize for non-batch mode
                    #elif len(processed_faces) <= iface:
                    #    processed_faces.append([])  # Initialize for batch mode if necessary
                        
                    # Ensure processed_faces is initialized properly
                    while len(processed_faces) <= iface:
                        processed_faces.append([])
                
                    processed_faces[iface].append((dummy_face, (x_start, y_start, x_end, y_end), 'skipped'))  # Mark as skipped

                    # Adjust countdown
                    continue  # Skip this face if it's outside the inpainting region

            # Debugging: Log the size of the face before any processing
            if self.debug:
                print(f"[FacePop Debug] Face #{iface+1} original size: {face.shape[:2]} (height, width)")

            # **NEW STEP**: Resize the face first
            if FacePopState.keep_ratio:
                aspect_ratio = face.shape[1] / face.shape[0]
                if aspect_ratio > 1:
                    new_width = FacePopState.scale_width
                    new_height = int(FacePopState.scale_width / aspect_ratio)
                else:
                    new_height = FacePopState.scale_height
                    new_width = int(FacePopState.scale_height * aspect_ratio)
            else:
                new_width = FacePopState.scale_width
                new_height = FacePopState.scale_height

            face_resized = cv2.resize(face, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # **NEW STEP**: Create a copy of the resized face for blending later
            original_face_copy = face_resized.copy()

            # Perform auto-alignment if the option is enabled
            original_angle = 0  # Default angle
            if FacePopState.auto_align_faces:
                # Detect landmarks using Mediapipe's Face Mesh on the resized face
                landmarks, eyes_mouth_mask = self.detect_landmarks(face_resized)

                angle = self.calculate_rotation_angle(landmarks)
                if angle != 0:
                    total_rotation_angle = angle

                    # Debugging: Log the rotation angle
                    if self.debug:
                        print(f"[FacePop Debug] Face #{iface+1} rotation angle: {angle:.2f} degrees")

                    # Rotate the face if necessary
                    face_resized = self.rotate_image(face_resized, total_rotation_angle, True)
                    if eyes_mouth_mask is not None:
                        eyes_mouth_mask = self.rotate_image(eyes_mouth_mask, total_rotation_angle)
                    original_angle = -total_rotation_angle  # Save the inverse angle for final overlay

                    # **NEW CODE STARTS HERE**
                    if self.debug:
                        # Draw landmarks on the rotated face
                        debug_face_with_landmarks = face_resized.copy()

                        # Check if the image has an alpha channel and convert it to BGR for drawing
                        if debug_face_with_landmarks.shape[2] == 4:
                            # Convert from BGRA to BGR
                            debug_face_with_landmarks = cv2.cvtColor(debug_face_with_landmarks, cv2.COLOR_BGRA2BGR)
                        elif debug_face_with_landmarks.shape[2] == 3:
                            # Image is already in BGR format
                            pass
                        else:
                            # Handle unexpected number of channels
                            print(f"[FacePop Debug] Unexpected number of channels in face image: {debug_face_with_landmarks.shape[2]}")

                        # Draw the landmarks
                        for (x_lm, y_lm) in landmarks:
                            cv2.circle(debug_face_with_landmarks, (int(x_lm), int(y_lm)), 2, (0, 0, 255), -1)

                        # Save the image in the debug folder
                        debug_output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_face_{iface+1}_landmarks.png")
                        cv2.imwrite(debug_output_path, debug_face_with_landmarks)
                        print(f"[FacePop Debug] Saved face with landmarks for face #{iface+1} at {debug_output_path}")
                    # **NEW CODE ENDS HERE**

                    # Debugging: Log the size of the face after rotation
                    if self.debug:
                        print(f"[FacePop Debug] Face #{iface+1} size after rotation: {face_resized.shape[:2]} (height, width)")
                else:
                    if self.debug:
                        print(f"[FacePop Debug] Face #{iface+1} skipped rotation (no significant angle detected)")
            else:
                if self.debug:
                    print(f"[FacePop Debug] Face #{iface+1} skipped alignment.")

            if eyes_mouth_mask is not None:
                # Save or use the mask as needed
                # For example, save the mask for debugging
                debug_output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_face_{iface+1}_eyes_mouth_mask.png")
                cv2.imwrite(debug_output_path, eyes_mouth_mask)

            # Apply the selected enhancement method
            if FacePopState.enhancement_method == "Unsharp Mask":
                face_resized = self.apply_unsharp_mask(face_resized)
            elif FacePopState.enhancement_method == "Bilateral Filter":
                face_resized = self.apply_bilateral_filter(face_resized)
            elif FacePopState.enhancement_method == "Median Filter":
                face_resized = self.apply_median_filter(face_resized)
            elif FacePopState.enhancement_method == "Hybrid":
                face_resized = self.enhance_image(face_resized)

            # Convert the resized and enhanced face back to an image
            face_image = Image.fromarray(face_resized)

            # Create a copy of the 'p' object for face processing
            p_copy = copy.copy(p)
            p_copy.init_images = [face_image]
            p_copy.width = new_width
            p_copy.height = new_height
            p_copy.batch_size = 1  # Process one face at a time
            p_copy.do_not_save_samples = True  # Prevents the default saving behavior

            p_copy.fp_x_start = x_start
            p_copy.fp_y_start = y_start
            p_copy.fp_x_end = x_end
            p_copy.fp_y_end = y_end
            p_copy.fp_angle = angle

            if FacePopState.face_seperate_proc:
                p_copy.steps = FacePopState.face_samping_steps
                p_copy.cfg_scale = FacePopState.face_cfg_scale
                p_copy.denoising_strength = FacePopState.face_denoising

            # Apply the cropped mask for inpainting if available
            if cropped_mask is not None:  # If there's a partial mask, resize it to match the face size
                cropped_mask_resized = cv2.resize(cropped_mask, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)  # Resize cropped mask to face size
                p_copy.mask = Image.fromarray(cropped_mask_resized)  # Assign the cropped inpainting mask
            else:
                p_copy.mask = None  # No mask needed if the face is fully within the inpainting region

            p_copy.image_mask = p_copy.mask

            # Apply ControlNet if available
            if FacePopState.controlNetModule:
                control_net_list = FacePopState.controlNetModule.get_all_units_in_processing(p_copy)
                if control_net_list and control_net_list[0].enabled:
                    FacePopState.controlNetModule.update_cn_script_in_processing(p_copy, control_net_list)

            if self.debug:
                print("[FacePop Debug] Starting PROCESS_IMAGES() for face")


            #self.toggle_scripts(False, FacePopState.deny_scripts_dict.get('faces', []))
            processed = process_images(p_copy) # Process the image using the Img2Img pipeline
            #self.toggle_scripts(True, FacePopState.deny_scripts_dict.get('faces', []))

            if self.debug:
                print("[FacePop Debug] Exited PROCESS_IMAGES()")

            # Handle batch or single image case
            num_images = len(processed.images)

            ### NON BATCH
            if FacePopState.batch_total == 1:
                # Non-batch mode: Process only image[0]
                processed_face_image = np.array(processed.images[0])

                modnet_used = False
                if FacePopState.modnet_model is not None:
                    # Convert the processed face to a PIL image for background removal
                    processed_face_pil = Image.fromarray(processed_face_image)
                    
                    # Remove the background using MODNet
                    modnet_image = self.remove_background_with_modnet(processed_face_pil)
                    
                    # Convert the MODNet result (modnet_image) back to a NumPy array for further processing
                    processed_face_image = np.array(modnet_image)  # Ensure it's in NumPy format
                    modnet_used = True  # Mark MODNet as used
                    
                    if self.debug:
                        # Save the resulting image with the background removed
                        output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_processed_face_{iface+1}_with_bg_removed.png")
                        modnet_image.save(output_path)  # Save as a PIL image
                        print(f"[FacePop Debug] Saved processed face #{iface+1} with background removed at {output_path}")

                if original_angle != 0:
                    # Rotate the processed face back to its original orientation
                    processed_face_image = self.rotate_image(processed_face_image, original_angle, True)

                    if self.debug:
                        print(f"[FacePop Debug] Rotated face back.")

                if not modnet_used:
                    processed_face_image = self.blend_with_background(original_face_copy, processed_face_image)
                
                # Convert the blended NumPy array back to a PIL Image for further processing or saving
                final_face_image = Image.fromarray(processed_face_image)

                if FacePopState.output_faces:
                    output_path = os.path.join(self.output_path(), f"{FacePopState.timestamp}_processed_face_{iface+1}.png")
                    final_face_image.save(output_path)
                    if self.debug:
                       print(f"[FacePop Debug] Saved processed face #{iface+1} as {output_path}")
            
                if len(processed_faces) == 0:
                    processed_faces.append([])

                processed_faces[0].append((final_face_image, (x_start, y_start, x_end - x_start, y_end - y_start), 'processed'))

            ### IS BATCH
            else:
                # Batch mode: Process each image from images[1] onward -- update for grid
                for batch_index in range(1, FacePopState.batch_count + 1):
                    processed_face_image = np.array(processed.images[batch_index])

                    modnet_used = False
                    if FacePopState.modnet_model is not None:
                        # Convert the processed face to a PIL image for background removal
                        processed_face_pil = Image.fromarray(processed_face_image)
                        modnet_image = self.remove_background_with_modnet(processed_face_pil)
                
                        # Convert the MODNet result (modnet_image) back to a NumPy array for further processing
                        processed_face_image = np.array(modnet_image)  # Ensure it's in NumPy format
                        modnet_used = True  # Mark MODNet as used
                
                        if self.debug:
                            # Save the resulting image with the background removed
                            output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_processed_face_{batch_index}_with_bg_removed.png")
                            modnet_image.save(output_path)
                            print(f"[FacePop Debug] Saved processed face #{batch_index} with background removed at {output_path}")

                    if original_angle != 0:
                        # Rotate the processed face back to its original orientation
                        processed_face_image = self.rotate_image(processed_face_image, original_angle, True)

                        if self.debug:
                            print(f"[FacePop Debug] Rotated face for batch {batch_index} back to original orientation.")

                    # Blend the rotated face back with the original image
                    if not modnet_used:
                        processed_face_image = self.blend_with_background(original_face_copy, processed_face_image)

                    # Convert the blended NumPy array back to a PIL Image for further processing or saving
                    final_face_image = Image.fromarray(processed_face_image)

                    if FacePopState.output_faces:
                        output_path = os.path.join(self.output_path(), f"{FacePopState.timestamp}_processed_face_{iface+1}_{batch_index}.png")
                        final_face_image.save(output_path)
                        if self.debug:
                            print(f"[FacePop Debug] Saved processed face #{iface+1} (batch {batch_index}) as {output_path}")

                    # Ensure processed_faces[batch_index - 1] is a list
                    if len(processed_faces) <= batch_index - 1:
                        processed_faces.append([])  # Initialize a new list if necessary

                    # Append the processed face data
                    processed_faces[batch_index - 1].append((final_face_image, (x_start, y_start, x_end - x_start, y_end - y_start), 'processed'))


        if self.debug:
            print("[FacePop Debug] Exiting 'crop_upscale_and_process_faces' method")

        return processed_faces

    def postprocess(self, p, processed, *args):
        """
        Handles the final postprocessing by creating the final composite image. This involves overlaying 
        processed faces onto the base image, saving the final image, and resetting the extension's state 
        once all faces have been processed.

        :param p: The processing object from AUTOMATIC1111's pipeline. Contains information such as 
                  image data, output paths, and various parameters.
        :param processed: The result object from the processing pipeline. It includes the images generated 
                          during processing.
        :param args: Additional arguments that may be passed to the method. Currently unused.
        :return: None. Modifies the `processed.images` attribute by replacing it with the final composite 
                 images stored in `FacePopState.preview_images`.
        """
        if not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        if self.debug:
            print("[FacePop Debug] Entered 'postprocess' method")

        if FacePopState.enabled == False:
            print("[FacePop Debug] PostProcess..not enabled.")
            if FacePopState.countdown == 0:  # final pass
                FacePopState.reset()
                FacePopState.started = False
            return
            
        if self.debug:
            print(f"[FacePop Debug] PostProcess countdown: {FacePopState.countdown}")

        if FacePopState.final_image_pass == True:
            return

        if FacePopState.is_processing_faces:
            if self.debug:
                print(f"[FacePop Debug] PostProcess processing faces, skipping.")
            return

        # Only process on the final pass (countdown == 0)
        if FacePopState.countdown == 0:

            p.width = FacePopState.proc_width
            p.height = FacePopState.proc_height

            if self.debug:
                #print("[FacePop Debug] Final pass: Processing and overlaying batch images")
                print(f"[FacePop Debug] Final pass: Faces detected: {len(FacePopState.faces)}")
                print(f"[FacePop Debug] Final pass: Processed faces data: {FacePopState.processed_faces}")

            FacePopState.preview_images = []

            # Check if we're in batch mode or non-batch mode
            num_images = len(processed.images)
            if self.debug:
                print(f"[FacePop Debug] Final pass: Number of images in 'processed': {num_images}")


            # Non-batch mode: process the only image at processed.images[0]
            o_image = FacePopState.original_image.copy()
            o_image  = np.array(o_image)#np.array(processed.images[0])

            ### NON BATCH MODE
            if num_images == 1:

                # Retrieve the faces corresponding to the image
                if len(FacePopState.processed_faces) > 0:
                    batch_faces = FacePopState.processed_faces[0]  # Non-batch, so only one set of faces
                    over_image = self.overlay_faces_on_image(o_image, batch_faces)#, FacePopState.faces)
                    final_image_pil = Image.fromarray(over_image)

                if FacePopState.combined_mask is not None:
                    # Ensure the mask is in grayscale mode ('L') for consistency
                    face_mask_pil = Image.fromarray(FacePopState.combined_mask)
                    if face_mask_pil.mode != 'L':
                        face_mask_pil = face_mask_pil.convert('L')
                        if self.debug:
                            print("[FacePop Debug] Final pass: Converted face_mask_pil to grayscale ('L') mode")
                    # Assign the mask to p.image_mask
                    #p.image_mask = face_mask_pil
                    if self.debug:
                        output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_mask_image.png")
                        face_mask_pil.save(output_path)
                        print(f"[FacePop Debug] Final pass: Saving debug combined_mask. {output_path}")
                else:
                    if self.debug:
                        print("[FacePop Debug] Final pass: No combined_mask found, skipping mask assignment.")

                if hasattr(p, 'image_mask') and p.image_mask is not None:
                    final_mask = self.combine_masks(p.image_mask, face_mask_pil)
                else:
                    # Handle case for txt2img where no image_mask is present
                    print("No image_mask present, skipping mask combination.")
                    final_mask = face_mask_pil  # Use only the face mask in this case

                # Resize the final mask (PIL) to proc_width and proc_height
                if final_mask is not None:
                    final_mask = final_mask.resize((FacePopState.proc_width, FacePopState.proc_height), Image.LANCZOS)
                    if self.debug:
                        #print(f"[FacePop Debug] Final pass: Resized final_mask to ({FacePopState.proc_width}, {FacePopState.proc_height})")
                        #output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_mask_inpaint_original.png")
                        #p.image_mask.save(output_path)
                        output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_mask_inpaint.png")
                        final_mask.save(output_path)
                        print(f"[FacePop Debug] Final pass: Saving combined_mask with inpaint mask. {output_path}")

                # Resize the final image (PIL) to proc_width and proc_height
                if final_image_pil is not None:
                    final_image_pil = final_image_pil.resize((FacePopState.proc_width, FacePopState.proc_height), Image.LANCZOS)
                    if self.debug:
                        print(f"[FacePop Debug] Final pass: Resized final_image_pil to ({FacePopState.proc_width}, {FacePopState.proc_height})")
                        output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_face_overlays.png")
                        final_image_pil.save(output_path)
                        print(f"[FacePop Debug] Final pass: Saving overlay image. {output_path}")

                if FacePopState.first_p is not None:
                    p_copy = copy.copy(FacePopState.first_p)
                    p_copy.image_mask = final_mask  # Assign the combined mask
                    p_copy.init_images = [final_image_pil]  # Assign the overlaid image
                    p_copy.width = FacePopState.proc_width
                    p_copy.height = FacePopState.proc_height
                    p_copy.batch_size = 1  # Set batch size to 1
                    p_copy.do_not_save_samples = True  # Prevents the default saving behavior

                    if self.debug:
                        print("[FacePop Debug] Starting final process.")

                    FacePopState.final_image_pass = True
                    self.toggle_scripts(False, FacePopState.deny_scripts_dict.get('final', []))
                    # Call the image processing function with p_copy
                    processed_copy = process_images(p_copy)
                    self.toggle_scripts(True, FacePopState.deny_scripts_dict.get('final', []))
                    FacePopState.final_image_pass = False

                    if self.debug:
                        print("[FacePop Debug] Exited final process.")

                    # Check if upscale_to_original is enabled and resize processed_copy.images[0] to the size of the original image
                    if FacePopState.upscale_to_original:
                        original_size = FacePopState.original_image.size  # Get the size (width, height) of the original image
                        processed_size = processed_copy.images[0].size  # Get the size of the processed image
                
                        if processed_size != original_size:
                            if self.debug:
                                print(f"[FacePop Debug] Resizing processed image from {processed_size} to match original image size: {original_size}")
                
                            # Resize processed_copy.images[0] to match the original image size using LANCZOS for high-quality scaling
                            processed_copy.images[0] = processed_copy.images[0].resize(original_size, Image.LANCZOS)
                        else:
                            if self.debug:
                                print(f"[FacePop Debug] Processed image is already the same size as the original image: {original_size}")


                    # Save the final composite image for this batch
                    final_output_path = os.path.join(self.output_path(), f"{FacePopState.timestamp}_final_composite.png")
                    processed_copy.images[0].save(final_output_path)
                    if self.debug:
                        print(f"[FacePop Debug] Saving final image: {final_output_path}")
                    FacePopState.preview_images.append(processed_copy.images[0])
                    if self.debug:
                        print(f"[FacePop Debug] Previews appended.")

            ### BATCH MODE
            elif num_images > 1:
                # Batch mode: process images[1:] and discard images[0] (junk) -- Update--- maybe junk unless grid is off
                start_idx = 0 if p.do_not_save_grid else 1
                for batch_index in range(start_idx, num_images):
                    if self.debug:
                        print(f"[FacePop Debug] Processing batch image #{batch_index}")

                    # Calculate which set of faces to use
                    face_index = (batch_index - 1) % len(FacePopState.processed_faces)
                    batch_faces = FacePopState.processed_faces[face_index]

                    if self.debug:
                        print(f"[FacePop Debug] Using face set {face_index} for batch image #{batch_index}")

                    # Get the original image and resize if necessary
                    final_image = FacePopState.original_image.copy()
                    final_image = np.array(final_image)

                    # Overlay the faces onto the current image
                    over_image = self.overlay_faces_on_image(final_image, batch_faces)
                    final_image_pil = Image.fromarray(over_image)

                    # Ensure combined_mask is handled
                    if FacePopState.combined_mask is not None:
                        # Ensure the mask is in grayscale mode ('L') for consistency
                        face_mask_pil = Image.fromarray(FacePopState.combined_mask)
                        if face_mask_pil.mode != 'L':
                            face_mask_pil = face_mask_pil.convert('L')
                            if self.debug:
                                print("[FacePop Debug] Batch pass: Converted face_mask_pil to grayscale ('L') mode")

                        if self.debug:
                            output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_mask_batch_{batch_index}.png")
                            face_mask_pil.save(output_path)
                            print(f"[FacePop Debug] Batch pass: Saving debug combined_mask for batch {batch_index}. {output_path}")
                    else:
                        if self.debug:
                            print(f"[FacePop Debug] Batch pass: No combined_mask found for batch {batch_index}, skipping mask assignment.")

                    # Combine the final mask
                    final_mask = self.combine_masks(p.image_mask, face_mask_pil)
                    final_mask = final_mask.convert('L')

                    # Resize the final mask (PIL) to proc_width and proc_height
                    if final_mask is not None:
                        final_mask = final_mask.resize((FacePopState.proc_width, FacePopState.proc_height), Image.LANCZOS)
                        if self.debug:
                            print(f"[FacePop Debug] Batch pass: Resized final_mask to ({FacePopState.proc_width}, {FacePopState.proc_height})")
                            output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_debug_mask_inpaint_batch_{batch_index}.png")
                            final_mask.save(output_path)
                            print(f"[FacePop Debug] Batch pass: Saving combined_mask with inpaint mask for batch {batch_index}. {output_path}")
    
                    # Resize the final image (PIL) to proc_width and proc_height
                    if final_image_pil is not None:
                        final_image_pil = final_image_pil.resize((FacePopState.proc_width, FacePopState.proc_height), Image.LANCZOS)
                        if self.debug:
                            print(f"[FacePop Debug] Batch pass: Resized final_image_pil to ({FacePopState.proc_width}, {FacePopState.proc_height})")
                            output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_face_overlays_batch_{batch_index}.png")
                            final_image_pil.save(output_path)
                            print(f"[FacePop Debug] Batch pass: Saving overlay image for batch {batch_index}. {output_path}")
    
                    # Perform final processing using a copy of the original parameters
                    if FacePopState.first_p is not None:
                        p_copy = copy.copy(FacePopState.first_p)
                        p_copy.image_mask = final_mask  # Assign the combined mask
                        p_copy.init_images = [final_image_pil]  # Assign the overlaid image
                        p_copy.width = FacePopState.proc_width
                        p_copy.height = FacePopState.proc_height
                        p_copy.batch_size = 1  # Set batch size to 1
                        p_copy.do_not_save_samples = True  # Prevents the default saving behavior

                        if self.debug:
                            print(f"[FacePop Debug] Starting final process for batch #{batch_index}.")
    
                        FacePopState.final_image_pass = True
                        self.toggle_scripts(False, FacePopState.deny_scripts_dict.get('final', []))
                        # Call the image processing function with p_copy
                        processed_copy = process_images(p_copy)
                        self.toggle_scripts(True, FacePopState.deny_scripts_dict.get('final', []))
                        FacePopState.final_image_pass = False

                        if self.debug:
                            print(f"[FacePop Debug] Exited final process for batch #{batch_index}.")

                        # Save the final composite image for this batch
                        FacePopState.batch_i = FacePopState.batch_i + 1

                        for idx in range(start_idx, FacePopState.batch_count + 1):
                            final_output_path = os.path.join(self.output_path(), f"{FacePopState.timestamp}_final_composite_batch_{FacePopState.batch_i}.png")
                            processed_copy.images[idx].save(final_output_path)
                            if self.debug:
                               print(f"[FacePop Debug] Saved final composite image for batch #{FacePopState.batch_i} at {final_output_path}")
                            FacePopState.preview_images.append(processed_copy.images[idx])
                            if self.debug:
                               print(f"[FacePop Debug] Previews appended.")


            # Reset static state variables after the final image is created
            if self.debug:
                print("[FacePop Debug] DONE!")

            FacePopState.reset()
            FacePopState.started = False
            processed.images = FacePopState.preview_images
            return processed

        if self.debug:
            print("[FacePop Debug] Exiting 'postprocess' method (non-final pass).")

    def detect_faces(self, image, p):
        """
        Detects faces in an input image using Mediapipe's Face Detection and optionally falls back to OpenCV's SSD face detection.
        Uses a mask image to prevent detecting the same face multiple times.
        """
        if self.debug:
            print(f"[FacePop Debug] Entered 'detect_faces' method.")
    
        # Initialize the mask image (black image)
        mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
        # Initialize the list to store all accepted faces
        all_faces = []
    
        # Define detection configurations
        detection_configs = []
    
        # Add initial detection methods
        detection_configs.append({'method': 'mediapipe', 'rotation': 0})
        detection_configs.append({'method': 'opencv', 'rotation': 0})
    
        # If aggressive detection is enabled, add rotated detection configurations
        if FacePopState.aggressive_detection:
            rotated_angles = [90, 180, 270]
            for angle in rotated_angles:
                detection_configs.append({'method': 'mediapipe', 'rotation': angle})
                detection_configs.append({'method': 'opencv', 'rotation': angle})
    
        # Loop through each detection configuration
        for config in detection_configs:
            method = config['method']
            rotation_angle = config['rotation']
    
            # Rotate the image and mask_image if rotation_angle != 0
            if rotation_angle != 0:
                rotated_image = self.rotate_image_cv(image, rotation_angle)
                rotated_mask_image = self.rotate_image_cv(mask_image, rotation_angle, is_mask=True)
            else:
                rotated_image = image.copy()
                rotated_mask_image = mask_image.copy()
    
            # Perform detection
            if method == 'mediapipe':
                # Convert image to RGB as Mediapipe requires RGB input
                if rotated_image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    
                # Initialize Mediapipe Face Detection if not already
                if FacePopState.face_detection is None:
                    FacePopState.initialize_face_detection(min_confidence=FacePopState.confidence_threshold)
    
                # Run Mediapipe Face Detection
                mediapipe_faces = FacePopState.face_detection.process(image_rgb)
    
                # Process Mediapipe detections
                if mediapipe_faces.detections:
                    for detection in mediapipe_faces.detections:
                        # Extract bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = rotated_image.shape
                        x_start = int(bboxC.xmin * iw)
                        y_start = int(bboxC.ymin * ih)
                        box_width = int(bboxC.width * iw)
                        box_height = int(bboxC.height * ih)
    
                        # Confidence score
                        confidence = detection.score[0]
    
                        # Check for overlap
                        bbox = (x_start, y_start, box_width, box_height)
                        if not self.is_overlap(bbox, rotated_mask_image):
                            # Accept detection
                            all_faces.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'rotation_angle': rotation_angle
                            })
                            # Update mask with ellipse
                            rotated_mask_image = self.update_mask(bbox, rotated_mask_image)
                        else:
                            if self.debug:
                                print(f"[FacePop Debug] Skipping overlapping face at ({x_start}, {y_start}, {box_width}, {box_height})")
    
                        # Stop detecting faces if the maximum number of faces is reached
                        if FacePopState.max_faces > 0 and len(all_faces) >= FacePopState.max_faces:
                            if self.debug:
                                print(f"[FacePop Debug] Maximum number of faces ({FacePopState.max_faces}) reached. Stopping detection.")
                            break
    
            elif method == 'opencv':
                # Detect faces using OpenCV
                opencv_faces = self.detect_faces_opencv(rotated_image, FacePopState.confidence_threshold)
                for face in opencv_faces:
                    x_start, y_start, box_width, box_height = face['bbox']
                    confidence = face['confidence']
    
                    # Check for overlap
                    bbox = (x_start, y_start, box_width, box_height)
                    if not self.is_overlap(bbox, rotated_mask_image):
                        # Accept detection
                        all_faces.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'rotation_angle': rotation_angle
                        })
                        # Update mask with ellipse
                        rotated_mask_image = self.update_mask(bbox, rotated_mask_image)
                    else:
                        if self.debug:
                            print(f"[FacePop Debug] Skipping overlapping face at ({x_start}, {y_start}, {box_width}, {box_height})")
    
                    # Stop detecting faces if the maximum number of faces is reached
                    if FacePopState.max_faces > 0 and len(all_faces) >= FacePopState.max_faces:
                        if self.debug:
                            print(f"[FacePop Debug] Maximum number of faces ({FacePopState.max_faces}) reached. Stopping detection.")
                        break
    
            # After processing, rotate mask_image back to original orientation if necessary
            if rotation_angle != 0:
                mask_image = self.rotate_image_cv(rotated_mask_image, -rotation_angle, is_mask=True)
            else:
                mask_image = rotated_mask_image.copy()
    
        # After all detections, map bounding boxes back to original orientation if they were rotated
        final_faces = []
        for face in all_faces:
            bbox = face['bbox']
            rotation_angle = face['rotation_angle']
            if rotation_angle != 0:
                # Rotate bbox back to original orientation
                rotated_bbox = self.rotate_bounding_box(bbox, -rotation_angle, image.shape[:2])
                face['bbox'] = rotated_bbox
                face['rotation_angle'] = 0  # Now in original orientation
            final_faces.append(face)
    
        # If debug is enabled, save the mask image to the debug folder
        if self.debug:
            # Ensure the debug path exists
            debug_output_path = os.path.join(self.debug_path(), f"{FacePopState.timestamp}_face_detection_mask.png")
            cv2.imwrite(debug_output_path, mask_image)
            print(f"[FacePop Debug] Saved face detection mask image at {debug_output_path}")
    
        if self.debug:
            print(f"[FacePop Debug] Exiting 'detect_faces' method with {len(final_faces)} faces detected.")
    
        return final_faces[:FacePopState.max_faces] if FacePopState.max_faces > 0 else final_faces


    def detect_faces_opencv(self, image, min_confidence):
        """
        Detects faces in an image using OpenCV's SSD face detection.

        :param image: The input image as a NumPy array (could be BGR, BGRA, or grayscale).
        :param min_confidence: The minimum confidence threshold for detections.
        :return: List of detected faces with bounding boxes and confidence scores.
        """
        if self.net is None:
            print("[FacePop Debug] OpenCV SSD model is not loaded.")
            return []

        try:
            # Ensure the image has 3 channels (BGR)
            if image.ndim == 2:
                # Grayscale image, convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if self.debug:
                    print("[FacePop Debug] Converted grayscale image to BGR.")
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    # BGRA to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    if self.debug:
                        print("[FacePop Debug] Converted BGRA image to BGR.")
                elif image.shape[2] == 3:
                    # Already BGR
                    pass
                else:
                    # Unexpected number of channels, convert to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    if self.debug:
                        print(f"[FacePop Debug] Unexpected number of channels ({image.shape[2]}), converted to BGR.")
            else:
                # Unexpected image format
                print(f"[FacePop Debug] Unexpected image dimensions ({image.ndim}), skipping detection.")
                return []

            # Prepare the blob for OpenCV's DNN
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            detected_faces = []
            ih, iw = image.shape[:2]

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < min_confidence:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
                (x_start, y_start, x_end, y_end) = box.astype("int")

                # Ensure the bounding boxes fall within the image dimensions
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(iw, x_end)
                y_end = min(ih, y_end)

                box_width = x_end - x_start
                box_height = y_end - y_start

                detected_faces.append({
                    'bbox': (x_start, y_start, box_width, box_height),
                    'confidence': confidence,
                    'rotation_angle': 0  # Placeholder, will be updated if rotated
                })

                if self.debug:
                    print(f"[FacePop Debug] OpenCV detected face at ({x_start}, {y_start}, {box_width}, {box_height}) with confidence {confidence:.2f}")

            return detected_faces

        except Exception as e:
            if self.debug:
                print(f"[FacePop Debug] Exception during OpenCV face detection: {e}")
            return []

    def rotate_image_cv(self, image, angle, is_mask=False):
        """
        Rotates an image or mask by the given angle using OpenCV.
        If is_mask is True, uses nearest neighbor interpolation to preserve mask values.
        """
        if angle == 0:
            return image.copy()

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Choose interpolation method
        if is_mask:
            # Use nearest neighbor interpolation for masks to preserve binary values
            interp_method = cv2.INTER_NEAREST
        else:
            interp_method = cv2.INTER_LINEAR

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=interp_method,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

        return rotated
        
    def is_overlap(self, bbox, mask_image, overlap_threshold=0.2):
        """
        Checks if the bounding box overlaps with any existing detections in the mask_image.
        Returns True if the overlap exceeds the overlap_threshold.
        """
        x_start, y_start, w, h = bbox
        x_end = x_start + w
        y_end = y_start + h
    
        # Ensure coordinates are within the mask bounds
        mask_height, mask_width = mask_image.shape[:2]
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(mask_width, x_end)
        y_end = min(mask_height, y_end)
    
        # Get the mask region that corresponds to the bbox
        mask_area = mask_image[y_start:y_end, x_start:x_end]
    
        # Calculate the percentage of the bbox area that is already white in mask_image
        total_pixels = mask_area.size
        overlapping_pixels = np.count_nonzero(mask_area)
    
        overlap_ratio = overlapping_pixels / total_pixels
    
        return overlap_ratio > overlap_threshold

    def update_mask(self, bbox, mask_image):
        """
        Updates the mask_image by drawing a white ellipse (oval) over the bounding box area.
        """
        x_start, y_start, w, h = bbox
        x_end = x_start + w
        y_end = y_start + h

        # Ensure coordinates are within the mask bounds
        mask_height, mask_width = mask_image.shape[:2]
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(mask_width, x_end)
        y_end = min(mask_height, y_end)

        # Calculate the center and axes lengths for the ellipse
        center_x = int((x_start + x_end) / 2)
        center_y = int((y_start + y_end) / 2)
        axes_length = (int(w / 2), int(h / 2))  # (major_axis_length, minor_axis_length)

        # Draw a filled white ellipse on the mask_image
        cv2.ellipse(mask_image, (center_x, center_y), axes_length,
                    angle=0, startAngle=0, endAngle=360,
                    color=255, thickness=-1)

        return mask_image

    def rotate_bounding_box(self, bbox, angle, image_shape):
        """
        Rotates a bounding box by the given angle around the center of the image.
        """
        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2

        (img_h, img_w) = image_shape[:2]
        center = (img_w / 2, img_h / 2)
    
        # Convert angle to radians and invert for rotation back to original
        angle_rad = np.deg2rad(angle)
    
        # Shift bbox center to origin
        x_shifted = x_center - center[0]
        y_shifted = y_center - center[1]
    
        # Apply rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_rotated = x_shifted * cos_a - y_shifted * sin_a
        y_rotated = x_shifted * sin_a + y_shifted * cos_a
    
        # Shift bbox center back
        x_rotated += center[0]
        y_rotated += center[1]
    
        # Calculate new top-left corner
        x_new = int(x_rotated - w / 2)
        y_new = int(y_rotated - h / 2)
    
        return (x_new, y_new, w, h)

    def non_max_suppression(self, detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []
    
        # Extract bounding boxes and confidence scores
        boxes = np.array([det['bbox'] for det in detections])
        confidences = np.array([det['confidence'] for det in detections])
    
        # Convert boxes to x1, y1, x2, y2 format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]  # x2 = x1 + width
        y2 = y1 + boxes[:, 3]  # y2 = y1 + height
    
        # Compute the area of the bounding boxes
        areas = (x2 - x1) * (y2 - y1)
        order = confidences.argsort()[::-1]  # Sort by confidence
    
        keep = []
    
        while order.size > 0:
            i = order[0]
            keep.append(detections[i])
    
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
    
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union
    
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
    
        return keep

    def overlay_faces_on_image(self, base_image, processed_faces):
        """
        Overlays processed faces onto the base image, resizing them if necessary, and applies an alpha mask for smooth blending.

        This method performs the following operations:
        - Iterates over each processed face along with its corresponding coordinates and status.
        - Resizes the processed face image to match the original face dimensions if required.
        - Converts images to appropriate formats (e.g., ensuring the presence of an alpha channel).
        - Applies an alpha mask to facilitate smooth blending of the processed face onto the base image.
        - Overlays the processed face onto the base image at the specified coordinates.
        - Returns the final composite image with all processed faces overlaid.

        :param base_image:
            The base image as a NumPy array (e.g., in BGR format) onto which processed faces will be overlaid.

        :param processed_faces:
            A list of tuples, each containing:
                - The processed face image as a PIL Image or NumPy array.
                - A tuple of coordinates (x, y, width, height) indicating where to place the face on the base image.
                - A status string indicating whether the face was 'processed' or 'skipped'.

        :return:
            The base image with all processed faces overlaid as a NumPy array.
        """
        if self.debug:
            print("[FacePop Debug] Entered 'overlay_faces_on_image' method")

        # Convert base_image to RGBA if it doesn't already have 4 channels
        if base_image.shape[2] == 3:  # RGB image
            # Add an alpha channel with full opacity (255)
            base_image = np.dstack([base_image, np.ones((base_image.shape[0], base_image.shape[1]), dtype=np.uint8) * 255])
            if self.debug:
                print("[FacePop Debug] Converted base image to RGBA")

        # Initialize the combined_mask if it's not already initialized
        if FacePopState.combined_mask is None:
            FacePopState.combined_mask = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.uint8)
            if self.debug:
                print("[FacePop Debug] Initialized combined_mask")

        # Iterate through each face to overlay them onto the base image
        for index, (processed_face_image, (x, y, w, h), status) in enumerate(processed_faces):
            if status == 'skipped':
                if self.debug:
                    print(f"[FacePop Debug] Skipping face #{index+1} as it was outside the inpainting region.")
                continue  # Skip this face if it was marked as 'skipped'

            # If the processed_face_image is a PIL Image, convert it to a NumPy array
            if not isinstance(processed_face_image, np.ndarray):
                processed_face_image = np.array(processed_face_image)
                if self.debug:
                    print(f"[FacePop Debug] Converted processed_face_image #{index+1} from PIL to NumPy array")

            # Ensure face image is RGBA to use its alpha transparency
            if processed_face_image.shape[2] == 3:  # If RGB, convert to RGBA
                processed_face_image = np.dstack([processed_face_image, np.ones((processed_face_image.shape[0], processed_face_image.shape[1]), dtype=np.uint8) * 255])
                if self.debug:
                    print(f"[FacePop Debug] Converted processed_face_image #{index+1} to RGBA")

            # Resize the face image to fit the corresponding base image area
            face_h, face_w = processed_face_image.shape[:2]
            base_h, base_w = base_image[y:y+h, x:x+w].shape[:2]

            if base_w > 0 and base_h > 0:
                if face_h != base_h or face_w != base_w:
                    if self.debug:
                        print(f"[FacePop Debug] Resizing mismatch for face #{index+1}: Face ({face_w}, {face_h}), Base ({base_w}, {base_h})")
                    processed_face_resized = cv2.resize(processed_face_image, (base_w, base_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    processed_face_resized = processed_face_image
                    if self.debug:
                        print(f"[FacePop Debug] No resizing needed for face #{index+1}")
            else:
                if self.debug:
                    print(f"[FacePop Debug] Invalid resize dimensions for face #{index+1}: base_w={base_w}, base_h={base_h}, skipping this face.")
                continue

            # **Step 1**: Extract and smooth the alpha channel of the face image
            alpha_face = processed_face_resized[..., 3].astype(np.float32) / 255.0  # Normalize to [0,1]

            if FacePopState.modnet_model is not None:
                # When MODNet is used, apply Gaussian blur for smoother transitions
                alpha_face_smoothed = cv2.GaussianBlur(alpha_face, (5, 5), 0, borderType=cv2.BORDER_CONSTANT)  # 5x5 kernel for smoothing
                alpha_face_smoothed = np.clip(alpha_face_smoothed, 0, 1)
                if self.debug:
                    print(f"[FacePop Debug] Applied Gaussian blur to alpha channel for face #{index+1}")
            else:
                # When MODNet is not used, skip Gaussian blur to prevent excessive transparency
                alpha_face_smoothed = alpha_face
                if self.debug:
                    print(f"[FacePop Debug] Skipped Gaussian blur for alpha channel for face #{index+1}")

            # **Step 2**: Generate the radial alpha mask with padding and feathering
            padding = int(w * (FacePopState.padding * .01))
            alpha_radial = self.generate_alpha_mask(base_w, base_h, padding, 10)

            # **Step 3**: Combine the face's alpha channel with the radial mask multiplicatively
            combined_alpha = alpha_face_smoothed * alpha_radial

            # **New Step 3.1**: Erode the combined alpha mask by 2 pixels where it meets transparent pixels
            # This helps eliminate seams by shrinking the mask edges slightly more than before

            # Convert combined_alpha to 8-bit for erosion
            alpha_uint8 = (combined_alpha * 255).astype(np.uint8)

            # Define a 3x3 kernel for erosion
            kernel = np.ones((3, 3), np.uint8)

            # Apply erosion twice to achieve 2-pixel erosion
            eroded_alpha_uint8 = cv2.erode(alpha_uint8, kernel, iterations=2)

            # Convert back to float32 [0,1]
            combined_alpha_eroded = eroded_alpha_uint8.astype(np.float32) / 255.0

            if self.debug:
                print(f"[FacePop Debug] Applied 2-pixel erosion to alpha mask for face #{index+1}")

            # **Step 4**: Apply a single Gaussian blur to the eroded_alpha for extra smoothing
            combined_alpha_blurred = cv2.GaussianBlur(combined_alpha_eroded, (7, 7), 0, borderType=cv2.BORDER_CONSTANT)  # 7x7 kernel for additional smoothing
            combined_alpha_blurred = np.clip(combined_alpha_blurred, 0, 1)

            # **Step 5**: Create a 3-channel alpha mask for blending
            combined_alpha_3ch = np.dstack([combined_alpha_blurred] * 3)
            if self.debug:
                print(f"[FacePop Debug] Applied additional Gaussian blur and created 3-channel alpha mask for face #{index+1}")

            # **Step 6**: Blend the processed face onto the base image using the alpha mask
            # Convert base_image and processed_face_resized to float32 for blending
            base_region = base_image[y:y+h, x:x+w, :3].astype(np.float32)
            face_region = processed_face_resized[..., :3].astype(np.float32)

            # Perform blending
            blended_region = (combined_alpha_3ch * face_region) + ((1 - combined_alpha_3ch) * base_region)
            blended_region = blended_region.astype(np.uint8)

            # Update the base image with the blended region
            base_image[y:y+h, x:x+w, :3] = blended_region

            # **New Step 7**: Update the combined_mask with the eroded and blurred alpha mask
            # Scale the alpha mask back to 0-255
            face_mask = (combined_alpha_blurred * 255).astype(np.uint8)

            # Place the face mask onto the combined_mask at the correct location
            FacePopState.combined_mask[y:y+h, x:x+w] = cv2.bitwise_or(FacePopState.combined_mask[y:y+h, x:x+w], face_mask)

            if FacePopState.modnet_model is not None:
                # **Optional Step**: Update the alpha channel of the base image
                # This ensures that areas where the face is overlaid have updated transparency
                base_image[y:y+h, x:x+w, 3] = np.clip(
                    base_image[y:y+h, x:x+w, 3].astype(np.float32) + (combined_alpha_blurred * 255),
                    0,
                    255
                ).astype(np.uint8)
                if self.debug:
                    print(f"[FacePop Debug] Updated alpha channel for face #{index+1}")
            else:
                # **When MODNet is not used**, ensure the alpha channel remains fully opaque
                base_image[y:y+h, x:x+w, 3] = 255
                if self.debug:
                    print(f"[FacePop Debug] Ensured full opacity for face #{index+1} without MODNet")

            if self.debug:
                print(f"[FacePop Debug] Overlayed and blended processed face #{index+1} onto the base image")

        if self.debug:
            print("[FacePop Debug] Exiting 'overlay_faces_on_image' method")

        return base_image



    def generate_alpha_mask(self, width, height, padding, feather_pixels=3):
        """
        Generate a radial gradient alpha mask that feathers from the padding boundary to the edges,
        incorporating a manual gradient box around the image borders to soften transitions.
    
        Parameters:
        - width: Width of the mask.
        - height: Height of the mask.
        - padding: Padding around the face to define the mask's inner boundary.
        - feather_pixels: Number of pixels to feather along the mask edges (default is 3).
    
        Returns:
        - alpha_mask: A 2D NumPy array representing the alpha mask.
        """
        # Create a meshgrid of coordinates (x, y)
        y_indices, x_indices = np.indices((height, width))
    
        # Calculate the center of the face image
        center_x, center_y = (width - 1) / 2, (height - 1) / 2
    
        # Compute the Euclidean distance of each pixel from the center
        distances_from_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    
        # Determine the maximum radius (distance from center to corner)
        max_radius = np.sqrt(center_x**2 + center_y**2)
    
        # Define inner and outer radii
        inner_radius = max_radius - padding
        outer_radius = max_radius
    
        # Handle cases where padding is larger than max_radius
        if inner_radius < 0:
            inner_radius = 0
    
        # Initialize alpha_mask with zeros
        alpha_mask = np.zeros_like(distances_from_center, dtype=np.float32)
    
        # Set alpha to 1 where distance <= inner_radius
        alpha_mask[distances_from_center <= inner_radius] = 1.0
    
        # For distances between inner_radius and outer_radius, compute alpha decreasing from 1 to 0
        mask_region = (distances_from_center > inner_radius) & (distances_from_center <= outer_radius)
        alpha_mask[mask_region] = 1.0 - ((distances_from_center[mask_region] - inner_radius) / (outer_radius - inner_radius))
    
        # Apply non-linear falloff for smoother transition
        alpha_mask[mask_region] = alpha_mask[mask_region] ** 2  # Adjust exponent as needed
    
        # Ensure alpha_mask values are between 0 and 1
        alpha_mask = np.clip(alpha_mask, 0, 1)
    
        # **Manual Feathering at the Image Edges**
        if feather_pixels > 0:
            # Initialize feathering mask with ones
            feather_mask = np.ones_like(alpha_mask, dtype=np.float32)
    
            # Compute the distance of each pixel from the nearest image border
            distance_to_edge = np.minimum.reduce([
                x_indices,  # Distance to left edge
                width - 1 - x_indices,  # Distance to right edge
                y_indices,  # Distance to top edge
                height - 1 - y_indices  # Distance to bottom edge
            ])
    
            # For pixels within the feathering region, adjust the feather_mask
            feather_region = distance_to_edge < feather_pixels
            feather_mask[feather_region] = distance_to_edge[feather_region] / feather_pixels
    
            # Combine the radial mask with the feathering mask
            alpha_mask = alpha_mask * feather_mask
    
            # Ensure alpha_mask values are between 0 and 1 after combination
            alpha_mask = np.clip(alpha_mask, 0, 1)
    
        return alpha_mask

    ### Applies an unsharp mask to the image for sharpening.
    ### Returns the sharpened image.
    def apply_unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Apply unsharp masking to enhance the sharpness of the image while preserving the alpha channel."""
        
        # Check if the image has an alpha channel (i.e., RGBA format)
        if image.shape[2] == 4:
            # Split the image into RGB and Alpha channels
            rgb_channels = image[:, :, :3]  # Extract RGB channels
            alpha_channel = image[:, :, 3]  # Extract the alpha channel
    
            # Apply the unsharp mask to the RGB channels only
            blurred = cv2.GaussianBlur(rgb_channels, kernel_size, sigma, borderType=cv2.BORDER_CONSTANT)
            sharpened_rgb = float(amount + 1) * rgb_channels - float(amount) * blurred
            sharpened_rgb = np.maximum(sharpened_rgb, np.zeros(sharpened_rgb.shape))
            sharpened_rgb = np.minimum(sharpened_rgb, 255 * np.ones(sharpened_rgb.shape))
            sharpened_rgb = sharpened_rgb.round().astype(np.uint8)
    
            if threshold > 0:
                low_contrast_mask = np.absolute(rgb_channels - blurred) < threshold
                np.copyto(sharpened_rgb, rgb_channels, where=low_contrast_mask)
    
            # Combine the sharpened RGB channels with the original alpha channel
            result_image = np.dstack((sharpened_rgb, alpha_channel))
    
        else:
            # If there is no alpha channel, proceed as normal
            blurred = cv2.GaussianBlur(image, kernel_size, sigma, borderType=cv2.BORDER_CONSTANT)
            sharpened = float(amount + 1) * image - float(amount) * blurred
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            sharpened = sharpened.round().astype(np.uint8)
    
            if threshold > 0:
                low_contrast_mask = np.absolute(image - blurred) < threshold
                np.copyto(sharpened, image, where=low_contrast_mask)
    
            result_image = sharpened
    
        return result_image

    ### Applies a bilateral filter to the image to reduce noise while preserving edges.
    ### Returns the filtered image.
    def apply_bilateral_filter(self, image, diameter=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filtering to reduce noise while preserving edges, without affecting the alpha channel."""
        
        # Check if the image has an alpha channel (RGBA format)
        if image.shape[2] == 4:
            # Split the image into RGB and Alpha channels
            rgb_channels = image[:, :, :3]  # Extract RGB channels
            alpha_channel = image[:, :, 3]  # Extract Alpha channel
    
            # Apply bilateral filter to the RGB channels only
            filtered_rgb = cv2.bilateralFilter(rgb_channels, diameter, sigma_color, sigma_space)
    
            # Combine the filtered RGB channels with the original alpha channel
            result_image = np.dstack((filtered_rgb, alpha_channel))
    
        else:
            # If there's no alpha channel, apply bilateral filtering directly
            result_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
        return result_image

    ### Applies a median filter to the image to reduce noise.
    ### Returns the filtered image.
    def apply_median_filter(self, image, kernel_size=3):
        """Apply median filtering to reduce noise while preserving the alpha channel."""
        
        # Check if the image has an alpha channel (RGBA format)
        if image.shape[2] == 4:
            # Split the image into RGB and Alpha channels
            rgb_channels = image[:, :, :3]  # Extract RGB channels
            alpha_channel = image[:, :, 3]  # Extract Alpha channel
    
            # Apply median filter to the RGB channels only
            filtered_rgb = cv2.medianBlur(rgb_channels, kernel_size)
    
            # Combine the filtered RGB channels with the original alpha channel
            result_image = np.dstack((filtered_rgb, alpha_channel))
    
        else:
            # If there's no alpha channel, apply median filtering directly
            result_image = cv2.medianBlur(image, kernel_size)
    
        return result_image

    ### A hybrid approach that first applies a bilateral filter and then sharpens the image using an unsharp mask.
    ### Returns the enhanced image.
    def enhance_image(self, image):
        """Apply a hybrid approach to reduce noise and enhance sharpness while preserving the alpha channel."""
        
        # Check if the image has an alpha channel (RGBA format)
        if image.shape[2] == 4:
            # Split the image into RGB and Alpha channels
            rgb_channels = image[:, :, :3]  # Extract RGB channels
            alpha_channel = image[:, :, 3]  # Extract Alpha channel

            # Step 1: Reduce noise with bilateral filtering on RGB channels
            denoised_rgb = self.apply_bilateral_filter(rgb_channels)

            # Step 2: Apply unsharp masking to enhance sharpness on the denoised RGB channels
            sharpened_rgb = self.apply_unsharp_mask(denoised_rgb)

            # Combine the sharpened RGB channels with the original alpha channel
            result_image = np.dstack((sharpened_rgb, alpha_channel))
    
        else:
            # If there's no alpha channel, apply both filters directly
            denoised_image = self.apply_bilateral_filter(image)
            result_image = self.apply_unsharp_mask(denoised_image)
    
        return result_image

    def blend_with_background(self, original_face_rgba, rotated_face_rgba):
        """Blends the rotated face with the original face copy to eliminate seams."""

        # Ensure both images have an alpha channel (RGBA)
        def ensure_alpha_channel(image):
            if image.shape[2] == 3:  # If there are only RGB channels
                alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255  # Fully opaque alpha
                image = np.concatenate((image, alpha_channel), axis=2)  # Add alpha channel
            return image

        # Ensure original and rotated faces have alpha channels
        original_face_rgba = ensure_alpha_channel(original_face_rgba)
        rotated_face_rgba = ensure_alpha_channel(rotated_face_rgba)

        # Ensure both images have the same size by resizing
        if original_face_rgba.shape != rotated_face_rgba.shape:
            new_size = (min(original_face_rgba.shape[1], rotated_face_rgba.shape[1]),  # width
                        min(original_face_rgba.shape[0], rotated_face_rgba.shape[0]))  # height

            original_face_rgba = cv2.resize(original_face_rgba, new_size, interpolation=cv2.INTER_LANCZOS4)
            rotated_face_rgba = cv2.resize(rotated_face_rgba, new_size, interpolation=cv2.INTER_LANCZOS4)

        # Create an alpha mask for blending based on non-black pixels in rotated_face_rgba (ignore alpha)
        mask = np.all(rotated_face_rgba[..., :3] != [0, 0, 0], axis=-1)

        # Create an empty array for the blended face with an alpha channel
        blended_face = np.zeros_like(rotated_face_rgba)

        # Blend the original face and rotated face based on the mask (only RGB channels)
        for c in range(3):  # Loop over RGB channels
            blended_face[..., c] = np.where(mask, rotated_face_rgba[..., c], original_face_rgba[..., c])

        # If you don't want to blend the alpha channel, keep the alpha from the original image
        blended_face[..., 3] = original_face_rgba[..., 3]  # Retain the alpha from the original image

        return blended_face


    def calculate_rotation_angle(self, landmarks):
        """
        Calculates the rotation angle of the face based on eye positions and nose.
        Mediapipe provides 468 landmarks; we use key points around the eyes and nose.
        """
        # Check if the necessary landmarks exist first (33 for left eye, 263 for right eye, 1 for nose)
        if len(landmarks) < 264:
            if self.debug:
               print("[FacePop Debug] Not enough landmarks detected for proper alignment.")
            return 0  # Skip rotation if landmarks are insufficient

        # Extract landmarks for the left eye, right eye, and nose tip
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]

        # Calculate the angle of the line between the eyes
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]

        # Calculate the eye rotation angle in degrees
        eye_angle = math.degrees(math.atan2(delta_y, delta_x))

        # Apply a threshold to avoid small unnecessary rotations
        if abs(eye_angle) < FacePopState.rotation_angle_threshold:  # Use the dynamic threshold
            if self.debug:
                print(f"[FacePop Debug] Skipping rotation: eye_angle {eye_angle:.2f} below threshold {FacePopState.rotation_angle_threshold:.2f}")
            return 0  # No need to rotate

        if self.debug:
            print(f"[FacePop Debug] Eye angle detected: {eye_angle:.2f} degrees")
        return eye_angle  # Return the calculated angle

    def rotate_image(self, image, angle, use_alpha=False):
        """
        Rotates an image (NumPy array or PIL Image) by the given angle using Pillow.
        If the input is a NumPy array, it will be converted to a PIL Image first.
    
        Parameters:
        - image: The input image (NumPy array or PIL Image).
        - angle: The angle by which to rotate the image.
        - use_alpha: If True, the image will maintain transparency (RGBA).
                     If False, the image will be converted to RGB with no alpha.
        
        Returns:
        - The rotated image with transparent (alpha) borders if use_alpha is True,
          or with a white background if use_alpha is False.
        """
        # Check if the image is a NumPy array and convert it to a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if use_alpha:
            # Ensure the image is in RGBA mode for transparency support
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            # Rotate with transparent background (0, 0, 0, 0) for alpha support
            rotated_image = image.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))  # Transparent fill
        else:
            # Ensure the image is in RGB mode for non-alpha support
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Rotate with white background (255, 255, 255) for no transparency
            rotated_image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))  # White fill

        # Convert the rotated image back to a NumPy array and return
        return np.array(rotated_image)

    def detect_landmarks(self, face_image):
        """
        Detects facial landmarks using MediaPipe's Face Mesh.
        Attempts multiple rotations if landmarks are not initially detected.
        
        :param face_image: The cropped face image as a NumPy array.
        :return: List of (x, y) tuples representing landmarks, and the eyes/mouth mask
        """
        confidence = max(0.90, FacePopState.confidence_threshold)
    
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence
        )
    
        # Convert BGR to RGB as required by MediaPipe
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
        # Detect landmarks using MediaPipe Face Mesh
        results = mp_face_mesh.process(face_rgb)
    
        landmarks = []

        if results.multi_face_landmarks:
            landmarks = self.extract_landmarks(results, face_image)
        else:
            landmarks = self.aggressive_landmark_detection(face_image)
    
        if landmarks:
            # Generate masks for eyes and mouth
            eyes_mouth_mask = self.create_eyes_mouth_mask(landmarks, face_image.shape[:2])
            return landmarks, eyes_mouth_mask
    
        return [], None

    def aggressive_landmark_detection(self, face_image):
        """
        Aggressively detects landmarks by rotating the image in 5-degree increments.
        Attempts multiple confidence levels as well.
        
        :param face_image: The cropped face image as a NumPy array.
        :return: List of (x, y) tuples representing landmarks or an empty list if not found.
        """
        img_height, img_width = face_image.shape[:2]
        
        # Convert to PIL RGB image
        face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Define rotation angles
        rotation_steps = list(range(5, 360, 5))  # 5-degree increments

        # Try different confidence levels as we rotate the image
        confidence = 0.99
        min_confidence = FacePopState.confidence_threshold
    
        while confidence >= min_confidence:
            # Initialize MediaPipe Face Mesh with the current confidence level
            mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=confidence)
            for angle in rotation_steps:
                rotated_pil = face_pil.rotate(angle, expand=False)
                rotated_image = np.array(rotated_pil)

                # Detect landmarks on the rotated image
                rotated_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
                results = mp_face_mesh.process(rotated_rgb)

                if results.multi_face_landmarks:
                    detected_landmarks = self.extract_landmarks(results, rotated_image)

                    # Correct landmarks back to original orientation
                    corrected_landmarks = self.rotate_landmarks_back(detected_landmarks, img_width, img_height, angle)
                    if self.debug:
                       print(f"[FacePop Debug] Landmarks detected at confidence={confidence:.2f}, angle={angle} degrees.")
                    return corrected_landmarks

            # Lower confidence by 0.1 for the next set of rotations
            confidence -= 0.02

        # If no landmarks detected after all attempts
        if self.debug:
            print("[FacePop Debug] No landmarks detected after aggressive detection attempts.")
        return []

    def extract_landmarks(self, results, face_image):
        """
        Extracts landmarks from MediaPipe's results and converts them to (x, y) format.
    
        :param results: MediaPipe's face landmark detection results.
        :param face_image: The face image to map the landmarks onto.
        :return: List of (x, y) tuples representing landmarks.
        """
        landmarks = []
        img_height, img_width = face_image.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                landmarks.append((x, y))
        
        return landmarks

    def rotate_landmarks_back(self, landmarks, img_width, img_height, angle):
        """
        Rotates landmarks back to the original image orientation.

        :param landmarks: List of (x, y) tuples detected on the rotated image.
        :param img_width: Width of the original image.
        :param img_height: Height of the original image.
        :param angle: Angle by which the image was rotated clockwise.
        :return: List of (x, y) tuples rotated back to original orientation.
        """
        if angle == 0:
            return landmarks

        angle_rad = np.deg2rad(angle)  # Correct: No need for negative angle
        center_x, center_y = img_width / 2.0, img_height / 2.0
        rotated_landmarks = []
    
        for (x, y) in landmarks:
            # Shift to origin
            x_shifted = x - center_x
            y_shifted = y - center_y
    
            # Apply reverse rotation
            new_x = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
            new_y = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
    
            # Shift back to original position
            new_x += center_x
            new_y += center_y
    
            # Clamp values to image boundaries
            new_x = max(0, min(img_width - 1, int(new_x)))
            new_y = max(0, min(img_height - 1, int(new_y)))
    
            rotated_landmarks.append((new_x, new_y))
    
        return rotated_landmarks
        
    def flip_landmarks(self, landmarks, img_height):
        """
        Flips the landmarks vertically to account for vertical flipping of the image.
        
        :param landmarks: List of (x, y) tuples.
        :param img_height: Height of the image.
        :return: List of flipped (x, y) tuples.
        """
        return [(x, img_height - y) for (x, y) in landmarks]

    def rotate_image_pil(self, image, angle, flip=False):
        """
        Rotates and optionally flips the image using PIL, ensuring consistent size and color space.
        
        :param image: PIL Image.
        :param angle: Angle in degrees.
        :param flip: Boolean indicating whether to flip the image vertically.
        :return: Rotated (and possibly flipped) NumPy array in BGR format.
        """
        if flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        rotated_image = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        rotated_np = np.array(rotated_image)
        rotated_bgr = cv2.cvtColor(rotated_np, cv2.COLOR_RGB2BGR)
        return rotated_bgr

    def is_face_in_inpaint_region(self, face_coords, mask):
        """
        Check if the face's bounding box is within the inpainting mask region.

        face_coords: Tuple (x, y, w, h) representing the bounding box of the face.
        mask: The binary mask of the inpaint region (1 for inpaint region, 0 for outside).

        Returns True if at least 20% of the face is inside the inpaint region,
        and returns the cropped mask if needed.
        """
        x_start, y_start, face_width, face_height = face_coords
        x_end = x_start + face_width
        y_end = y_start + face_height
    
        # Ensure coordinates are within the mask bounds
        mask_height, mask_width = mask.shape[:2]
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(mask_width, x_end)
        y_end = min(mask_height, y_end)
    
        # Get the mask region that corresponds to the face's bounding box
        face_mask_region = mask[y_start:y_end, x_start:x_end]
    
        # Calculate the percentage of the face's area that overlaps with the inpaint region
        total_pixels = face_mask_region.size
        inpaint_pixels = np.sum(face_mask_region > 0)  # Count white (1) pixels in the face region

        # Define a threshold: Include the face if 20% or more is inside the inpaint region
        overlap_threshold = 0.2
        overlap_ratio = inpaint_pixels / total_pixels
    
        # Return True if the overlap is >= threshold, otherwise False
        return overlap_ratio >= overlap_threshold, face_mask_region if overlap_ratio >= overlap_threshold else None

    def remove_background_with_modnet(self, image):
        """
        Removes the background from the provided image using the MODNet model, resulting in an image
        with a transparent or specified background.

        This method performs the following operations:
        - Converts the input image to the appropriate format for MODNet processing.
        - Preprocesses the image by resizing and normalizing it.
        - Runs the MODNet model to generate a matte (alpha channel) representing the foreground.
        - Resizes the matte to match the original image dimensions.
        - Combines the original image with the matte to produce the final image with the background removed.

        :param image:
            The input image from which the background is to be removed. This can be a PIL Image or a NumPy array
            in RGB or BGR format.

        :return:
            A PIL Image with the background removed. The image retains the original size and format with an alpha
            channel applied for transparency.
        """
        if FacePopState.modnet_model is None:
            raise RuntimeError("[FacePop Debug] MODNet model has not been loaded")
    
        # Transform image for MODNet
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        # Convert image to NumPy and adjust channels
        im = np.asarray(image)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)  # Convert grayscale to RGB
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]  # Convert RGBA to RGB
    
        # Convert image to tensor
        image_tensor = im_transform(Image.fromarray(im)).unsqueeze(0)
    
        # Resize image for MODNet
        im_h, im_w = image.size[1], image.size[0]
        im_rh, im_rw = self.adjust_image_size(im_h, im_w)
        image_tensor = nn.functional.interpolate(image_tensor, size=(im_rh, im_rw), mode='area')

        # Perform inference
        with torch.no_grad():
            _, _, matte = FacePopState.modnet_model(image_tensor, inference=True)
    
        # Resize matte to original image size and convert to numpy
        matte_resized = nn.functional.interpolate(matte, size=(im_h, im_w), mode='area')
        matte_resized = matte_resized[0][0].cpu().numpy()
    
        # Convert matte to an alpha channel (0-255)
        alpha_channel = (matte_resized * 255).astype('uint8')

        # Convert the original image to RGBA and combine with the new alpha channel
        image_rgba = image.convert("RGBA")
        r, g, b, _ = image_rgba.split()
        image_with_alpha = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_channel)))
    
        return image_with_alpha

    def adjust_image_size(self, im_h, im_w, ref_size=512):
        """Helper function to adjust image size to MODNet's requirements"""
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        return im_rh, im_rw

    def combine_masks(self, p_image_mask, face_mask_pil):
        """
        Combines the primary image mask with the face-specific mask to create a unified mask for processing.
    
        This method performs the following operations:
        - Converts the face-specific mask from a PIL Image to a NumPy array.
        - Ensures both masks are in the same mode and size.
        - Combines the masks by using a bitwise OR operation, effectively merging the regions to be processed.
        - Returns the combined mask as a PIL Image for consistency with downstream processing steps.
    
        :param p_image_mask:
            The primary image mask from the processing pipeline. This is typically a grayscale image where
            white regions indicate areas to be processed (e.g., inpainting regions).
        
        :param face_mask_pil:
            The face-specific mask as a PIL Image. This mask highlights the regions around detected faces
            that require special processing, such as enhancement or background removal.
    
        :return:
            A combined PIL Image mask that merges the primary image mask and the face-specific mask.
            The resulting mask ensures that both general processing areas and face-specific areas are
            appropriately handled in subsequent processing steps.
        """
        if p_image_mask is None:
            # If there's no primary image mask, invert the face mask and return it
            face_mask_array = np.array(face_mask_pil.convert('L'))
            face_mask_inverted = cv2.bitwise_not(face_mask_array)
            combined_mask = Image.fromarray(face_mask_inverted)
        else:
            # Ensure both masks are the same size by resizing face_mask_pil to match p_image_mask
            if face_mask_pil.size != p_image_mask.size:
                face_mask_pil = face_mask_pil.resize(p_image_mask.size, Image.NEAREST)
    
            # Convert both masks to grayscale NumPy arrays
            mask1 = np.array(p_image_mask.convert('L'))  # Inpaint area (white is inpaint)
            mask2 = np.array(face_mask_pil.convert('L'))  # Face area (black should override)
    
            # Invert the face mask so black becomes white (face) and white becomes black (non-face)
            mask2_inv = cv2.bitwise_not(mask2)
    
            # Combine using bitwise AND to give precedence to the black areas in face_mask_pil
            combined = cv2.bitwise_and(mask1, mask2_inv)
    
            # Convert back to PIL Image
            combined_mask = Image.fromarray(combined)
    
        return combined_mask

    def toggle_scripts(self, able: bool, scripts_list):
        """
        Enable or disable the specified scripts dynamically.
        """
        # Store the original scripts if FacePopState.scripts is empty
        if len(FacePopState.scripts) == 0:
            FacePopState.scripts = copy.copy(scripts.scripts_img2img.alwayson_scripts)
        
        if able:
            # Restore the original scripts
            scripts.scripts_img2img.alwayson_scripts = copy.copy(FacePopState.scripts)
            if self.debug:
                print("[FacePop Debug] Scripts have been restored.")
        else:
            # Disable the specified scripts by removing them from the list
            for script_name in scripts_list:
                for i, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
                    if hasattr(script, 'title') and callable(getattr(script, 'title')):
                        if script.title() == script_name:
                            scripts.scripts_img2img.alwayson_scripts.pop(i)
                            if self.debug:
                                print(f"[FacePop Debug] {script_name} script has been disabled.")
                            #break

    def load_scripts_from_file(self, file_path):
        """
        Load scripts from an INI-like file and return a dictionary with sections and script names.
        """
        scripts_dict = {}
        current_section = None
    
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith(';'):  # Skip empty lines and comments
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        scripts_dict[current_section] = []
                    elif current_section:
                        scripts_dict[current_section].append(line)
        else:
            print(f"[FacePop Debug] File {file_path} does not exist.")

        return scripts_dict
        
    def create_eyes_mouth_mask(self, landmarks, image_shape):
        # Indices for facial features
        left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        right_eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
        # Combined upper and lower outer lip indices
        mouth_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,       # Upper outer lip (from left to right)
            375, 321, 405, 314, 17, 84, 181, 91, 146               # Lower outer lip (from right to left)
        ]
        
        # Combined upper and lower inner lip indices
        mouth_inner_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,      # Upper inner lip (from left to right)
            324, 318, 402, 317, 14, 87, 178, 88, 95                # Lower inner lip (from right to left)
        ]

        # Extract points for each feature
        left_eye_points = [landmarks[i] for i in left_eye_indices]
        right_eye_points = [landmarks[i] for i in right_eye_indices]
        mouth_points = [landmarks[i] for i in mouth_indices]
    
        # Create an empty mask
        img_height, img_width = image_shape
        mask = np.ones((img_height, img_width), dtype=np.uint8) * 255  # All white mask
    
        # Convert points to NumPy arrays
        left_eye_contour = np.array(left_eye_points, dtype=np.int32)
        right_eye_contour = np.array(right_eye_points, dtype=np.int32)
        mouth_contour = np.array(mouth_points, dtype=np.int32)
    
        # Fill the polygons on the mask
        cv2.fillPoly(mask, [left_eye_contour], 0)
        cv2.fillPoly(mask, [right_eye_contour], 0)
        cv2.fillPoly(mask, [mouth_contour], 0)
    
        return mask

    def postprocess_image(self, p, *args):
        if not FacePopState.enabled:
           if self.debug:
               print(f"[FacePop Debug] Skipping postprocess_image().")
           return  # Skip this stage if you've already processed everything

    def get_output_path(self, output_path, base_output_dir):
        """
        Returns the output path for saving images. Replaces all instances of "[date]" with the current date
        in "YYYY-MM-DD" format. If "[date]" is not present, uses the provided path as is.

        :param output_path: The user-specified output path (may contain "[date]").
        :param base_output_dir: The base directory where samples are stored (e.g., p.outpath_samples).
        :return: Full output path to save images with "[date]" replaced by the current date.
        """
        # Define the placeholder to look for
        placeholder = "[date]"

        # Check if the placeholder exists in the output_path
        if placeholder in output_path:
            # Get the current date in "YYYY-MM-DD" format
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            # Replace all instances of "[date]" with the current date
            formatted_output_path = output_path.replace(placeholder, current_date)
            if self.debug:
                print(f"[FacePop Debug] Replaced '{placeholder}' with '{current_date}' in output path.")
        else:
            # Use the user-provided custom output path as is
            formatted_output_path = output_path
            if self.debug:
                print(f"[FacePop Debug] No placeholder '{placeholder}' found in output path. Using provided path as is.")

        # Generate the full output path by joining with the base directory
        full_output_path = os.path.join(base_output_dir, formatted_output_path.strip("/\\"))

        # Create the directory if it doesn't exist
        try:
            os.makedirs(full_output_path, exist_ok=True)
            if self.debug:
                print(f"[FacePop Debug] Created or verified existence of output directory: {full_output_path}")
        except Exception as e:
            print(f"[FacePop Error] Failed to create output directory '{full_output_path}': {e}")
            # Optionally, you can set a fallback path or handle the error as needed
            full_output_path = base_output_dir  # Fallback to base directory

        return full_output_path

    def output_path(self):
        os.makedirs(FacePopState.output_path, exist_ok=True)  # Create the debug directory if it doesn't exist
        return FacePopState.output_path

    def debug_path(self):
        if self.debug:
            os.makedirs(FacePopState.debug_path, exist_ok=True)  # Create the debug directory if it doesn't exist
        return FacePopState.debug_path

### EOF