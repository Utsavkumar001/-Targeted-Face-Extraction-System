class TargetedFaceExtractor:
    def __init__(self, input_folder, output_folder, reference_folder,
                 match_threshold=0.55,detection_size=320,     # NEW: Smaller = faster
                 resize_images=True):
        """
        Extract only specific people from images

        Args:
            input_folder: Folder with all your images
            output_folder: Where to save results
            reference_folder: Folder with reference faces (name = filename)
            match_threshold: How strict matching should be (0.5-0.6 recommended)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.reference_folder = reference_folder
        self.match_threshold = match_threshold

        print("🚀 Initializing Face Recognition...")
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ Model loaded")

        self.target_people = {}  # {name: embedding}
        self.matches = defaultdict(list)  # {name: [image_paths]}

    def load_reference_faces(self):
        """Load reference faces and generate embeddings"""
        print(f"\n📸 Loading reference faces from: {self.reference_folder}")

        if not os.path.exists(self.reference_folder):
            raise ValueError(f"Reference folder not found: {self.reference_folder}")

        ref_files = [f for f in os.listdir(self.reference_folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(ref_files) == 0:
            raise ValueError("No reference images found! Add images named as: Alice.jpg, Bob.jpg, etc.")

        print(f"Found {len(ref_files)} reference images")

        for ref_file in ref_files:
            # Extract name from filename (without extension)
            person_name = os.path.splitext(ref_file)[0]
            ref_path = os.path.join(self.reference_folder, ref_file)

            # Read image and detect face
            img = cv2.imread(ref_path)
            if img is None:
                print(f"  ⚠️  Could not read: {ref_file}")
                continue

            faces = self.app.get(img)

            if len(faces) == 0:
                print(f"  ⚠️  No face detected in: {ref_file}")
                continue

            if len(faces) > 1:
                print(f"  ⚠️  Multiple faces in {ref_file}, using largest face")
                # Use the largest face
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

            # Store embedding
            self.target_people[person_name] = faces[0].embedding
            print(f"  ✓ Loaded: {person_name}")

        if len(self.target_people) == 0:
            raise ValueError("No valid reference faces loaded!")

        print(f"\n✅ Successfully loaded {len(self.target_people)} target people:")
        for name in self.target_people.keys():
            print(f"  - {name}")

    def search_in_images(self):
        """Search for target people in all images"""
        image_files = self._get_image_files()
        print(f"\n🔍 Searching for target people in {len(image_files)} images...")

        total_faces_found = 0
        images_with_matches = 0

        for img_path in tqdm(image_files, desc="Scanning images"):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Detect all faces in image
                faces = self.app.get(img)

                if len(faces) == 0:
                    continue

                # Check each face against target people
                image_has_match = False

                for face in faces:
                    face_embedding = face.embedding

                    # Compare with each target person
                    best_match_name = None
                    best_similarity = -1

                    for person_name, ref_embedding in self.target_people.items():
                        similarity = cosine_similarity(
                            face_embedding.reshape(1, -1),
                            ref_embedding.reshape(1, -1)
                        )[0][0]

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_name = person_name

                    # If similarity is above threshold, it's a match
                    if best_similarity >= (1 - self.match_threshold):
                        self.matches[best_match_name].append({
                            'image_path': img_path,
                            'similarity': best_similarity,
                            'bbox': face.bbox
                        })
                        total_faces_found += 1
                        image_has_match = True

                if image_has_match:
                    images_with_matches += 1

            except Exception as e:
                print(f"\n⚠️  Error processing {os.path.basename(img_path)}: {e}")

        print(f"\n📊 Search Results:")
        print(f"  Images scanned: {len(image_files)}")
        print(f"  Images with matches: {images_with_matches}")
        print(f"  Total faces found: {total_faces_found}")

        print(f"\n👥 Matches per person:")
        for person_name in self.target_people.keys():
            count = len(self.matches[person_name])
            print(f"  {person_name}: {count} images")

    def organize_results(self):
        """Create folders only for target people with their images"""
        print(f"\n📁 Organizing results...")

        if not any(self.matches.values()):
            print("❌ No matches found for any target person!")
            return

        for person_name, match_list in self.matches.items():
            if len(match_list) == 0:
                continue

            # Create person folder
            person_folder = os.path.join(self.output_folder, person_name)
            os.makedirs(person_folder, exist_ok=True)

            # Copy images (avoid duplicates)
            copied_images = set()

            for match in match_list:
                img_path = match['image_path']

                if img_path in copied_images:
                    continue

                # Copy image
                img_name = os.path.basename(img_path)
                dest_path = os.path.join(person_folder, img_name)

                # Handle duplicate filenames
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(img_name)
                    dest_path = os.path.join(person_folder, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.copy2(img_path, dest_path)
                copied_images.add(img_path)

            print(f"  ✓ {person_name}: {len(copied_images)} images copied")

        print(f"\n✅ Results saved to: {self.output_folder}")

    def _get_image_files(self):
        """Get all image files from input folder"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    image_files.append(os.path.join(root, file))

        return image_files

    def run(self):
        """Execute the complete pipeline"""
        print("=" * 60)
        print("🎯 TARGETED FACE EXTRACTION")
        print("=" * 60)

        # Step 1: Load reference faces
        self.load_reference_faces()

        # Step 2: Search for them in images
        self.search_in_images()

        # Step 3: Organize results
        self.organize_results()

        print("\n" + "=" * 60)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 60)

# ============================================
# DEPLOYMENT: Gradio Web Interface
# ============================================
# Run this cell AFTER your TargetedFaceExtractor class is defined
# No need to modify existing code!

pip install -q gradio

import gradio as gr
import tempfile
import zipfile
from pathlib import Path

class GradioWrapper:
    """Wrapper to make TargetedFaceExtractor work with Gradio"""
    
    def __init__(self):
        self.extractor = None
    
    def process_files(self, reference_files, input_files, threshold):
        """
        Process uploaded files using the existing TargetedFaceExtractor
        
        Args:
            reference_files: Gradio uploaded reference images
            input_files: Gradio uploaded search images
            threshold: Match threshold value
        
        Returns:
            status_message: Text summary of results
            zip_file: ZIP file path for download
        """
        
        # Validation
        if not reference_files or len(reference_files) == 0:
            return "❌ Please upload at least one reference face image!", None
        
        if not input_files or len(input_files) == 0:
            return "❌ Please upload images to search through!", None
        
        # Create temporary directories
        temp_base = tempfile.mkdtemp()
        ref_folder = os.path.join(temp_base, "reference_faces")
        input_folder = os.path.join(temp_base, "input_images")
        output_folder = os.path.join(temp_base, "output")
        
        os.makedirs(ref_folder, exist_ok=True)
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            status = "🚀 Starting face extraction...\n\n"
            
            # Step 1: Save reference files
            status += "📸 Processing reference faces:\n"
            for ref_file in reference_files:
                # Get filename from uploaded file
                if hasattr(ref_file, 'name'):
                    filename = os.path.basename(ref_file.name)
                    src_path = ref_file.name
                else:
                    filename = f"person_{len(os.listdir(ref_folder))}.jpg"
                    src_path = ref_file
                
                dest_path = os.path.join(ref_folder, filename)
                shutil.copy2(src_path, dest_path)
                status += f"  ✓ {Path(filename).stem}\n"
            
            # Step 2: Save input files
            status += f"\n🖼️  Processing {len(input_files)} input images...\n"
            for idx, input_file in enumerate(input_files):
                if hasattr(input_file, 'name'):
                    filename = os.path.basename(input_file.name)
                    src_path = input_file.name
                else:
                    filename = f"image_{idx}.jpg"
                    src_path = input_file
                
                dest_path = os.path.join(input_folder, filename)
                shutil.copy2(src_path, dest_path)
            
            status += "✓ All images uploaded\n\n"
            
            # Step 3: Run the extractor (YOUR EXISTING CODE!)
            status += "="*60 + "\n"
            status += "🎯 RUNNING FACE EXTRACTION\n"
            status += "="*60 + "\n\n"
            
            # Create extractor instance with your existing class
            self.extractor = TargetedFaceExtractor(
                input_folder=input_folder,
                output_folder=output_folder,
                reference_folder=ref_folder,
                match_threshold=threshold
            )
            
            # Run the extraction
            self.extractor.run()
            
            # Step 4: Collect results
            status += "\n" + "="*60 + "\n"
            status += "📊 RESULTS SUMMARY\n"
            status += "="*60 + "\n\n"
            
            # Count results per person
            total_images = 0
            for person_folder in os.listdir(output_folder):
                person_path = os.path.join(output_folder, person_folder)
                if os.path.isdir(person_path):
                    image_count = len([f for f in os.listdir(person_path)
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    status += f"📁 {person_folder}: {image_count} images\n"
                    total_images += image_count
            
            if total_images == 0:
                status += "\n⚠️  No matches found! Try:\n"
                status += "  • Increasing the threshold (make it more lenient)\n"
                status += "  • Using clearer reference photos\n"
                status += "  • Checking if the people are actually in the images\n"
                return status, None
            
            status += f"\n✅ Total: {total_images} images organized\n"
            
            # Step 5: Create ZIP file for download
            zip_path = os.path.join(temp_base, "face_extraction_results.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_folder)
                        zipf.write(file_path, arcname)
            
            status += f"\n📦 Results packaged as ZIP file\n"
            status += "⬇️  Click 'Download Results' button below\n"
            
            return status, zip_path
            
        except Exception as e:
            error_msg = f"❌ ERROR: {str(e)}\n\n"
            error_msg += "Please check:\n"
            error_msg += "  • Reference images contain clear faces\n"
            error_msg += "  • Input images are valid\n"
            error_msg += "  • File formats are supported (JPG, PNG)\n"
            return error_msg, None

# Create wrapper instance
wrapper = GradioWrapper()

# ============================================
# Create Gradio Interface
# ============================================

def create_demo():
    """Create the Gradio web interface"""
    
    with gr.Blocks(
        title="Face Extraction System",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange")
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🎯 Targeted Face Extraction System
        
        **Find and organize photos of specific people from a large image collection**
        
        """)
        
        # Instructions
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown("""
            ### Step-by-Step Instructions:
            
            1. **Upload Reference Faces** (Required)
               - Upload clear, frontal face photos of people you want to find
               - The filename will be used as the person's name
               - Example: `Alice.jpg`, `Bob.png`, `Charlie.jpeg`
               - Tip: One face per image works best
            
            2. **Upload Images to Search** (Required)
               - Upload all your photos where you want to find these people
               - Can include multiple people per photo
               - Supports JPG, PNG formats
            
            3. **Adjust Match Threshold** (Optional)
               - **Lower (0.45-0.50)**: Stricter matching, fewer false positives
               - **Medium (0.55)**: Balanced (recommended)
               - **Higher (0.60-0.65)**: More lenient, catches more matches
            
            4. **Click "🚀 Extract Faces"**
               - Wait for processing (2-3 seconds per image)
               - Review the status log
            
            5. **Download Results**
               - Click the download button to get a ZIP file
               - Extract the ZIP to see organized folders by person
            
            ---
            
            ### 💡 Tips for Best Results:
            - Use high-quality reference photos (clear, well-lit, frontal view)
            - Reference photos should show the face clearly without sunglasses/masks
            - If you're missing matches, increase the threshold
            - If you're getting wrong people, decrease the threshold
            - Processing time: approximately 2-3 seconds per image
            
            ### 📁 Output Structure:
```
            results.zip/
            ├── Alice/
            │   ├── photo1.jpg
            │   ├── photo5.jpg
            │   └── photo12.jpg
            ├── Bob/
            │   ├── photo2.jpg
            │   └── photo8.jpg
            └── Charlie/
                └── photo3.jpg
```
            """)
        
        # Main interface
        with gr.Row():
            # Left column: Uploads
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Files")
                
                reference_upload = gr.File(
                    label="📸 Reference Faces (name files as: Alice.jpg, Bob.jpg, etc.)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath"
                )
                
                input_upload = gr.File(
                    label="🖼️ Images to Search Through",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.40,
                    maximum=0.70,
                    value=0.55,
                    step=0.05,
                    label="🎚️ Match Threshold",
                    info="Lower = stricter, Higher = more lenient"
                )
                
                extract_btn = gr.Button(
                    "🚀 Extract Faces",
                    variant="primary",
                    size="lg"
                )
                
                # Example section
                with gr.Accordion("📝 Naming Examples", open=False):
                    gr.Markdown("""
                    **Good filename examples:**
                    - `Alice.jpg` → Creates folder named "Alice"
                    - `Bob Smith.png` → Creates folder named "Bob Smith"
                    - `john_doe.jpeg` → Creates folder named "john_doe"
                    
                    **Avoid:**
                    - `IMG_1234.jpg` (not descriptive)
                    - `photo.png` (not descriptive)
                    """)
            
            # Right column: Results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                
                status_box = gr.Textbox(
                    label="Status Log",
                    lines=20,
                    max_lines=25,
                    placeholder="Status messages will appear here...",
                    show_copy_button=True
                )
                
                download_box = gr.File(
                    label="⬇️ Download Results (ZIP)",
                    type="filepath"
                )
                
                # Statistics (optional)
                gr.Markdown("""
                ---
                ### ℹ️ System Information
                - **Model**: InsightFace (ArcFace)
                - **Detection**: MTCNN
                - **Embedding Size**: 512D
                - **Processing**: CPU (Colab)
                """)
        
        # Connect the button
        extract_btn.click(
            fn=wrapper.process_files,
            inputs=[reference_upload, input_upload, threshold_slider],
            outputs=[status_box, download_box],
            api_name="extract_faces"  # Enables API access
        )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
        <p>Built with InsightFace, Gradio, and ❤️</p>
        <p>⚠️ This app processes images locally. Files are not stored permanently.</p>
        </div>
        """)
    
    return demo

# ============================================
# Launch the Application
# ============================================

# Create the demo
demo = create_demo()

# Launch options
print("🚀 Launching Gradio Interface...")
print("="*60)

# For Google Colab - Creates a public shareable link
demo.launch(
    share=True,           # Creates public URL (required for Colab)
    debug=True,           # Show detailed error messages
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    inbrowser=True        # Automatically open in browser
)

print("="*60)
print("✅ Application is running!")
print("📱 Share the link above with anyone to use the app")
print("⚠️  The link expires when you stop the Colab runtime")



