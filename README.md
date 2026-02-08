# 🎯 Targeted Face Extraction System

An AI-powered system that automatically finds and organizes photos of specific people from large image collections using deep learning face recognition.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

- ✅ **Automated Face Detection** - Detects faces in images using MTCNN
- ✅ **Deep Learning Recognition** - Uses ArcFace embeddings for accurate face matching
- ✅ **Batch Processing** - Process hundreds of images at once
- ✅ **Smart Organization** - Automatically creates folders by person name
- ✅ **Web Interface** - Easy-to-use Gradio UI (no coding required)
- ✅ **High Accuracy** - State-of-the-art face recognition technology
- ✅ **Multi-face Support** - Handles images with multiple people
- ✅ **Customizable Threshold** - Adjust matching strictness

## 🎬 Demo

🔗 **Live Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/face-extraction)

### Example Output

**Input**: 
- 6 reference face images (Alice, Bob, Charlie, etc.)
- 152 mixed photos

**Output**: 
- Organized folders with matched photos
- Processing time: ~30 seconds (with GPU)
```
results/
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

## 🚀 Quick Start

Visit our https://huggingface.co/spaces/utsav05/ImageClustering and:

1. Upload reference face photos
2. Upload images to search through
3. Click "Extract Faces"
4. Download organized results


```

## 🛠️ Technical Details

### Architecture
```
Input Images → Face Detection → Feature Extraction → Clustering → Organization
     ↓              (MTCNN)         (ArcFace)         (DBSCAN)       ↓
 Multiple                                                        Organized
   People                                                         Folders
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| **Face Detection** | MTCNN (Multi-task Cascaded CNN) |
| **Face Recognition** | ArcFace (Additive Angular Margin Loss) |
| **Embeddings** | 512-dimensional feature vectors |
| **Similarity Metric** | Cosine Similarity |
| **Framework** | InsightFace, OpenCV |
| **Interface** | Gradio |
| **Backend** | Python 3.8+ |

### Performance

| Metric | Value |
|--------|-------|
| **Processing Speed (CPU)** | ~1-2 seconds/image |
| **Processing Speed (GPU)** | ~0.2-0.3 seconds/image |
| **Accuracy (LFW)** | 99.8% |
| **Embedding Size** | 512D |
| **Supported Formats** | JPG, PNG, BMP, TIFF |

## ⚙️ Configuration

### Parameters
```python
TargetedFaceExtractor(
    input_folder="./images",           # Folder with images to search
    output_folder="./output",          # Where to save results
    reference_folder="./references",   # Reference face photos
    match_threshold=0.55,              # 0.45-0.65 (lower = stricter)
    detection_size=320,                # 224/320/640 (smaller = faster)
    resize_images=True                 # Resize large images for speed
)
```

### Threshold Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.45-0.50 | Very strict | When you need high precision |
| 0.50-0.55 | Balanced | **Recommended for most cases** |
| 0.55-0.60 | Lenient | When faces vary significantly |
| 0.60-0.65 | Very lenient | Risk of false positives |

## 📊 Use Cases

- 📸 **Photo Organization** - Sort thousands of family photos by person
- 🎥 **Event Photography** - Find specific people in event photos
- 🔍 **Search & Retrieval** - Locate all photos containing a specific person
- 📱 **Digital Albums** - Auto-organize phone gallery
- 🏢 **Corporate** - Organize employee photos
- 🎓 **Education** - Sort class photos by student

## 🎓 How It Works

### Step-by-Step Process

1. **Load Reference Faces**
```
   Reference photos → Face detection → Extract embeddings → Store as reference
```

2. **Process Search Images**
```
   For each image:
     - Detect all faces
     - Extract embeddings
     - Compare with reference embeddings
     - If similarity > threshold → Match found
```

3. **Organize Results**
```
   Create folder for each person → Copy matched images → Avoid duplicates
```

### Algorithm Details

**Face Detection:**
- Uses MTCNN (Multi-task Cascaded Convolutional Networks)
- Detects faces at multiple scales
- Returns bounding boxes and facial landmarks

**Feature Extraction:**
- ArcFace model trained on millions of faces
- Generates 512-dimensional embedding per face
- Invariant to pose, lighting, expression

**Matching:**
- Cosine similarity between embeddings
- Threshold-based classification
- Returns best match per detected face

## 📈 Performance Optimization

### Speed Up Processing

**Enable GPU (20x faster):**
```python
# In Colab: Runtime → Change runtime type → GPU
extractor = TargetedFaceExtractor(
    use_gpu=True,          # Enable GPU acceleration
    detection_size=320     # Smaller = faster
)
```

**Reduce Image Size:**
```python
extractor = TargetedFaceExtractor(
    resize_images=True,
    max_image_size=1920    # Resize larger images
)
```

**Batch Processing:**
- Process 100+ images in parallel
- Utilizes multi-core CPU

### Accuracy Improvements

**Use High-Quality References:**
- Clear, frontal face photos
- Good lighting
- No obstructions (sunglasses, masks)
- One face per reference image

**Adjust Threshold:**
- Start with 0.55
- Decrease if getting false matches
- Increase if missing correct matches

**Multiple Reference Photos:**
```
references/
├── Alice/
│   ├── ref1.jpg
│   ├── ref2.jpg
│   └── ref3.jpg
└── Bob/
    └── ref1.jpg
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

1. Check existing issues first
2. Create new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots (if applicable)

### Feature Requests

1. Open an issue with tag `enhancement`
2. Describe the feature and use case
3. Explain why it would be useful

### Pull Requests

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2024 [Utsav Kumar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🙏 Acknowledgments

- **InsightFace** - Face recognition framework
- **Gradio** - Web interface framework
- **ArcFace** - Face recognition model
- **MTCNN** - Face detection algorithm

## 📚 Research Papers

This project is based on the following research:

1. **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
   - [Paper](https://arxiv.org/abs/1801.07698)

2. **MTCNN**: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks", SPL 2016
   - [Paper](https://arxiv.org/abs/1604.02878)

## 🐛 Troubleshooting

### Common Issues

**Issue: "No faces detected in reference images"**
```
Solution:
- Ensure reference photos show clear faces
- Check image quality and lighting
- Try different reference photos
```

**Issue: "Too many false matches"**
```
Solution:
- Decrease threshold (e.g., from 0.55 to 0.50)
- Use higher quality reference photos
- Enable quality filtering
```

**Issue: "Missing correct matches"**
```
Solution:
- Increase threshold (e.g., from 0.55 to 0.60)
- Use multiple reference photos per person
- Check if faces are clearly visible in images
```

**Issue: "Out of memory"**
```
Solution:
- Reduce detection_size to 224 or 320
- Enable image resizing
- Process fewer images at once
```

## 📞 Support

- **Email**: 05utsavkumar@gmail.com

## 🗺️ Roadmap

- [ ] Add video processing support
- [ ] Real-time webcam face recognition
- [ ] Mobile app (iOS/Android)
- [ ] Face clustering (group unknown faces)
- [ ] Multi-language support
- [ ] Cloud storage integration (Google Drive, Dropbox)
- [ ] API endpoints for integration
- [ ] Docker deployment
- [ ] Face age/emotion detection


## 🌍 Community

Join our community:
- ⭐ Star this repo if you find it useful
- 🐛 Report bugs and request features
- 💬 Share your use cases
- 🤝 Contribute code or documentation

## 📖 Citation

If you use this project in your research, please cite:
```bibtex
@software{face_extraction_system,
  author = {Utsav Kumar},
  title = {Targeted Face Extraction System},
  year = {2024},
  url = {https://github.com/Utsavkumar001/-Targeted-Face-Extraction-System}
}
```

---

<div align="center">

**Built with ❤️ using InsightFace and Gradio**

[Report Bug](https://github.com/YOUR_USERNAME/face-extraction-system/issues) · 
[Request Feature](https://github.com/YOUR_USERNAME/face-extraction-system/issues) · 
[Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/face-extraction)

</div>
```

---

