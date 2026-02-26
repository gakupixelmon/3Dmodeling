# 3Dmodeling
# Image-to-3D Visual Hull API

This project provides a fast and lightweight REST API to generate 3D models (`.glb`) from 2D orthogonal images using the Visual Hull (Shape-from-Silhouette) algorithm. 

Heavy deep learning frameworks like PyTorch are **not required** for the core 3D reconstruction. It purely relies on classic computer vision and geometric processing, making it highly portable and suitable for rapid prototyping.

## Features
* **Fast 3D Generation**: Generates 3D mesh data using the Marching Cubes algorithm.
* **Automatic Background Removal**: Integrates `rembg` to automatically extract object silhouettes.

* **FastAPI Backend**: Ready-to-use asynchronous API with CORS enabled.

## Algorithm Overview
This project implements the **Visual Hull** method. For a set of $N$ images, each image provides a 2D silhouette. By back-projecting these 2D silhouettes into 3D space, we obtain a visual cone $C_i$ for each view. The 3D geometry of the object, $V$, is approximated by the intersection of all visual cones:

$$V = \bigcap_{i=1}^{N} C_i$$

In this implementation, space is discretized into a 3D voxel grid. Each voxel is evaluated against the 2D masks, and the final surface mesh is extracted using the Marching Cubes algorithm (`skimage.measure.marching_cubes`).

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/gakupixelmon/3Dmodeling.git
cd 3Dmodeling
```
### 2. Create and Activate a Virtual Environment (Recommended)
windows
```bash
python -m venv venv
venv\Scripts\activate
```
macOS, Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install fastapi uvicorn python-multipart numpy opencv-python trimesh scikit-image rembg Pillow
```
### 4. Start the Application
```bash
uvicorn main:app --reload
```
### 5. Access via Browser
Once the server is running, open your browser and go to the following URL. You can test the 3D model generation through an intuitive UI:
