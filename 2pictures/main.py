import os
import uuid
import json
import numpy as np
import cv2
import trimesh
from skimage import measure
from rembg import remove
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 変更点: 生成したモデルをローカルのmodelsフォルダから配信できるようにする
os.makedirs("models", exist_ok=True)
app.mount("/models", StaticFiles(directory="models"), name="models")

class Interactive_VisualHull_API:
    def __init__(self, resolution=256):
        self.res = resolution

    def remove_background(self, img_cv):
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        result_pil = remove(pil_img)
        result_np = np.array(result_pil)
        
        if result_np.shape[2] == 4:
            mask_obj = result_np[:, :, 3]
        else:
            mask_obj = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)

        _, mask_obj = cv2.threshold(mask_obj, 128, 255, cv2.THRESH_BINARY)
        img_masked = cv2.bitwise_and(img_cv, img_cv, mask=mask_obj)
        return mask_obj, img_masked

    def rotate_image_and_mask(self, img, mask, angle):
        h, w = img.shape[:2]
        cX, cY = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated_img = cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))
        rotated_mask = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
        return rotated_img, rotated_mask

    def align_and_process(self, image_path, points_json):
        img_cv = cv2.imread(image_path)
        h, w = img_cv.shape[:2]

        points = json.loads(points_json)
        p1 = (int(points[0]['x'] * w), int(points[0]['y'] * h))
        p2 = (int(points[1]['x'] * w), int(points[1]['y'] * h))

        mask_obj, img_masked = self.remove_background(img_cv)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = np.sqrt(dx**2 + dy**2)

        theta = np.degrees(np.arctan2(dy, dx))
        rot_angle = 90 - theta

        rotated_img, rotated_mask = self.rotate_image_and_mask(img_masked, mask_obj, rot_angle)

        M_obj = cv2.moments(rotated_mask)
        center_x = int(M_obj["m10"] / M_obj["m00"])
        center_y = int(M_obj["m01"] / M_obj["m00"])

        return rotated_img, rotated_mask, (center_x, center_y), distance

    def create_mesh(self, path_front, path_left, points1_json, points2_json, output_filename):
        img1, mask1, center1, dist1 = self.align_and_process(path_front, points1_json)
        img2, mask2, center2, dist2 = self.align_and_process(path_left, points2_json)
        
        scale_factor = dist1 / dist2
        base_scale = dist1 / 2.0
        side_scale = dist2 / 2.0
        
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        
        max_x = max(center1[0], w1 - center1[0]) / base_scale
        max_y1 = max(center1[1], h1 - center1[1]) / base_scale
        max_z = max(center2[0], w2 - center2[0]) / side_scale
        max_y2 = max(center2[1], h2 - center2[1]) / side_scale
        
        voxel_range = max(max_x, max_y1, max_z, max_y2) * 1.1
        norm_scale = base_scale

        x, y, z = np.mgrid[-voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j]
        voxels = np.ones(x.shape, dtype=bool)
        
        u1 = (x * norm_scale) + center1[0]
        v1 = (y * -norm_scale) + center1[1]
        u2 = (z * norm_scale / scale_factor) + center2[0]
        v2 = (y * -norm_scale / scale_factor) + center2[1]

        def check_mask(mask, u, v):
            h, w = mask.shape
            u_idx, v_idx = u.astype(int), v.astype(int)
            valid = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
            res = np.zeros(u.shape, dtype=bool)
            res[valid] = mask[v_idx[valid], u_idx[valid]] > 0
            return res

        voxels &= check_mask(mask1, u1, v1)
        voxels &= check_mask(mask2, u2, v2)

        voxels_padded = np.pad(voxels, pad_width=1, mode='constant', constant_values=False)
        verts, faces, _, _ = measure.marching_cubes(voxels_padded, 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.repair.fix_normals(mesh)
        trimesh.smoothing.filter_taubin(mesh, iterations=15)

        vx = (verts[:, 0] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range
        vy = (verts[:, 1] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range
        vz = (verts[:, 2] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range

        uv1_u = np.clip(np.round((vx * norm_scale) + center1[0]).astype(int), 0, w1 - 1)
        uv1_v = np.clip(np.round((vy * -norm_scale) + center1[1]).astype(int), 0, h1 - 1)
        uv2_u = np.clip(np.round((vz * norm_scale / scale_factor) + center2[0]).astype(int), 0, w2 - 1)
        uv2_v = np.clip(np.round((vy * -norm_scale / scale_factor) + center2[1]).astype(int), 0, h2 - 1)

        colors_front = img1[uv1_v, uv1_u][:, ::-1]
        colors_left  = img2[uv2_v, uv2_u][:, ::-1]

        weight_z, weight_x = np.abs(mesh.vertex_normals[:, 2]), np.abs(mesh.vertex_normals[:, 0]) 
        use_front = weight_z > weight_x
        final_colors = np.where(use_front[:, None], colors_front, colors_left)

        alpha = np.full((len(final_colors), 1), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = np.hstack((final_colors, alpha))
        
        mesh.apply_translation(-mesh.centroid)
        mesh.export(output_filename)


@app.post("/generate")
async def generate_model(
    image1: UploadFile = File(...), 
    image2: UploadFile = File(...),
    points1: str = Form(...),
    points2: str = Form(...)
):
    try:
        front_path = f"tmp_front_{uuid.uuid4()}.jpg"
        left_path = f"tmp_left_{uuid.uuid4()}.jpg"
        glb_filename = f"model_{uuid.uuid4()}.glb"

        with open(front_path, "wb") as buffer:
            shutil.copyfileobj(image1.file, buffer)
        with open(left_path, "wb") as buffer:
            shutil.copyfileobj(image2.file, buffer)

        # 3Dモデル生成
        converter = Interactive_VisualHull_API(resolution=256)
        
        # 🌟 変更点: modelsフォルダ内に保存する
        output_filepath = os.path.join("models", glb_filename)
        converter.create_mesh(front_path, left_path, points1, points2, output_filepath)
        
        # 🌟 変更点: ローカルサーバーのURLを返す
        glb_url = f"http://localhost:8000/models/{glb_filename}"

        # 一時ファイル（画像）の削除
        os.remove(front_path)
        os.remove(left_path)

        return {"glb_url": glb_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
