import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import torch
import torchio as tio
from torchvision.transforms import v2
from torchvision import tv_tensors
from scipy.ndimage import gaussian_filter, center_of_mass

import sys
sys.path.insert(0, '/home/zzx/Cardiac_Function_Analysis/CardiacAI')
from tasks.strain.models import VxmDense
from tasks.seg3d.models.unet import UNet
from tasks.landmark.models.hrnet import HighResolutionNet

def center_by_centroid(img, mask, output_size=(128, 128)):
    """
    Center img and mask based on the centroid of the mask in each D slice,
    and crop/pad to output_size.
    
    Args:
        img: Tensor of shape [D, H, W]
        mask: Tensor of shape [D, H, W] (binary mask)
        output_size: Tuple of (height, width) for output
        
    Returns:
        Centered img and mask tensors of shape [D, output_height, output_width]
    """
    assert img.shape == mask.shape, "img and mask must have the same shape"
    D, H, W = img.shape
    out_h, out_w = output_size
    
    # Initialize output tensors
    centered_img = torch.zeros((D, out_h, out_w), dtype=img.dtype, device=img.device)
    centered_mask = torch.zeros((D, out_h, out_w), dtype=mask.dtype, device=mask.device)
    
    for d in range(D):
        # Calculate centroid for current slice
        slice_mask = mask[d%(D//2) + (D//2)].cpu().numpy() #
        coords = np.argwhere(slice_mask == 1)
        
        if len(coords) > 0:
            cy, cx = coords.mean(axis=0).astype(int)  # (y, x)
        else:
            cy, cx = H // 2, W // 2  # Fallback to center
        
        # Calculate translation needed to center the centroid
        translate_y = out_h // 2 - cy
        translate_x = out_w // 2 - cx
        
        # Calculate source and target regions
        src_y_start = max(0, -translate_y)
        src_y_end = min(H, out_h - translate_y)
        src_x_start = max(0, -translate_x)
        src_x_end = min(W, out_w - translate_x)
        
        tgt_y_start = max(0, translate_y)
        tgt_y_end = min(out_h, H + translate_y)
        tgt_x_start = max(0, translate_x)
        tgt_x_end = min(out_w, W + translate_x)
        
        # Copy valid regions
        if src_y_end > src_y_start and src_x_end > src_x_start:
            centered_img[d, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = \
                img[d, src_y_start:src_y_end, src_x_start:src_x_end]
            centered_mask[d, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = \
                mask[d, src_y_start:src_y_end, src_x_start:src_x_end]
    
    return centered_img, centered_mask


class CardiacRegistrator:
    def __init__(self, registration_model_path, device, target_size):
        self.device = device
        self.registration_model = self.load_registration_model(registration_model_path)
        self.target_size = target_size
        self.target_spacing = 1.25

    def load_registration_model(self, model_path):
        model = VxmDense.load(model_path, self.device)
        model.eval()
        model.to(self.device)
        return model
    
    def preprocess(self, img_array, mask_array, origin_spacing):
        self.origin_size = img_array.shape[-2:]
        img = tv_tensors.Image(img_array)
        mask = tv_tensors.Mask(mask_array)
        mean = img.mean()
        std = img.std()
        print('image shape: ', img.shape[-2:])
        print('image spacing: ', origin_spacing[-2:])
        self.resample_size = [int(l * os / self.target_spacing) 
                         for l, os in zip(self.origin_size[-2:], origin_spacing[-2:])]
        
        transforms = v2.Compose([
            v2.Normalize(mean=[mean], std=[std]),
            v2.Resize(self.resample_size),
            v2.CenterCrop(self.target_size)
        ])
        return transforms(img), transforms(mask)
    
    def preprocess_V2(self, img, mask, origin_spacing):
        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)
        
        mean = img.mean()
        std = img.std()
        #print(mean.item(), std.item())
        resample_size = [int(l * os / self.target_spacing) for l, os in zip(img.shape[-2:], origin_spacing[-2:])]
        
        transforms_img = v2.Compose([
            v2.Normalize(mean=[mean], std=[std]),
            v2.Resize(resample_size)
        ])
        img = transforms_img(img)
        mask = transforms_img(mask)
        img, mask = center_by_centroid(img, mask, output_size=self.target_size)
        return img,mask
    
    def postprocess(self, disp, moved):
        disp = tv_tensors.Image(disp)
        moved = tv_tensors.Image(moved)
        transforms = v2.Compose([
            v2.CenterCrop(self.resample_size),
            v2.Resize(self.origin_size)
        ])
        return transforms(disp).cpu().detach().numpy(), transforms(moved).cpu().detach().numpy()

    def process(self, ed_image_path, es_image_path, ed_mask, es_mask, center=True):
        """ 加载数据 """
        print('Register::process')
        ed_image_nib = nib.load(ed_image_path)
        ed_image_spacing = ed_image_nib.header.get_zooms()[::-1]
        ed_image_array = ed_image_nib.get_fdata().transpose(2, 1, 0).astype(np.float32)
        if isinstance(ed_mask, (str, Path)):
            ed_mask_array = nib.load(ed_mask).get_fdata().transpose(2, 1, 0)
            if 'frame' in ed_mask:
                ed_mask_array[ed_mask_array==1] = 4
                ed_mask_array[ed_mask_array==3] = 1
        else:
            ed_mask_array = ed_mask
            ed_mask_array[ed_mask_array==1] = 4
            ed_mask_array[ed_mask_array==3] = 1
        
        es_image_nib = nib.load(es_image_path)
        es_image_spacing = es_image_nib.header.get_zooms()[::-1]
        es_image_array = es_image_nib.get_fdata().transpose(2, 1, 0).astype(np.float32)
        if isinstance(es_mask, (str, Path)):
            es_mask_array = nib.load(es_mask).get_fdata().transpose(2, 1, 0)
            if 'frame' in str(es_mask):
                es_mask_array[es_mask_array==1] = 4
                es_mask_array[es_mask_array==3] = 1
        else:
            es_mask_array = es_mask
            es_mask_array[es_mask_array==1] = 4
            es_mask_array[es_mask_array==3] = 1
        
        """ 预处理 """
        # ed_image_array: [D,H,W]
        if center:
            N = es_image_array.shape[0]
            paired_img_ndarray = np.concatenate((es_image_array, ed_image_array), axis=0) # [2D,H,W]
            paired_gt_ndarray = np.concatenate((es_mask_array, ed_mask_array), axis=0)
            paired_img_ndarray, paired_gt_ndarray = self.preprocess_V2(paired_img_ndarray, paired_gt_ndarray, ed_image_spacing)
            es_image_preprocess = paired_img_ndarray[0:N].unsqueeze(1).to(self.device)
            ed_image_preprocess = paired_img_ndarray[N:].unsqueeze(1).to(self.device)
            es_mask_preprocess = paired_gt_ndarray[0:N].unsqueeze(1).to(self.device) # [D,1,128,128]
            ed_mask_preprocess = paired_gt_ndarray[N:] # [D,128,128]
        else:
            ed_image_preprocess, ed_mask_preprocess = self.preprocess(ed_image_array, ed_mask_array, ed_image_spacing)
            ed_image_preprocess = ed_image_preprocess.unsqueeze(1).to(self.device)

            es_image_preprocess, es_mask_preprocess = self.preprocess(es_image_array, es_mask_array, es_image_spacing)
            es_image_preprocess = es_image_preprocess.unsqueeze(1).to(self.device)
            es_mask_preprocess = es_mask_preprocess.unsqueeze(1).to(self.device)

        """ 模型推理 """
        # es_image_preprocess: [D,1,H,W]
        # flow: [D,2,H,W], 其中坐标维度顺序就是图像的维度顺序，vxm中已手动修正
        moved, flow = self.registration_model(es_image_preprocess, ed_image_preprocess, registration=True)
        moved_mask = self.registration_model.transformer(es_mask_preprocess.float(), flow, mode='nearest')
        """ 将flow转换为实际位移距离mm """
        # flow: [D,2,H,W]
        # flow = flow * self.target_spacing
        """ 后处理 """
        #displacement, moved = self.postprocess(flow, moved)
        return mov_to_numpy(flow), mov_to_numpy(moved),\
               mov_to_numpy(moved_mask), mov_to_numpy(ed_mask_preprocess)

def mov_to_numpy(tensor):
    return tensor.squeeze().cpu().detach().numpy()

class CardiacSaxSegmentation:
    # ACDC标签语义：1-RV, 2-MYO, 3-LV
    RVC = 1
    LVM = 2
    LVC = 3

    def __init__(self, model_path, device, target_spacing=1.25, patch_size=(192, 192, 1),
                 position_dir='/home/zzx/data/ACDC/Position'):
        self.device = device
        self.segmentation_model = self.load_model(model_path)
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.position_dir = position_dir

    def load_model(self, model_path):
        # 对齐 CardiacAI/tasks/seg3d/runner.py 的模型构建与加载方式
        model = UNet(in_channels=1, num_classes=4, base_c=48, drop_rate=None)
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid seg3d checkpoint format: {model_path}")
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load seg3d weights from {model_path}. "
                "Please make sure model definition and checkpoint source are both from CardiacAI seg3d."
            ) from e
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _get_position_from_txt(txt_path):
        a = np.loadtxt(txt_path, dtype=str)
        return (
            max(int(a[0][7]) - 20, 0),
            int(a[0][-1]) + 20,
            max(int(a[1][7]) - 20, 0),
            int(a[1][-1]) + 20,
        )

    def _extract_patient_id(self, image_path):
        name = Path(image_path).name
        # e.g. patient061_frame01.nii.gz
        pid = ''.join(ch for ch in name if ch.isdigit())[:3]
        return int(pid) if pid else None

    def preprocess(self, img_array, origin_spacing, image_path):
        # 对齐 Seg3DTestDataset: [H,W,D] + z-score + optional VOI + resample(xy) + per-slice CropOrPad
        img_hwd = img_array.astype(np.float32)  # [H,W,D]
        img_hwd = (img_hwd - img_hwd.mean()) / (img_hwd.std() + 1e-6)
        orig_h, orig_w, orig_d = img_hwd.shape

        # optional VOI crop from position txt
        if self.position_dir:
            pid = self._extract_patient_id(image_path)
            pos_path = Path(self.position_dir) / f'patient{pid}_VOIPosition.txt' if pid is not None else None
        else:
            pos_path = None

        if pos_path is not None and pos_path.exists():
            y0, y1, x0, x1 = self._get_position_from_txt(str(pos_path))
            y0, y1 = max(0, y0), min(orig_h, y1)
            x0, x1 = max(0, x0), min(orig_w, x1)
            img_hwd = img_hwd[y0:y1, x0:x1, :]
        else:
            y0, y1, x0, x1 = 0, orig_h, 0, orig_w

        voi_h, voi_w, d = img_hwd.shape
        sp_h, sp_w = origin_spacing[0], origin_spacing[1]
        res_h = int(voi_h * sp_h / self.target_spacing)
        res_w = int(voi_w * sp_w / self.target_spacing)

        # resample XY per-slice
        x = torch.from_numpy(np.transpose(img_hwd, (2, 0, 1))).unsqueeze(1).float()  # [D,1,H,W]
        x = torch.nn.functional.interpolate(x, size=(res_h, res_w), mode='bilinear', align_corners=False)
        x = x.squeeze(1)  # [D,res_h,res_w]

        target_h, target_w = self.patch_size[0], self.patch_size[1]
        crop_y0 = max((res_h - target_h) // 2, 0)
        crop_x0 = max((res_w - target_w) // 2, 0)
        crop_y1 = min(crop_y0 + target_h, res_h)
        crop_x1 = min(crop_x0 + target_w, res_w)
        x = x[:, crop_y0:crop_y1, crop_x0:crop_x1]

        pad_h = target_h - x.shape[1]
        pad_w = target_w - x.shape[2]
        pad_top = max(pad_h // 2, 0)
        pad_bottom = max(pad_h - pad_top, 0)
        pad_left = max(pad_w // 2, 0)
        pad_right = max(pad_w - pad_left, 0)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

        # convert to [H,W,D] for seg3d test-style inference
        x_hwd = np.transpose(x.cpu().numpy(), (1, 2, 0))

        self.seg_geom = {
            'orig_h': orig_h, 'orig_w': orig_w, 'orig_d': orig_d,
            'voi_y0': y0, 'voi_y1': y1, 'voi_x0': x0, 'voi_x1': x1,
            'voi_h': voi_h, 'voi_w': voi_w,
            'res_h': res_h, 'res_w': res_w,
            'crop_y0': crop_y0, 'crop_y1': crop_y1,
            'crop_x0': crop_x0, 'crop_x1': crop_x1,
            'pad_top': pad_top, 'pad_bottom': pad_bottom,
            'pad_left': pad_left, 'pad_right': pad_right,
        }
        return x_hwd

    def postprocess(self, pred_hwd):
        # inverse: patch -> resampled VOI -> original VOI -> original full image
        g = self.seg_geom
        pred = torch.from_numpy(pred_hwd.astype(np.float32)).permute(2, 0, 1)  # [D,H,W]

        y0 = g['pad_top']
        y1 = self.patch_size[0] - g['pad_bottom'] if g['pad_bottom'] > 0 else self.patch_size[0]
        x0 = g['pad_left']
        x1 = self.patch_size[1] - g['pad_right'] if g['pad_right'] > 0 else self.patch_size[1]
        pred = pred[:, y0:y1, x0:x1]

        canvas = torch.zeros((pred.shape[0], g['res_h'], g['res_w']), dtype=pred.dtype)
        canvas[:, g['crop_y0']:g['crop_y1'], g['crop_x0']:g['crop_x1']] = pred

        # inverse resample to VOI size
        canvas = canvas.unsqueeze(1)  # [D,1,H,W]
        canvas = torch.nn.functional.interpolate(canvas, size=(g['voi_h'], g['voi_w']), mode='nearest')
        voi_pred = canvas.squeeze(1).permute(1, 2, 0).numpy().astype(np.int16)  # [voi_h,voi_w,D]

        full_pred = np.zeros((g['orig_h'], g['orig_w'], g['orig_d']), dtype=np.int16)
        full_pred[g['voi_y0']:g['voi_y1'], g['voi_x0']:g['voi_x1'], :] = voi_pred

        # return [D,W,H] to match existing downstream expectation (e.g. 9,256,216)
        return np.transpose(full_pred, (2, 1, 0))

    def process(self, ed_image_path):
        """返回与原图空间对齐的多类别标签图（心肌=2）。"""
        if isinstance(ed_image_path, str):
            # 与 Seg3DTestDataset 保持一致：使用 torchio 读取 [H,W,D] 与 spacing
            tio_img = tio.ScalarImage(ed_image_path)
            ed_image_spacing = tio_img.spacing
            ed_image_array = tio_img.numpy()[0].astype(np.float32)  # [H,W,D]
        else:
            raise TypeError('CardiacSaxSegmentation.process currently expects image path string input.')

        img_hwd = self.preprocess(ed_image_array, ed_image_spacing, ed_image_path)  # [H,W,D]
        img = torch.from_numpy(img_hwd).to(self.device)
        h, w, d = img.shape
        ps_d = max(int(self.patch_size[2]), 1)
        pad_d = (ps_d - 1) // 2
        img_pad = torch.nn.functional.pad(img, (pad_d, pad_d), mode='replicate')

        num_classes = 4
        prob = np.zeros((num_classes, h, w, d), dtype=np.float32)
        with torch.no_grad():
            for z in range(d):
                patch = img_pad[:, :, z:z + ps_d].permute(2, 0, 1).unsqueeze(0)  # [1,d,H,W]
                pred = self.segmentation_model(patch.to(self.device))  # [1,C,H,W]
                prob[:, :, :, z] = pred.squeeze(0).detach().cpu().numpy()

        pred_cls = np.argmax(prob, axis=0).astype(np.int16)  # [H,W,D]
        label = self.postprocess(pred_cls)
        return label


class CardiacLandmark:
    LVC = 1
    LVM = 2
    RVC = 3

    def __init__(self, model_path, device):
        self.device = device
        self.landmark_model = self.load_model(model_path)
        self.target_size = (160, 160)
        self.target_spacing = 1.25

    def load_model(self, model_path):
        # 对齐 CardiacAI/tasks/landmark/runner.py 的测试配置
        model = HighResolutionNet(base_channel=32, num_joints=2, head='srp')
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid landmark checkpoint format: {model_path}")
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load landmark weights from {model_path}. "
                "Please make sure head/num_joints and checkpoint source are both from CardiacAI landmark."
            ) from e
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, img_array, origin_spacing, mask):
        # 对齐 landmark test: z-score -> 按target_spacing重采样 -> 以ROI中心裁剪/填充到160x160
        if img_array.ndim == 2:
            img_array = img_array[None, ...]
        img_array = img_array.astype(np.float32)
        img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-6)

        self.origin_size = img_array.shape[-2:]
        self.resample_size = [
            int(l * os / self.target_spacing)
            for l, os in zip(self.origin_size, origin_spacing[-2:])
        ]
        res_h, res_w = self.resample_size

        x = torch.from_numpy(img_array).unsqueeze(1).float()  # [D,1,H,W]
        x = torch.nn.functional.interpolate(x, size=(res_h, res_w), mode='bilinear', align_corners=False)
        x = x.squeeze(1)

        mask2d = mask.astype(np.float32)
        if mask2d.ndim == 3:
            mask2d = (mask2d == self.LVM).any(axis=0).astype(np.float32)
        mask_t = torch.from_numpy(mask2d)[None, None]
        mask_t = torch.nn.functional.interpolate(mask_t, size=(res_h, res_w), mode='nearest').squeeze().numpy()

        coords = np.argwhere(mask_t > 0)
        if coords.size == 0:
            cy, cx = res_h // 2, res_w // 2
        else:
            cy, cx = coords.mean(axis=0).astype(int)

        target_h, target_w = self.target_size
        y_min = max(0, cy - target_h // 2)
        x_min = max(0, cx - target_w // 2)
        y_max = min(y_min + target_h, res_h)
        x_max = min(x_min + target_w, res_w)

        x = x[:, y_min:y_max, x_min:x_max]

        pad_h = target_h - x.shape[1]
        pad_w = target_w - x.shape[2]
        pad_top = max(pad_h // 2, 0)
        pad_bottom = max(pad_h - pad_top, 0)
        pad_left = max(pad_w // 2, 0)
        pad_right = max(pad_w - pad_left, 0)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

        self.landmark_geom = {
            'res_h': res_h, 'res_w': res_w,
            'y_min': y_min, 'y_max': y_max,
            'x_min': x_min, 'x_max': x_max,
            'pad_top': pad_top, 'pad_bottom': pad_bottom,
            'pad_left': pad_left, 'pad_right': pad_right,
        }
        return x.to(self.device)

    def postprocess(self, landmark):
        # 网络输出 keypoint 格式按 (x, y)，映射回原图空间
        g = self.landmark_geom
        lm = np.asarray(landmark, dtype=np.float32)
        if lm.ndim == 2:
            lm = lm[None, ...]

        lm[..., 0] = lm[..., 0] - g['pad_left'] + g['x_min']
        lm[..., 1] = lm[..., 1] - g['pad_top'] + g['y_min']

        sx = self.origin_size[1] / g['res_w']
        sy = self.origin_size[0] / g['res_h']
        lm[..., 0] *= sx
        lm[..., 1] *= sy

        lm[..., 0] = np.clip(lm[..., 0], 0, self.origin_size[1] - 1)
        lm[..., 1] = np.clip(lm[..., 1], 0, self.origin_size[0] - 1)
        return lm.squeeze()

    def process(self, ed_image_path, mask):
        """返回原图空间 landmark 坐标（x, y）。"""
        if isinstance(ed_image_path, str):
            ed_image_nib = nib.load(ed_image_path)
            ed_image_spacing = ed_image_nib.header.get_zooms()[::-1]
            ed_image_array = ed_image_nib.get_fdata().transpose(2, 1, 0).astype(np.float32)
            ed_image_preprocess = self.preprocess(ed_image_array, ed_image_spacing, mask)
        else:
            ed_image_preprocess = torch.from_numpy(ed_image_path).float()
            if ed_image_preprocess.ndim == 2:
                ed_image_preprocess = ed_image_preprocess.unsqueeze(0)

        ed_image_preprocess = ed_image_preprocess.unsqueeze(1).to(self.device)

        with torch.no_grad():
            pred = self.landmark_model(ed_image_preprocess)
            if isinstance(pred, list):
                pred = pred[-1]

        # 对齐 landmark test: 从热图解码关键点坐标（argmax）
        if pred.ndim == 4:
            bsz, num_joints, h, w = pred.shape
            flat = pred.reshape(bsz, num_joints, -1)
            hard_index = torch.argmax(flat, dim=-1)
            rows, cols = torch.unravel_index(hard_index, (h, w))
            landmark = torch.stack([cols, rows], dim=-1).float().cpu().numpy()  # [B, K, 2], (x,y)
        else:
            landmark = mov_to_numpy(pred)

        landmark = self.postprocess(landmark)
        return landmark


class CardiacStrainCalculator:
    def __init__(self, registration_model_path, segmentation_model_path, device, target_size):
        self.registration_api = CardiacRegistrator(registration_model_path, device, target_size)
        self.segmentation_api = CardiacSaxSegmentation(segmentation_model_path, device)

    def register_images(self, ed_image, es_image):
        # 使用配准模型进行配准，得到变形场
        deformation_field = self.registration_api.process(ed_image, es_image)
        return deformation_field

    def segment_myocardium(self, ed_image):
        # 使用分割模型得到心肌区域
        myocardium_mask = self.segmentation_api.process(ed_image)
        return myocardium_mask

    def calculate_strain(self, deformation_field, myocardium_mask):
        # 计算径向和周向应变
        # 获取心肌区域的坐标
        myocardium_coords = np.where(myocardium_mask > 0)

        # 初始化应变数组
        radial_strain = np.zeros_like(myocardium_mask, dtype=np.float32)
        circumferential_strain = np.zeros_like(myocardium_mask, dtype=np.float32)

        # 遍历心肌区域的每个像素，计算应变
        for coord in zip(*myocardium_coords):
            # 获取当前像素的变形向量
            deformation_vector = deformation_field[coord]

            # 计算径向应变和周向应变
            # 这里假设径向应变和周向应变的计算方法已经定义
            radial_strain[coord] = np.linalg.norm(deformation_vector)  # 伪代码
            circumferential_strain[coord] = np.arctan2(deformation_vector[1], deformation_vector[0])  # 伪代码
        radial_strain = "计算径向应变的代码"
        circumferential_strain = "计算周向应变的代码"
        return radial_strain, circumferential_strain

    def process(self, ed_image, es_image):
        deformation_field, moved, ed_process,es_process = self.register_images(ed_image, es_image)
        myocardium_mask = self.segment_myocardium(moved)
        #radial_strain, circumferential_strain = self.calculate_strain(deformation_field, myocardium_mask)
        return deformation_field, moved, ed_process, es_process, myocardium_mask

def compute_green_lagrange_strain(displacement_field, mask,gaussian=False, sigma=1):
    """
    计算 Green-Lagrange 应变张量
    :param displacement_field: 位移场，形状为 [H, W, 2]，表示每个像素的位移向量 (u, v)
    :param mask: 左心室的掩码，形状为 [H, W]，1 表示左心室区域，0 表示背景
    :return: Green-Lagrange 应变张量，形状为 [H, W, 2, 2]
    """
    H, W, _ = displacement_field.shape

    # 初始化应变张量
    E = np.zeros((H, W, 2, 2))
    
    # displacement_field[..., 1]对应第一个维度的位移
    # displacement_field[..., 0]对应第二个维度的位移
    # 计算位移梯度
    if gaussian:
        sigma = sigma
        du_dx = gaussian_filter(displacement_field[..., 0], sigma=sigma, order=[1, 0])  # du/dx
        du_dy = gaussian_filter(displacement_field[..., 0], sigma=sigma, order=[0, 1])  # du/dy
        dv_dx = gaussian_filter(displacement_field[..., 1], sigma=sigma, order=[1, 0])  # dv/dx
        dv_dy = gaussian_filter(displacement_field[..., 1], sigma=sigma, order=[0, 1])  # dv/dy
    else:
        du_dx,du_dy = np.gradient(np.squeeze(displacement_field[..., 0]))
        dv_dx,dv_dy = np.gradient(np.squeeze(displacement_field[..., 1]))
        # print(du_dx[140,164])
    # 计算变形梯度 F
    F = np.zeros((H, W, 2, 2))
    F[..., 0, 0] = 1 + du_dx  # F11
    F[..., 0, 1] = du_dy      # F12
    F[..., 1, 0] = dv_dx      # F21
    F[..., 1, 1] = 1 + dv_dy  # F22

    # 计算 Green-Lagrange 应变张量 E
    for i in range(H):
        for j in range(W):
            if mask[i, j]==2:  # 只在左心室区域计算
                F_T = F[i, j].T  # F 的转置
                E[i, j] = 0.5 * (F_T @ F[i, j] - np.eye(2))

    return E

def cartesian_to_polar_stress(sigma_xx, sigma_yy, sigma_xy, sigma_yx, theta):
    """
    将应力从笛卡尔坐标系转换到极坐标系
    :param sigma_xx: 笛卡尔坐标系下的 xx 应力分量
    :param sigma_yy: 笛卡尔坐标系下的 yy 应力分量
    :param sigma_xy: 笛卡尔坐标系下的 xy 应力分量
    :param theta: 极坐标系下的角度（弧度）
    :return: 极坐标系下的应力分量 (sigma_rr, sigma_theta_theta, sigma_r_theta)
    """
    cos_theta = np.cos(np.deg2rad(theta))#np.cos(theta)
    sin_theta = np.sin(np.deg2rad(theta))#np.sin(theta)

    # 计算极坐标系下的应力分量
    sigma_rr = (sigma_xx * cos_theta**2 + 
                sigma_yy * sin_theta**2 + 
                sigma_xy * sin_theta * cos_theta + 
                sigma_yx * sin_theta * cos_theta)

    sigma_theta_theta = (sigma_xx * sin_theta**2 + 
                         sigma_yy * cos_theta**2 - 
                         sigma_xy * sin_theta * cos_theta -
                         sigma_yx * sin_theta * cos_theta)

    sigma_r_theta = ((sigma_yy - sigma_xx) * sin_theta * cos_theta + 
                     sigma_xy * (cos_theta**2 - sin_theta**2))

    return sigma_rr, sigma_theta_theta, sigma_r_theta

def transform_stress_field(sigma_xx_field, sigma_yy_field, 
                           sigma_xy_field, sigma_yx_field,theta_field):
    """
    将整个应力场从笛卡尔坐标系转换到极坐标系
    :param sigma_xx_field: 笛卡尔坐标系下的 xx 应力场
    :param sigma_yy_field: 笛卡尔坐标系下的 yy 应力场
    :param sigma_xy_field: 笛卡尔坐标系下的 xy 应力场
    :param theta_field: 极坐标系下的角度场（弧度）
    :return: 极坐标系下的应力场 (sigma_rr_field, sigma_theta_theta_field, sigma_r_theta_field)
    """
    sigma_rr_field = np.zeros_like(sigma_xx_field)
    sigma_theta_theta_field = np.zeros_like(sigma_xx_field)
    sigma_r_theta_field = np.zeros_like(sigma_xx_field)

    for i in range(sigma_xx_field.shape[0]):
        for j in range(sigma_xx_field.shape[1]):
            sigma_rr_field[i, j], sigma_theta_theta_field[i, j], \
                sigma_r_theta_field[i, j] = cartesian_to_polar_stress(
                sigma_xx_field[i, j], sigma_yy_field[i, j], 
                sigma_xy_field[i, j], sigma_yx_field[i,j], theta_field[i, j]
            )

    return sigma_rr_field, sigma_theta_theta_field, sigma_r_theta_field

def calculate_theta_field(mask):
    """
    计算每个像素相对于质心的角度场
    :param mask: 左心室的掩码，形状为 [H, W]
    :return: 角度场 theta_field，形状为 [H, W]
    """
    H, W = mask.shape

    # 计算质心
    yc, xc = center_of_mass(mask==1)
    # 初始化角度场
    theta_field = np.zeros((H, W))

    # 计算每个像素的角度
    for i in range(H):
        for j in range(W):
            if mask[i, j]==2:  # 只在左心室区域计算
                dx = j - xc
                dy = i - yc
                theta_field[i, j] = np.rad2deg(np.arctan2(dx, dy)) #+ 180

    return theta_field

def roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)

def roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    #print(nx,ny)
    return roll(x,  int(nx//2-cx), int(ny//2-cy))

from pathlib import Path

checkpoint_root = Path('/home/zzx/Cardiac_Function_Analysis/CardiacAI/checkpoints')
path_to_registration_model = checkpoint_root/'strain'/'checkpoint-299.pt'
path_to_segmentation_model = checkpoint_root/'seg3d'/'checkpoint-299.pt'
path_to_landmark_model = checkpoint_root/'landmark'/'best_checkpoint.pt'

device = 'cuda:0'
cardiac_regis = CardiacRegistrator(path_to_registration_model, device, target_size=(128, 128))
segmentation_api = CardiacSaxSegmentation(path_to_segmentation_model, device)
landmark_api = CardiacLandmark(path_to_landmark_model, device)

def calculate_angle(center, point):
    """计算点相对于中心点的角度（0-360度）"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return np.degrees(np.arctan2(dy, dx)) % 360


def create_sector_labels(image_shape, center, p1, p2, six=True):
    """根据中心点和两条基准线生成4/6扇形标签图。"""
    h, w = image_shape
    y_coords, x_coords = np.indices((h, w))
    x_coords = x_coords - center[0]
    y_coords = y_coords - center[1]
    angles = np.degrees(np.arctan2(y_coords, x_coords)) % 360

    angle1 = calculate_angle(center, p1)
    angle2 = calculate_angle(center, p2)
    if (angle2 - angle1) % 360 > 180:
        angle1, angle2 = angle2, angle1

    if six:
        sector_edges = np.linspace(angle1, angle1 + 360, 7) % 360
        sector_edges[1] = (angle1 + (angle2 - angle1) / 2) % 360
        sector_edges[2] = angle2
        sector_edges[3] = (angle2 + (360 - (angle2 - angle1)) / 4) % 360
        sector_edges[4] = (angle2 + (360 - (angle2 - angle1)) / 2) % 360
        sector_edges[5] = (angle2 + 3 * (360 - (angle2 - angle1)) / 4) % 360
        labels = [1, 2, 3, 4, 5, 6]
    else:
        sector_edges = np.linspace(angle1, angle1 + 360, 5) % 360
        sector_edges[1] = angle2
        sector_edges[2] = (angle2 + (360 - (angle2 - angle1)) / 3) % 360
        sector_edges[3] = (angle2 + 2 * (360 - (angle2 - angle1)) / 3) % 360
        labels = [1, 2, 3, 4]

    label_map = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(labels)):
        start = sector_edges[i]
        end = sector_edges[i + 1]
        if start < end:
            in_sector = (angles >= start) & (angles < end)
        else:
            in_sector = (angles >= start) | (angles < end)
        label_map[in_sector] = labels[i]

    label_map[int(np.round(center[1])), int(np.round(center[0]))] = labels[0]
    return label_map


def build_aha16_label_map(mask, center, p1, p2):
    """构建AHA 16分段标签图（外圈6 + 中圈6 + 内圈4）。"""
    h, w = mask.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    myocardium = mask == 2
    if not np.any(myocardium):
        raise ValueError('分割结果中没有心肌区域(label=2)，无法进行16分段统计。')

    r_vals = r[myocardium]
    r_inner, r_outer = np.percentile(r_vals, 5), np.percentile(r_vals, 95)
    t1 = r_inner + (r_outer - r_inner) / 3.0
    t2 = r_inner + 2.0 * (r_outer - r_inner) / 3.0

    sector6 = create_sector_labels(mask.shape, center, p1, p2, six=True)
    sector4 = create_sector_labels(mask.shape, center, p1, p2, six=False)

    aha_map = np.zeros((h, w), dtype=np.uint8)
    outer_ring = myocardium & (r >= t2) & (r <= r_outer)
    middle_ring = myocardium & (r >= t1) & (r < t2)
    inner_ring = myocardium & (r >= r_inner) & (r < t1)

    aha_map[outer_ring] = sector6[outer_ring]
    aha_map[middle_ring] = sector6[middle_ring] + 6
    aha_map[inner_ring] = sector4[inner_ring] + 12
    return aha_map


def compute_16_segment_means(value_field, aha16_map):
    """统计每个AHA分段的均值和像素数。"""
    means, counts = [], []
    for seg_id in range(1, 17):
        seg_vals = value_field[aha16_map == seg_id]
        seg_vals = seg_vals[np.isfinite(seg_vals)]
        counts.append(int(seg_vals.size))
        means.append(float(np.nanmean(seg_vals)) if seg_vals.size > 0 else np.nan)
    return np.array(means, dtype=np.float32), np.array(counts, dtype=np.int32)


def split_depth_groups(num_slices):
    """按深度三等分切片索引：基底/中间/心尖。"""
    idx = np.arange(num_slices)
    groups = np.array_split(idx, 3)
    return groups[0], groups[1], groups[2]


def build_slice_segment_map(mask_slice, center, p1, p2, group_name):
    """为单层切片构建分段标签图，并映射到全局16段编号。"""
    if group_name == 'basal':
        sector_map = create_sector_labels(mask_slice.shape, center, p1, p2, six=True)   # 1..6
        global_offset = 0
    elif group_name == 'mid':
        sector_map = create_sector_labels(mask_slice.shape, center, p1, p2, six=True)   # 1..6
        global_offset = 6
    else:  # apical
        sector_map = create_sector_labels(mask_slice.shape, center, p1, p2, six=False)  # 1..4
        global_offset = 12

    seg_map = np.zeros_like(mask_slice, dtype=np.uint8)
    myocardium = (mask_slice == 2)
    seg_map[myocardium] = sector_map[myocardium] + global_offset
    return seg_map


def decode_landmarks_from_slice(landmark_api, img_slice, device):
    """对单层图像推理landmark并返回 (p1, p2)。"""
    img_slice = img_slice.astype(np.float32)
    img_slice = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-6)
    img_tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    img_tensor = F.interpolate(img_tensor, size=(160, 160), mode='bilinear', align_corners=False)

    with torch.no_grad():
        landmark_pred = landmark_api.landmark_model(img_tensor)
        if isinstance(landmark_pred, list):
            landmark_pred = landmark_pred[-1]

    heatmap = landmark_pred.squeeze().detach().cpu().numpy()  # [2,160,160]
    if heatmap.ndim != 3:
        raise RuntimeError(f'Unexpected landmark heatmap shape: {heatmap.shape}')

    coords = []
    for k in range(heatmap.shape[0]):
        flat_idx = np.argmax(heatmap[k])
        y, x = np.unravel_index(flat_idx, heatmap[k].shape)
        coords.append([x, y])  # (x, y)
    coords = np.array(coords, dtype=np.float32)  # [2,2]

    h, w = img_slice.shape
    landmark_xy = np.zeros_like(coords, dtype=np.float32)
    landmark_xy[:, 0] = coords[:, 0] * (w / 160.0)  # x
    landmark_xy[:, 1] = coords[:, 1] * (h / 160.0)  # y
    p1 = (float(landmark_xy[0, 0]), float(landmark_xy[0, 1]))
    p2 = (float(landmark_xy[1, 0]), float(landmark_xy[1, 1]))
    return p1, p2


def compute_depth_segment_maps(mask_volume, moved_img_volume, landmark_api, device):
    """构建每层的16段标签图（按深度分层+层内角度分段）。"""
    d = mask_volume.shape[0]
    slice_maps = {}
    slice_refs = {}

    basal_idx, mid_idx, apical_idx = split_depth_groups(d)
    group_of = {}
    for z in basal_idx:
        group_of[int(z)] = 'basal'
    for z in mid_idx:
        group_of[int(z)] = 'mid'
    for z in apical_idx:
        group_of[int(z)] = 'apical'

    for z in range(d):
        mask_slice = mask_volume[z]
        if not np.any(mask_slice == 2) or not np.any(mask_slice == 1):
            continue

        cy, cx = center_of_mass(mask_slice == 1)
        center = (float(cx), float(cy))
        p1, p2 = decode_landmarks_from_slice(landmark_api, moved_img_volume[z], device)

        seg_map = build_slice_segment_map(mask_slice, center, p1, p2, group_of[z])
        slice_maps[z] = seg_map
        slice_refs[z] = (center, p1, p2)

    return (basal_idx, mid_idx, apical_idx), slice_maps, slice_refs


def aggregate_16_segment_values(value_volume, slice_maps):
    """基于预先构建的16段标签图，聚合每段数值均值和像素计数。"""
    seg_values = {seg_id: [] for seg_id in range(1, 17)}
    seg_counts = np.zeros(16, dtype=np.int32)

    for z, seg_map in slice_maps.items():
        vals = value_volume[z]
        for seg_id in range(1, 17):
            seg_vals = vals[seg_map == seg_id]
            seg_vals = seg_vals[np.isfinite(seg_vals)]
            if seg_vals.size > 0:
                seg_values[seg_id].append(seg_vals)
                seg_counts[seg_id - 1] += int(seg_vals.size)

    means = np.zeros(16, dtype=np.float32)
    means[:] = np.nan
    for seg_id in range(1, 17):
        if len(seg_values[seg_id]) > 0:
            means[seg_id - 1] = float(np.nanmean(np.concatenate(seg_values[seg_id])))

    return means, seg_counts


def visualize_segmentation_reference(image, aha16_map, center, p1, p2, save_path=None):
    """可视化landmark与16分段标签。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(image, cmap='gray')
    ax1.plot(center[0], center[1], 'ro', markersize=8, label='Center')
    ax1.plot(p1[0], p1[1], 'yo', markersize=8, label='Landmark 1')
    ax1.plot(p2[0], p2[1], 'go', markersize=8, label='Landmark 2')
    ax1.plot([center[0], p1[0]], [center[1], p1[1]], 'y-')
    ax1.plot([center[0], p2[0]], [center[1], p2[1]], 'g-')
    ax1.legend()
    ax1.set_title('Landmark Reference')

    im = ax2.imshow(aha16_map, cmap='tab20', vmin=0, vmax=16)
    plt.colorbar(im, ax=ax2)
    ax2.set_title('AHA 16-Segment Label Map')
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def create_custom_bullseye(values, title="Custom Bullseye", center_radius=0.2, gap_width=0.02, save_path=None):
    """绘制16分段牛眼图（外圈6 + 中圈6 + 内圈4）。"""
    if len(values) != 16:
        raise ValueError("需要16个数值 (外圈6 + 中圈6 + 内圈4)")

    values = np.asarray(values, dtype=np.float32)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4575b4', '#ffffbf', '#d73027'])

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError('输入分段值全为NaN，无法绘制牛眼图。')

    vmin, vmax = np.nanmin(values), np.nanmax(values)
    norm_values = (values - vmin) / (vmax - vmin + 1e-10)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))

    # 注意：半径是从中心向外累加绘制，因此 layers 顺序是“内圈 -> 外圈”
    # 映射目标：外圈=基底(1-6, start_idx=0)，中圈=中间层(7-12, start_idx=6)，内圈=心尖(13-16, start_idx=12)
    layers = [
        {'sectors': 4, 'width': 0.20, 'start_idx': 12},  # inner: apical 13-16
        {'sectors': 6, 'width': 0.22, 'start_idx': 6},   # middle: mid 7-12
        {'sectors': 6, 'width': 0.30, 'start_idx': 0}    # outer: basal 1-6
    ]

    current_radius = center_radius
    for layer_idx, layer in enumerate(layers):
        sectors = layer['sectors']
        layer_width = layer['width']

        for sector in range(sectors):
            theta_start = sector * (2 * np.pi / sectors) + gap_width * 0.5
            theta_end = (sector + 1) * (2 * np.pi / sectors) - gap_width * 0.5

            r_inner = current_radius + (gap_width if layer_idx > 0 else 0)
            r_outer = r_inner + layer_width

            idx = layer['start_idx'] + sector
            color = '#d9d9d9' if np.isnan(values[idx]) else cmap(norm_values[idx])

            ax.fill_between(
                np.linspace(theta_start, theta_end, 100),
                r_inner,
                r_outer,
                color=color,
                alpha=0.90,
                edgecolor='white',
                linewidth=1.2
            )

            label_radius = (r_inner + r_outer) / 2
            text_value = 'NA' if np.isnan(values[idx]) else f"{values[idx]:.2f}"
            ax.text(
                (theta_start + theta_end) / 2,
                label_radius,
                text_value,
                ha='center',
                va='center',
                fontsize=11,
                color='black'
            )

        current_radius += layer_width + gap_width

    ax.add_patch(plt.Circle((0, 0), center_radius, color='white', transform=ax.transData._b))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Strain', fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    plt.title(title, fontsize=15, pad=25)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

import torch
import torch.nn.functional as F
import SimpleITK as sitk

# 1) 准备病例与输入路径
pat_id = 61
root_p = '/home/zzx/data/ACDC/training'
path_to_ed_image = Path(f'{root_p}/patient{pat_id:03d}/patient{pat_id:03d}_frame01.nii.gz')
path_to_es_image = Path(f'{root_p}/patient{pat_id:03d}/patient{pat_id:03d}_frame10.nii.gz')
output_dir = Path('/home/zzx/Cardiac_Function_Analysis/Myocardial_Strain/outputs') / f'patient{pat_id:03d}'
output_dir.mkdir(parents=True, exist_ok=True)
print(path_to_ed_image, path_to_es_image)
# 2) 分割：使用模型预测ED/ES掩膜，作为后续配准与应变计算的默认输入
seg_ed_mask = segmentation_api.process(str(path_to_ed_image))
seg_es_mask = segmentation_api.process(str(path_to_es_image))

# 3) 配准：使用分割掩膜辅助配准中心化
# 返回形状:
# deformation_field: [D, 2, H, W]
# moved_img: [D, H, W]
# ed_mask_reg: [D, H, W]
deformation_field, moved_img, moved_mask, ed_mask_reg = cardiac_regis.process(
    path_to_ed_image,
    path_to_es_image,
    seg_ed_mask,
    seg_es_mask,
    center=True
)

# 4) 逐层计算应变场（用于3D分层分段统计）
d = deformation_field.shape[0]
sigma_rr_volume = np.zeros((d, moved_img.shape[1], moved_img.shape[2]), dtype=np.float32)
sigma_cc_volume = np.zeros_like(sigma_rr_volume)
sigma_rr_volume[:] = np.nan
sigma_cc_volume[:] = np.nan

for z in range(d):
    mask_slice_z = ed_mask_reg[z]
    if not np.any(mask_slice_z == 2):
        continue
    displacement_field_z = deformation_field[z].transpose((1, 2, 0))  # [H,W,2]
    E = compute_green_lagrange_strain(displacement_field_z, mask_slice_z, gaussian=False, sigma=3)
    sigma_xx_field = E[..., 0, 0]
    sigma_yy_field = E[..., 1, 1]
    sigma_xy_field = E[..., 0, 1]
    sigma_yx_field = E[..., 1, 0]
    theta_field = calculate_theta_field(mask_slice_z)
    sigma_rr_field, sigma_theta_theta_field, _ = transform_stress_field(
        sigma_xx_field, sigma_yy_field, sigma_xy_field, sigma_yx_field, theta_field
    )
    sigma_rr_volume[z] = sigma_rr_field
    sigma_cc_volume[z] = sigma_theta_theta_field

# 5) 16分段统计：按深度三等分(基底/中间/心尖) + 层内角度分段(6/6/4)
depth_groups, slice_maps, slice_refs = compute_depth_segment_maps(
    ed_mask_reg, moved_img, landmark_api, device
)
rr_values_16, rr_counts_16 = aggregate_16_segment_values(sigma_rr_volume, slice_maps)
cc_values_16, cc_counts_16 = aggregate_16_segment_values(sigma_cc_volume, slice_maps)

print('depth groups (basal/mid/apical):', [list(g) for g in depth_groups])

print('RR segment counts:', rr_counts_16)
print('CC segment counts:', cc_counts_16)
print('RR empty segments:', np.where(rr_counts_16 == 0)[0] + 1)
print('CC empty segments:', np.where(cc_counts_16 == 0)[0] + 1)
print('RR value range:', np.nanmin(rr_values_16), np.nanmax(rr_values_16))
print('CC value range:', np.nanmin(cc_values_16), np.nanmax(cc_values_16))

# 6) 可视化（选一个有效切片展示参考）
valid_slices = sorted(slice_maps.keys())
slice_idx = valid_slices[len(valid_slices) // 2] if valid_slices else 0
mask_slice = ed_mask_reg[slice_idx]
aha16_map = slice_maps[slice_idx] if slice_idx in slice_maps else np.zeros_like(mask_slice, dtype=np.uint8)
center, p1, p2 = slice_refs[slice_idx] if slice_idx in slice_refs else ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

# 7) 输出图像：landmark参考 + 两张牛眼图
visualize_segmentation_reference(
    mask_slice, aha16_map, center, p1, p2,
    save_path=output_dir / 'landmark_reference.png'
)
create_custom_bullseye(
    rr_values_16,
    title='Radial Strain Bullseye (AHA16)',
    save_path=output_dir / 'bullseye_radial.png'
)
create_custom_bullseye(
    cc_values_16,
    title='Circumferential Strain Bullseye (AHA16)',
    save_path=output_dir / 'bullseye_circumferential.png'
)
