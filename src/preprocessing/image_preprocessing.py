"""
Image Preprocessing Module for MedVision-AI.

Provides a full image preprocessing pipeline for medical imaging data,
including resizing, normalization, denoising, contrast enhancement,
region cropping, patch extraction, CLAHE, and DICOM-specific handling.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter

logger = logging.getLogger(__name__)

# Type aliases
ImageType = Union[np.ndarray, "PIL.Image.Image", torch.Tensor]
BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ImagePreprocessConfig:
    """Configuration for the ImagePreprocessor.

    Attributes:
        target_size: Default output (height, width).
        normalization_mean: Per-channel mean for standardisation.
        normalization_std: Per-channel std for standardisation.
        denoise_method: ``"gaussian"`` or ``"median"``.
        denoise_strength: Kernel size or sigma for denoising.
        clahe_clip_limit: Clip limit for CLAHE.
        clahe_grid_size: Grid size for CLAHE.
        patch_size: Default (height, width) for patch extraction.
        patch_stride: Stride for patch extraction (defaults to patch_size).
        interpolation: Interpolation mode (``"bilinear"`` or ``"nearest"``).
        output_dtype: Output numpy dtype (``"float32"`` or ``"uint8"``).
        dicom_window_center: Default window center for DICOM windowing.
        dicom_window_width: Default window width for DICOM windowing.
    """

    target_size: tuple[int, int] = (224, 224)
    normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    denoise_method: str = "gaussian"
    denoise_strength: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    patch_size: tuple[int, int] = (64, 64)
    patch_stride: Optional[tuple[int, int]] = None
    interpolation: str = "bilinear"
    output_dtype: str = "float32"
    dicom_window_center: Optional[float] = None
    dicom_window_width: Optional[float] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Image preprocessing pipeline for medical images.

    Provides methods for resizing, normalization, denoising, contrast
    enhancement, region cropping, patch extraction, and CLAHE.  DICOM
    images are handled transparently with optional window/level adjustment.

    Args:
        config: An :class:`ImagePreprocessConfig` instance.

    Example::

        preprocessor = ImagePreprocessor(ImagePreprocessConfig(target_size=(512, 512)))
        tensor = preprocessor.preprocess(image_array)  # -> torch.Tensor [C, H, W]
    """

    def __init__(self, config: Optional[ImagePreprocessConfig] = None) -> None:
        self._config = config or ImagePreprocessConfig()
        logger.info(
            "ImagePreprocessor initialised (target_size=%s, interp=%s)",
            self._config.target_size, self._config.interpolation,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image: ImageType,
        apply_clahe: bool = True,
        denoise: bool = True,
    ) -> torch.Tensor:
        """Execute the full preprocessing pipeline on *image*.

        Pipeline order:
        1. Convert to numpy (if PIL or DICOM)
        2. Resize to target size
        3. Denoise (optional)
        4. Apply CLAHE (optional)
        5. Normalize pixel values to [0, 1]
        6. Standardise with configured mean/std
        7. Convert to torch tensor (C, H, W)

        Args:
            image: Input image (numpy array, PIL Image, or path).
            apply_clahe: Whether to apply CLAHE.
            denoise: Whether to apply denoising.

        Returns:
            A ``torch.Tensor`` of shape ``(C, H, W)``.
        """
        img = self._to_numpy(image)

        # Resize
        target_h, target_w = self._config.target_size
        img = self.resize(img, (target_h, target_w))

        # Denoise
        if denoise:
            img = self.denoise(img)

        # CLAHE
        if apply_clahe:
            img = self.apply_clahe(img)

        # Normalize to [0, 1]
        img = self.normalize(img)

        # Standardise
        img = self._standardise(img)

        # To tensor (C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3:
            img = img.transpose(2, 0, 1)

        tensor = torch.from_numpy(img.astype(np.float32))
        logger.debug("Preprocessed image -> tensor %s", tuple(tensor.shape))
        return tensor

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def resize(
        self,
        image: np.ndarray,
        size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Resize *image* to (height, width).

        Uses bilinear interpolation for float images and nearest-neighbour
        for integer masks by default.

        Args:
            image: Input image array (HWC or HW).
            size: Target (height, width). Defaults to config value.

        Returns:
            Resized image.
        """
        target_size = size or self._config.target_size
        target_h, target_w = target_size

        if image.shape[:2] == (target_h, target_w):
            return image

        # Simple area-based resampling using scipy zoom
        from scipy.ndimage import zoom as scipy_zoom

        h, w = image.shape[:2]
        zoom_h = target_h / h
        zoom_w = target_w / w

        if image.ndim == 2:
            order = 1 if self._config.interpolation == "bilinear" else 0
            return scipy_zoom(image, (zoom_h, zoom_w), order=order)

        order = 1 if self._config.interpolation == "bilinear" else 0
        result = scipy_zoom(image, (zoom_h, zoom_w, 1), order=order)
        return result

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to the [0, 1] range.

        Handles both uint8 and float input images.

        Args:
            image: Input image.

        Returns:
            Float32 image in [0, 1].
        """
        img = image.astype(np.float64)

        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        return img.astype(np.float32)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Denoise the image using the configured method.

        Args:
            image: Input image.

        Returns:
            Denoised image.
        """
        if self._config.denoise_method == "gaussian":
            sigma = self._config.denoise_strength
            if image.ndim == 2:
                return gaussian_filter(image, sigma=sigma).astype(image.dtype)
            # Per-channel Gaussian
            result = np.zeros_like(image)
            for ch in range(image.shape[2]):
                result[:, :, ch] = gaussian_filter(image[:, :, ch], sigma=sigma)
            return result.astype(image.dtype)

        elif self._config.denoise_method == "median":
            size = int(self._config.denoise_strength)
            size = max(1, size | 1)  # ensure odd
            if image.ndim == 2:
                return median_filter(image, size=size).astype(image.dtype)
            result = np.zeros_like(image)
            for ch in range(image.shape[2]):
                result[:, :, ch] = median_filter(image[:, :, ch], size=size)
            return result.astype(image.dtype)

        else:
            logger.warning("Unknown denoise method: %s", self._config.denoise_method)
            return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using histogram stretching.

        Clips the lower and upper percentiles and rescales to [0, 1].

        Args:
            image: Input image.

        Returns:
            Contrast-enhanced image.
        """
        img = image.astype(np.float64)
        p2, p98 = np.percentile(img, (2, 98))
        if p98 - p2 > 0:
            img = np.clip((img - p2) / (p98 - p2), 0, 1)
        return img.astype(np.float32)

    def crop_to_region(
        self,
        image: np.ndarray,
        bbox: BBox,
    ) -> np.ndarray:
        """Crop the image to the bounding box region.

        Args:
            image: Input image (HWC or HW).
            bbox: ``(x_min, y_min, x_max, y_max)`` bounding box.

        Returns:
            Cropped image.
        """
        x_min, y_min, x_max, y_max = bbox
        h, w = image.shape[:2]

        # Clamp to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            logger.warning("Invalid bounding box %s for image of size (%d, %d)", bbox, h, w)
            return image

        if image.ndim == 3:
            return image[y_min:y_max, x_min:x_max, :]
        return image[y_min:y_max, x_min:x_max]

    def extract_patches(
        self,
        image: np.ndarray,
        patch_size: Optional[tuple[int, int]] = None,
        stride: Optional[tuple[int, int]] = None,
    ) -> list[np.ndarray]:
        """Extract overlapping or non-overlapping patches from the image.

        Args:
            image: Input image (HWC or HW).
            patch_size: (height, width) of each patch.
            stride: (row_stride, col_stride). Defaults to patch_size.

        Returns:
            A list of numpy arrays, each a patch.
        """
        ps = patch_size or self._config.patch_size
        st = stride or self._config.patch_stride or ps

        ph, pw = ps
        sh, sw = st
        h, w = image.shape[:2]

        patches: list[np.ndarray] = []
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                if image.ndim == 3:
                    patches.append(image[y : y + ph, x : x + pw, :])
                else:
                    patches.append(image[y : y + ph, x : x + pw])

        logger.debug("Extracted %d patches (size=%s, stride=%s)", len(patches), ps, st)
        return patches

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).

        Uses a pure-numpy implementation that divides the image into tiles,
        computes clipped histograms per tile, and bilinearly interpolates
        the resulting mapping.

        Args:
            image: Input image (HW or HWC). If multi-channel, CLAHE is
                applied to the luminance channel only.

        Returns:
            CLAHE-enhanced image.
        """
        img = image.astype(np.float64)

        # If multi-channel, convert to luminance, apply CLAHE, merge back
        if img.ndim == 3:
            # Simple luminance: average of channels
            luminance = img.mean(axis=2)
            enhanced_lum = _clahe_core(
                luminance,
                clip_limit=self._config.clahe_clip_limit,
                grid_size=self._config.clahe_grid_size,
            )
            # Scale each channel proportionally
            scale = np.where(luminance > 0, enhanced_lum / (luminance + 1e-10), 1.0)
            result = img * scale[:, :, np.newaxis]
            return np.clip(result, 0, img.max()).astype(image.dtype)

        result = _clahe_core(
            img,
            clip_limit=self._config.clahe_clip_limit,
            grid_size=self._config.clahe_grid_size,
        )
        return result.astype(image.dtype)

    # ------------------------------------------------------------------
    # DICOM support
    # ------------------------------------------------------------------

    def load_dicom(
        self,
        path: Union[str, Path],
        apply_windowing: bool = True,
    ) -> np.ndarray:
        """Load a DICOM file and return a numpy pixel array.

        Optionally applies window/level adjustment for CT images.

        Args:
            path: Path to the .dcm file.
            apply_windowing: Whether to apply window/level adjustment.

        Returns:
            Pixel data as a numpy array.

        Raises:
            ImportError: If ``pydicom`` is not installed.
        """
        try:
            import pydicom  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pydicom is required for DICOM loading. "
                "Install it with: pip install pydicom"
            ) from exc

        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float64)

        # Rescale using slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept

        if apply_windowing:
            pixel_array = self._apply_windowing(pixel_array, ds)

        return pixel_array

    def _apply_windowing(self, pixel_array: np.ndarray, ds: Any) -> np.ndarray:
        """Apply DICOM window/level (center/width) transformation.

        If the config specifies center/width those values are used;
        otherwise the values from the DICOM metadata are used.
        """
        center = self._config.dicom_window_center
        width = self._config.dicom_window_width

        if center is None:
            center = float(getattr(ds, "WindowCenter", pixel_array.mean()))
        if width is None:
            width = float(getattr(ds, "WindowWidth", pixel_array.max() - pixel_array.min()))

        # Handle WindowCenter being a list (some DICOM files)
        if isinstance(center, (list, tuple)):
            center = float(center[0])
        if isinstance(width, (list, tuple)):
            width = float(width[0])

        if width <= 0:
            return pixel_array

        lower = center - width / 2.0
        upper = center + width / 2.0
        windowed = np.clip(pixel_array, lower, upper)
        windowed = (windowed - lower) / (upper - lower)
        return windowed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(image: ImageType) -> np.ndarray:
        """Convert various image types to a numpy array."""
        if isinstance(image, np.ndarray):
            return image.copy()
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            return arr
        if isinstance(image, (str, Path)):
            from PIL import Image  # type: ignore[import-untyped]
            pil = Image.open(str(image)).convert("RGB")
            return np.array(pil)
        if hasattr(image, "convert"):
            # PIL Image
            pil = image.convert("RGB")  # type: ignore[union-attr]
            return np.array(pil)
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def _standardise(self, image: np.ndarray) -> np.ndarray:
        """Standardise with configured mean and std (per-channel)."""
        if image.ndim == 2:
            mean = self._config.normalization_mean[0]
            std = self._config.normalization_std[0]
            return (image - mean) / (std + 1e-8)

        n_channels = image.shape[2] if image.ndim == 3 else 1
        means = self._config.normalization_mean
        stds = self._config.normalization_std

        # Broadcast if fewer values than channels
        while len(means) < n_channels:
            means = means + means
        while len(stds) < n_channels:
            stds = stds + stds

        means = means[:n_channels]
        stds = stds[:n_channels]

        result = image.astype(np.float32)
        for ch in range(n_channels):
            result[:, :, ch] = (result[:, :, ch] - means[ch]) / (stds[ch] + 1e-8)
        return result


# ---------------------------------------------------------------------------
# CLAHE core implementation (pure numpy)
# ---------------------------------------------------------------------------

def _clahe_core(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a single-channel float image.

    Args:
        image: 2-D float array with values in [0, 1] or any range.
        clip_limit: Histogram clip limit.
        grid_size: Number of tiles in (rows, cols).

    Returns:
        Equalised image as float64.
    """
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min <= 0:
        return img

    # Normalise to [0, 1]
    img_norm = (img - img_min) / (img_max - img_min)

    h, w = img_norm.shape
    grid_h, grid_w = grid_size

    tile_h = max(1, h // grid_h)
    tile_w = max(1, w // grid_w)
    n_bins = 256

    # Compute mapping for each tile
    mappings: list[list[np.ndarray]] = []
    for ty in range(grid_h):
        row_mappings: list[np.ndarray] = []
        for tx in range(grid_w):
            y0 = ty * tile_h
            x0 = tx * tile_w
            y1 = min(y0 + tile_h, h)
            x1 = min(x0 + tile_w, w)
            tile = img_norm[y0:y1, x0:x1]

            # Histogram
            hist, _ = np.histogram(tile.ravel(), bins=n_bins, range=(0, 1))
            # Clip
            clip_val = max(1, int(clip_limit * tile.size / n_bins))
            excess = max(0, hist.sum() - clip_val * n_bins)
            hist = np.clip(hist, 0, clip_val)
            # Redistribute excess
            redistrib = excess // n_bins
            hist += redistrib

            # CDF
            cdf = hist.cumsum()
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            mapping = cdf
            row_mappings.append(mapping)
        mappings.append(row_mappings)

    # Bilinear interpolation
    result = np.zeros_like(img_norm)
    for y in range(h):
        for x in range(w):
            # Fractional tile position
            fy = (y + 0.5) / tile_h - 0.5
            fx = (x + 0.5) / tile_w - 0.5

            ty1 = max(0, int(math.floor(fy)))
            ty2 = min(grid_h - 1, ty1 + 1)
            tx1 = max(0, int(math.floor(fx)))
            tx2 = min(grid_w - 1, tx1 + 1)

            wy = fy - ty1
            wx = fx - tx1
            wy = max(0.0, min(1.0, wy))
            wx = max(0.0, min(1.0, wx))

            bin_idx = min(int(img_norm[y, x] * (n_bins - 1)), n_bins - 1)

            v11 = mappings[ty1][tx1][bin_idx]
            v12 = mappings[ty1][tx2][bin_idx]
            v21 = mappings[ty2][tx1][bin_idx]
            v22 = mappings[ty2][tx2][bin_idx]

            val = (
                v11 * (1 - wy) * (1 - wx)
                + v12 * (1 - wy) * wx
                + v21 * wy * (1 - wx)
                + v22 * wy * wx
            )
            result[y, x] = val

    return result * (img_max - img_min) + img_min
