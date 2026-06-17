"""
Medical Image Augmentation Module for MedVision-AI.

Provides a composable augmentation pipeline for medical imaging data.
Each augmentation is a callable transform; transforms can be chained using
the ``AugmentationPipeline`` builder.  All transforms accept and return
numpy arrays (HWC, uint8 or float32) and are designed to be deterministic
when a seed is set.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Configuration for medical image augmentation.

    Attributes:
        probability: Default probability of applying each augmentation.
        max_rotation_angle: Maximum rotation angle in degrees.
        brightness_range: (min_factor, max_factor) for brightness adjustment.
        contrast_range: (min_factor, max_factor) for contrast adjustment.
        noise_std: Standard deviation of Gaussian noise.
        cutout_n_holes: Number of cutout holes.
        cutout_size: Size of each cutout hole (as fraction of image dimension).
        elastic_alpha: Alpha parameter for elastic deformation.
        elastic_sigma: Sigma parameter for elastic deformation.
        seed: Random seed for reproducibility.
        interpolation_order: Order of spline interpolation (0=nearest, 1=bilinear, 3=bicubic).
    """

    probability: float = 0.5
    max_rotation_angle: float = 15.0
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.03
    cutout_n_holes: int = 1
    cutout_size: float = 0.15
    elastic_alpha: float = 34.0
    elastic_sigma: float = 4.0
    seed: Optional[int] = None
    interpolation_order: int = 1


# ---------------------------------------------------------------------------
# Base transform
# ---------------------------------------------------------------------------

class BaseTransform(ABC):
    """Abstract base class for all augmentation transforms.

    Subclasses must implement :meth:`_apply`.
    """

    def __init__(self, probability: float = 0.5, seed: Optional[int] = None) -> None:
        self.probability = probability
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the transform with the configured probability.

        Args:
            image: Input image as a numpy array (HWC or HW).

        Returns:
            The augmented image (or the original if the transform was skipped).
        """
        if self._rng.random() < self.probability:
            return self._apply(image)
        return image

    @abstractmethod
    def _apply(self, image: np.ndarray) -> np.ndarray:
        """Implement the actual augmentation logic."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability:.2f})"


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

class RandomRotate(BaseTransform):
    """Randomly rotate the image by an angle in ``[-max_angle, max_angle]``.

    Args:
        max_angle: Maximum rotation angle in degrees.
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        max_angle: float = 15.0,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.max_angle = max_angle

    def _apply(self, image: np.ndarray) -> np.ndarray:
        angle = self._rng.uniform(-self.max_angle, self.max_angle)
        return rotate_image(image, angle)


class RandomFlip(BaseTransform):
    """Randomly flip the image horizontally and/or vertically.

    Args:
        horizontal: Allow horizontal flipping.
        vertical: Allow vertical flipping.
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.horizontal = horizontal
        self.vertical = vertical

    def _apply(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        if self.horizontal and self._rng.random() < 0.5:
            result = np.fliplr(result).copy()
        if self.vertical and self._rng.random() < 0.5:
            result = np.flipud(result).copy()
        return result


class RandomBrightness(BaseTransform):
    """Randomly adjust image brightness.

    Multiplies pixel values by a factor drawn from ``brightness_range``.

    Args:
        brightness_range: (min_factor, max_factor).
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.brightness_range = brightness_range

    def _apply(self, image: np.ndarray) -> np.ndarray:
        factor = self._rng.uniform(*self.brightness_range)
        result = image.astype(np.float32) * factor
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0.0, 1.0).astype(np.float32)
        return result


class RandomContrast(BaseTransform):
    """Randomly adjust image contrast.

    Uses the formula: ``out = factor * (image - mean) + mean``.

    Args:
        contrast_range: (min_factor, max_factor).
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        contrast_range: tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.contrast_range = contrast_range

    def _apply(self, image: np.ndarray) -> np.ndarray:
        factor = self._rng.uniform(*self.contrast_range)
        img_float = image.astype(np.float32)
        mean = img_float.mean()
        result = factor * (img_float - mean) + mean
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0.0, 1.0).astype(np.float32)
        return result


class ElasticDeform(BaseTransform):
    """Apply elastic deformation to the image.

    Useful for medical images where local distortions simulate natural
    anatomical variation.

    Args:
        alpha: Scaling factor for the displacement field.
        sigma: Standard deviation for Gaussian smoothing of displacements.
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        alpha: float = 34.0,
        sigma: float = 4.0,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.alpha = alpha
        self.sigma = sigma

    def _apply(self, image: np.ndarray) -> np.ndarray:
        return elastic_deform_image(image, alpha=self.alpha, sigma=self.sigma, rng=self._rng)


class GaussianNoise(BaseTransform):
    """Add Gaussian noise to the image.

    Args:
        std: Standard deviation of the noise (relative to pixel range).
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        std: float = 0.03,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.std = std

    def _apply(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32)
        scale = 255.0 if image.dtype == np.uint8 else 1.0
        noise = np.random.default_rng(self._rng.randint(0, 2**31)).normal(
            0, self.std * scale, img_float.shape
        ).astype(np.float32)
        result = img_float + noise
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0.0, 1.0).astype(np.float32)
        return result


class Cutout(BaseTransform):
    """Apply cutout (random erasing) to the image.

    Randomly masks out one or more rectangular regions.

    Args:
        n_holes: Number of rectangular holes.
        size: Size of each hole as a fraction of the smaller image dimension.
        fill_value: Pixel value used to fill the holes.
        probability: Probability of applying the transform.
        seed: Optional random seed.
    """

    def __init__(
        self,
        n_holes: int = 1,
        size: float = 0.15,
        fill_value: float = 0.0,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.n_holes = n_holes
        self.size = size
        self.fill_value = fill_value

    def _apply(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        h, w = result.shape[:2]
        min_dim = min(h, w)
        hole_size = max(1, int(self.size * min_dim))

        for _ in range(self.n_holes):
            y = self._rng.randint(0, max(0, h - hole_size))
            x = self._rng.randint(0, max(0, w - hole_size))
            if result.ndim == 3:
                result[y : y + hole_size, x : x + hole_size, :] = self.fill_value
            else:
                result[y : y + hole_size, x : x + hole_size] = self.fill_value
        return result


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Composable augmentation pipeline.

    Transforms are applied in order.  Each transform decides independently
    whether to apply itself based on its probability.

    Example::

        pipeline = (
            AugmentationPipeline()
            .add(RandomRotate(max_angle=10, probability=0.7))
            .add(RandomFlip(probability=0.5))
            .add(GaussianNoise(std=0.02, probability=0.3))
        )
        augmented = pipeline(image)
    """

    def __init__(self) -> None:
        self._transforms: list[BaseTransform] = []

    def add(self, transform: BaseTransform) -> "AugmentationPipeline":
        """Append a transform to the pipeline.

        Args:
            transform: A :class:`BaseTransform` instance.

        Returns:
            The pipeline itself for fluent chaining.
        """
        self._transforms.append(transform)
        return self

    def remove(self, transform_type: type) -> "AugmentationPipeline":
        """Remove all transforms of *transform_type* from the pipeline.

        Args:
            transform_type: The class of transform to remove.

        Returns:
            The pipeline itself for fluent chaining.
        """
        self._transforms = [t for t in self._transforms if not isinstance(t, transform_type)]
        return self

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply all transforms in sequence.

        Args:
            image: Input image array.

        Returns:
            Augmented image.
        """
        result = image
        for transform in self._transforms:
            result = transform(result)
        return result

    def __len__(self) -> int:
        return len(self._transforms)

    def __repr__(self) -> str:
        names = [repr(t) for t in self._transforms]
        return f"AugmentationPipeline([{', '.join(names)}])"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MedicalImageAugmentation:
    """High-level medical image augmentation manager.

    Wraps individual augmentation methods and provides a default pipeline
    based on the supplied configuration.  Users can either call individual
    methods or use :meth:`apply_augmentations` to run the full pipeline.

    Args:
        config: An :class:`AugmentationConfig` instance.

    Example::

        aug = MedicalImageAugmentation(AugmentationConfig(seed=42))
        augmented = aug.apply_augmentations(image_array)
    """

    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        self._config = config or AugmentationConfig()
        seed = self._config.seed
        self._pipeline = self._build_default_pipeline(seed)
        logger.info(
            "MedicalImageAugmentation initialised (seed=%s, pipeline_len=%d)",
            seed, len(self._pipeline),
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply the full augmentation pipeline to *image*.

        Args:
            image: Input image as a numpy array (HWC, uint8 or float32).

        Returns:
            Augmented image.
        """
        return self._pipeline(image)

    def _build_default_pipeline(self, seed: Optional[int]) -> AugmentationPipeline:
        """Construct the default augmentation pipeline from the config."""
        p = self._config.probability
        return (
            AugmentationPipeline()
            .add(RandomRotate(
                max_angle=self._config.max_rotation_angle,
                probability=p,
                seed=seed,
            ))
            .add(RandomFlip(probability=p, seed=seed))
            .add(RandomBrightness(
                brightness_range=self._config.brightness_range,
                probability=p,
                seed=seed,
            ))
            .add(RandomContrast(
                contrast_range=self._config.contrast_range,
                probability=p,
                seed=seed,
            ))
            .add(ElasticDeform(
                alpha=self._config.elastic_alpha,
                sigma=self._config.elastic_sigma,
                probability=p,
                seed=seed,
            ))
            .add(GaussianNoise(
                std=self._config.noise_std,
                probability=p,
                seed=seed,
            ))
            .add(Cutout(
                n_holes=self._config.cutout_n_holes,
                size=self._config.cutout_size,
                probability=p,
                seed=seed,
            ))
        )

    # ------------------------------------------------------------------
    # Individual augmentation methods (can be called standalone)
    # ------------------------------------------------------------------

    def random_rotate(
        self,
        image: np.ndarray,
        max_angle: Optional[float] = None,
    ) -> np.ndarray:
        """Rotate the image by a random angle.

        Args:
            image: Input image.
            max_angle: Override the config default when provided.

        Returns:
            Rotated image.
        """
        angle = max_angle or self._config.max_rotation_angle
        actual_angle = np.random.uniform(-angle, angle)
        return rotate_image(image, actual_angle)

    def random_flip(self, image: np.ndarray) -> np.ndarray:
        """Randomly flip the image horizontally and/or vertically.

        Args:
            image: Input image.

        Returns:
            Flipped image.
        """
        result = image.copy()
        if np.random.random() < 0.5:
            result = np.fliplr(result).copy()
        if np.random.random() < 0.5:
            result = np.flipud(result).copy()
        return result

    def random_brightness(
        self,
        image: np.ndarray,
        brightness_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Randomly adjust brightness.

        Args:
            image: Input image.
            brightness_range: Override the config default when provided.

        Returns:
            Brightness-adjusted image.
        """
        rng = brightness_range or self._config.brightness_range
        factor = np.random.uniform(*rng)
        img_float = image.astype(np.float32) * factor
        if image.dtype == np.uint8:
            return np.clip(img_float, 0, 255).astype(np.uint8)
        return np.clip(img_float, 0.0, 1.0).astype(np.float32)

    def random_contrast(
        self,
        image: np.ndarray,
        contrast_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Randomly adjust contrast.

        Args:
            image: Input image.
            contrast_range: Override the config default when provided.

        Returns:
            Contrast-adjusted image.
        """
        rng = contrast_range or self._config.contrast_range
        factor = np.random.uniform(*rng)
        img_float = image.astype(np.float32)
        mean = img_float.mean()
        result = factor * (img_float - mean) + mean
        if image.dtype == np.uint8:
            return np.clip(result, 0, 255).astype(np.uint8)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def elastic_deform(
        self,
        image: np.ndarray,
        alpha: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Apply elastic deformation.

        Args:
            image: Input image.
            alpha: Override the config default when provided.
            sigma: Override the config default when provided.

        Returns:
            Deformed image.
        """
        a = alpha or self._config.elastic_alpha
        s = sigma or self._config.elastic_sigma
        return elastic_deform_image(image, alpha=a, sigma=s)

    def gaussian_noise(
        self,
        image: np.ndarray,
        std: Optional[float] = None,
    ) -> np.ndarray:
        """Add Gaussian noise.

        Args:
            image: Input image.
            std: Override the config default when provided.

        Returns:
            Noisy image.
        """
        s = std or self._config.noise_std
        img_float = image.astype(np.float32)
        scale = 255.0 if image.dtype == np.uint8 else 1.0
        noise = np.random.normal(0, s * scale, img_float.shape).astype(np.float32)
        result = img_float + noise
        if image.dtype == np.uint8:
            return np.clip(result, 0, 255).astype(np.uint8)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def cutout(
        self,
        image: np.ndarray,
        n_holes: Optional[int] = None,
        size: Optional[float] = None,
    ) -> np.ndarray:
        """Apply random cutout.

        Args:
            image: Input image.
            n_holes: Override the config default when provided.
            size: Override the config default when provided.

        Returns:
            Image with cutout applied.
        """
        nh = n_holes or self._config.cutout_n_holes
        sz = size or self._config.cutout_size
        transform = Cutout(n_holes=nh, size=sz, probability=1.0)
        return transform._apply(image)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by *angle* degrees around its centre.

    Uses bilinear interpolation with zero-fill for out-of-bounds pixels.

    Args:
        image: Input image (HWC or HW).
        angle: Rotation angle in degrees (counter-clockwise).

    Returns:
        Rotated image with the same shape as the input.
    """
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    radians = math.radians(angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)

    # Centre of rotation
    cx, cy = w / 2.0, h / 2.0

    # Build output coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Inverse mapping: output -> input
    x_centered = x_coords - cx
    y_centered = y_coords - cy
    src_x = cos_a * x_centered + sin_a * y_centered + cx
    src_y = -sin_a * x_centered + cos_a * y_centered + cy

    coords = [src_y.ravel(), src_x.ravel()]
    if image.ndim == 3:
        result = np.zeros_like(image)
        for ch in range(image.shape[2]):
            result[:, :, ch] = map_coordinates(
                image[:, :, ch], coords, order=1, mode="constant", cval=0.0
            ).reshape(h, w)
    else:
        result = map_coordinates(
            image, coords, order=1, mode="constant", cval=0.0
        ).reshape(h, w)

    return result


def elastic_deform_image(
    image: np.ndarray,
    alpha: float = 34.0,
    sigma: float = 4.0,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Apply elastic deformation to an image.

    Generates smooth random displacement fields and warps the image
    accordingly using spline interpolation.

    Args:
        image: Input image (HWC or HW).
        alpha: Scaling factor for displacement magnitudes.
        sigma: Gaussian smoothing sigma for the displacement field.
        rng: Optional ``random.Random`` instance for reproducibility.

    Returns:
        Deformed image with the same shape as the input.
    """
    rng = rng or random.Random()
    h, w = image.shape[:2]

    np_rng = np.random.default_rng(rng.randint(0, 2**31))

    # Random displacement fields
    dx = np_rng.random((h, w)).astype(np.float32) * 2 - 1
    dy = np_rng.random((h, w)).astype(np.float32) * 2 - 1

    # Smooth the displacement fields
    dx = gaussian_filter(dx, sigma=sigma) * alpha
    dy = gaussian_filter(dy, sigma=sigma) * alpha

    # Coordinate grids
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    indices = [
        (y + dy).ravel(),
        (x + dx).ravel(),
    ]

    if image.ndim == 3:
        result = np.zeros_like(image)
        for ch in range(image.shape[2]):
            result[:, :, ch] = map_coordinates(
                image[:, :, ch], indices, order=1, mode="reflect"
            ).reshape(h, w)
    else:
        result = map_coordinates(
            image, indices, order=1, mode="reflect"
        ).reshape(h, w)

    return result
