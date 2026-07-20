"""DashScope (Qwen Cloud) image generation backend.

Exposes Alibaba Cloud Model Studio's Wan image models through the
native DashScope multimodal-generation API as an ImageGenProvider.

Works with both token-plan keys (sk-sp-*) and PAYG keys (sk-ws-*).
Set DASHSCOPE_BASE_URL to override the endpoint (defaults to the
Singapore token-plan endpoint).

Models on the token plan:
  - wan2.7-image:      Fast text-to-image
  - wan2.7-image-pro:  Higher fidelity, supports image-to-image editing

API: POST {base}/api/v1/services/aigc/multimodal-generation/generation
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# DashScope uses W*H format for size specification.
_SIZES = {
    "landscape": "1280*720",
    "square": "1024*1024",
    "portrait": "720*1280",
}

_MODELS = [
    {
        "id": "wan2.7-image",
        "display": "Wan 2.7 Image",
        "speed": "~5s",
        "strengths": "Fast text-to-image generation",
        "price": "Token plan included",
    },
    {
        "id": "wan2.7-image-pro",
        "display": "Wan 2.7 Image Pro",
        "speed": "~10s",
        "strengths": "Higher fidelity, interactive editing, image-to-image",
        "price": "Token plan included",
    },
]

# Default base URL -- Singapore token plan endpoint.
# Override with DASHSCOPE_BASE_URL for PAYG (https://dashscope-intl.aliyuncs.com)
# or China region (https://dashscope.aliyuncs.com).
_DEFAULT_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com"


def _base_url() -> str:
    return os.environ.get("DASHSCOPE_BASE_URL", "").strip() or _DEFAULT_BASE_URL


def _load_dashscope_image_config() -> Dict[str, Any]:
    """Read ``image_gen.dashscope`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        ds = section.get("dashscope") if isinstance(section, dict) else None
        return ds if isinstance(ds, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.dashscope config: %s", exc)
        return {}


class DashScopeImageGenProvider(ImageGenProvider):
    """Alibaba Cloud DashScope / Qwen Cloud image generation backend.

    Uses the native DashScope multimodal-generation API (not the
    OpenAI-compatible endpoint, which does not serve image models
    on the token plan).
    """

    @property
    def name(self) -> str:
        return "dashscope"

    @property
    def display_name(self) -> str:
        return "DashScope (Qwen Cloud)"

    def is_available(self) -> bool:
        return bool(os.environ.get("QWEN_API_KEY", "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return list(_MODELS)

    def default_model(self) -> Optional[str]:
        return "wan2.7-image"

    def capabilities(self) -> Dict[str, Any]:
        # wan2.7-image-pro supports image-to-image editing via
        # reference images in the messages content array.
        return {"modalities": ["text", "image"], "max_reference_images": 1}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DashScope (Qwen Cloud)",
            "badge": "token-plan",
            "tag": "Wan 2.7 image models via Alibaba Cloud Model Studio",
            "env_vars": [
                {
                    "key": "QWEN_API_KEY",
                    "prompt": "Qwen Cloud API key (sk-sp-* for token plan)",
                    "url": "https://modelstudio.console.alibabacloud.com/",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("QWEN_API_KEY", "").strip()
        if not api_key:
            return error_response(
                error=(
                    "QWEN_API_KEY not set. Run `hermes tools` -> Image "
                    "Generation -> DashScope to configure."
                ),
                error_type="auth_required",
                provider=self.name,
                aspect_ratio=aspect,
            )

        # Model selection: explicit kwarg > config > default
        ds_cfg = _load_dashscope_image_config()
        model_id = (
            kwargs.get("model")
            or ds_cfg.get("model")
            or self.default_model()
        )

        # Determine modality from source images
        sources: List[str] = []
        if image_url:
            sources.append(image_url)
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        modality = "image" if sources else "text"

        # Image-to-image requires the pro model
        if modality == "image" and model_id == "wan2.7-image":
            model_id = "wan2.7-image-pro"

        size = _SIZES.get(aspect, _SIZES["square"])

        # Build the DashScope multimodal-generation request.
        # Content array: source images first, then the text prompt.
        content: List[Dict[str, str]] = []
        for src in sources:
            content.append({"image": src})
        content.append({"text": prompt})

        payload = {
            "model": model_id,
            "input": {
                "messages": [
                    {"role": "user", "content": content}
                ]
            },
            "parameters": {"size": size},
        }

        base = _base_url().rstrip("/")
        url = f"{base}/api/v1/services/aigc/multimodal-generation/generation"

        try:
            import requests

            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("DashScope image generation failed", exc_info=True)
            return error_response(
                error=f"DashScope API request failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Check for API-level errors (code field present and non-empty)
        if data.get("code"):
            return error_response(
                error=f"DashScope error: {data['code']}: {data.get('message', '')}",
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Extract image URL from response.
        # Shape: output.choices[0].message.content[] where each item
        # is {"type": "image", "image": "<url>"} or just {"image": "<url>"}.
        image_ref: Optional[str] = None
        try:
            choices = data["output"]["choices"]
            msg_content = choices[0]["message"]["content"]
            for item in msg_content:
                if isinstance(item, dict):
                    image_ref = item.get("image") or item.get("url")
                    if image_ref:
                        break
        except (KeyError, IndexError, TypeError) as exc:
            return error_response(
                error=f"Unexpected DashScope response structure: {exc}",
                error_type="parse_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not image_ref:
            return error_response(
                error="DashScope response contained no image URL",
                error_type="empty_response",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Cache the image locally -- DashScope OSS URLs are ephemeral
        # (signed with ~24h expiry).
        short = model_id.replace(".", "_")
        try:
            saved_path = save_url_image(image_ref, prefix=f"dashscope_{short}")
            image_ref = str(saved_path)
        except Exception as exc:
            logger.debug(
                "DashScope: caching image URL failed (%s); returning URL", exc
            )

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality=modality,
            extra={"size": size},
        )


def register(ctx) -> None:
    """Plugin entry point -- wire DashScopeImageGenProvider into the registry."""
    ctx.register_image_gen_provider(DashScopeImageGenProvider())
