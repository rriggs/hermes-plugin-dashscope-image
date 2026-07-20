"""DashScope (Qwen Cloud) image generation backend.

Exposes Alibaba Cloud Model Studio's Wan image models through the
native DashScope multimodal-generation API as an ImageGenProvider.

Works with both token-plan keys (sk-sp-*) and PAYG keys (sk-ws-*).

Configuration (config.yaml):

    image_gen:
      provider: dashscope
      dashscope:
        api: https://token-plan.ap-southeast-1.maas.aliyuncs.com
        key_env: QWEN_API_KEY
        model: wan2.7-image          # optional default model

All keys are optional. Defaults:
  - api:     https://token-plan.ap-southeast-1.maas.aliyuncs.com
  - key_env: QWEN_API_KEY
  - model:   wan2.7-image

For PAYG users:
  - api:     https://dashscope-intl.aliyuncs.com
  - key_env: DASHSCOPE_API_KEY

API: POST {api}/api/v1/services/aigc/multimodal-generation/generation
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
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

# Defaults when config keys are absent.
_DEFAULT_API = "https://token-plan.ap-southeast-1.maas.aliyuncs.com"
_DEFAULT_KEY_ENV = "QWEN_API_KEY"

# Max file size for base64 inline encoding (10 MB).
_MAX_INLINE_BYTES = 10 * 1024 * 1024


def _to_data_uri(path: str) -> str:
    """Convert a local image file to a base64 data URI for the API.

    DashScope accepts data URIs in the ``image`` field of multimodal
    messages.  Local paths (absolute or relative) are detected and
    converted automatically so callers can pass file paths directly.

    Raises ``ValueError`` for missing files or unsupported sizes.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"Image file not found: {path}")
    if p.stat().st_size > _MAX_INLINE_BYTES:
        raise ValueError(
            f"Image too large for inline encoding ({p.stat().st_size} bytes, "
            f"max {_MAX_INLINE_BYTES}). Use a URL instead."
        )
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"  # safe fallback
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _normalize_source(src: str) -> str:
    """Return a URL or data URI suitable for the DashScope API.

    - http/https URLs and data URIs pass through unchanged.
    - Local file paths are converted to base64 data URIs.
    """
    if src.startswith(("http://", "https://", "data:")):
        return src
    return _to_data_uri(src)


def _load_config() -> Dict[str, Any]:
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


def _resolve_api(cfg: Dict[str, Any]) -> str:
    """Resolve the API base URL: config > default."""
    api = cfg.get("api", "")
    if isinstance(api, str) and api.strip():
        return api.strip().rstrip("/")
    return _DEFAULT_API


def _resolve_key_env(cfg: Dict[str, Any]) -> str:
    """Resolve the env var name holding the API key: config > default."""
    key_env = cfg.get("key_env", "")
    if isinstance(key_env, str) and key_env.strip():
        return key_env.strip()
    return _DEFAULT_KEY_ENV


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
        cfg = _load_config()
        key_env = _resolve_key_env(cfg)
        return bool(os.environ.get(key_env, "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return list(_MODELS)

    def default_model(self) -> Optional[str]:
        return "wan2.7-image"

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 1}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DashScope (Qwen Cloud)",
            "badge": "token-plan",
            "tag": "Wan 2.7 image models via Alibaba Cloud Model Studio",
            "env_vars": [
                {
                    "key": "QWEN_API_KEY",
                    "prompt": "Qwen Cloud API key (sk-sp-* for token plan, sk-ws-* for PAYG)",
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

        cfg = _load_config()
        key_env = _resolve_key_env(cfg)
        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            return error_response(
                error=(
                    f"{key_env} not set. Configure image_gen.dashscope.key_env "
                    f"in config.yaml and set the env var, or run `hermes tools` "
                    f"-> Image Generation -> DashScope."
                ),
                error_type="auth_required",
                provider=self.name,
                aspect_ratio=aspect,
            )

        # Model selection: explicit kwarg > config > default
        model_id = (
            kwargs.get("model")
            or cfg.get("model")
            or self.default_model()
        )

        # Determine modality from source images
        sources: List[str] = []
        if image_url:
            sources.append(_normalize_source(image_url))
        sources.extend(
            _normalize_source(s)
            for s in (normalize_reference_images(reference_image_urls) or [])
        )
        modality = "image" if sources else "text"

        # Image-to-image requires the pro model
        if modality == "image" and model_id == "wan2.7-image":
            model_id = "wan2.7-image-pro"

        size = _SIZES.get(aspect, _SIZES["square"])

        # Build the DashScope multimodal-generation request.
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

        base = _resolve_api(cfg)
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

        # Check for API-level errors
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

        # Cache the image locally -- DashScope OSS URLs are ephemeral.
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
