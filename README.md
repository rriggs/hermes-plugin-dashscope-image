# hermes-plugin-dashscope-image

Image generation plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) using Alibaba Cloud DashScope (Qwen Cloud).

Brings Alibaba's Wan 2.7 image models to Hermes's `image_generate` tool. Works with both token plan keys (`sk-sp-*`) and pay-as-you-go keys (`sk-ws-*`).

## Models

| Model | Strengths |
|-------|-----------|
| `wan2.7-image` | Fast text-to-image (~5s) |
| `wan2.7-image-pro` | Higher fidelity, image-to-image editing (~10s) |

Both models are included in the Alibaba Cloud AI Token Plan subscription.

## Install

```bash
hermes plugins install rriggs/hermes-plugin-dashscope-image
hermes plugins enable dashscope
```

## Configure

Set your API key in `~/.hermes/.env` (or your profile's `.env`):

```bash
QWEN_API_KEY=***
```

Then set the provider:

```bash
hermes config set image_gen.provider dashscope
```

For PAYG users (non-token-plan), also set the base URL:

```bash
# In .env:
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com
```

The default base URL points to the Singapore token plan endpoint.

## Usage

Once configured, Hermes's `image_generate` tool routes through DashScope automatically:

- **Text-to-image**: just provide a prompt
- **Image-to-image**: provide `image_url` (auto-upgrades to wan2.7-image-pro)

Generated images are cached locally under `$HERMES_HOME/cache/images/` since DashScope OSS URLs expire after ~24 hours.

## API Details

This plugin uses the native DashScope multimodal-generation API:

```
POST {base}/api/v1/services/aigc/multimodal-generation/generation
```

The OpenAI-compatible `/images/generations` endpoint is NOT used because the token plan endpoint does not serve image models on that path.

## Requirements

- Hermes Agent v0.18+
- `QWEN_API_KEY` environment variable
- `requests` Python package (bundled with Hermes)

## License

MIT
