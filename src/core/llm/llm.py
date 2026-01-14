import os
import requests
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from src.core.schemas import LLMConfig, LLMGenerationParams
from src.utils.logger import get_logger


class LLMClient:
    """
    OpenAI-compatible API interface (используем OpenRouter).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)

        # Загружаем дефолтную конфигурацию LLM из service config (если есть)
        self.default_llm_config: Optional[LLMConfig] = None
        if hasattr(cfg, "llm"):
            try:
                self.default_llm_config = LLMConfig.model_validate(
                    OmegaConf.to_object(cfg.llm)
                )
            except Exception as e:
                self.logger.warning(f"Failed to parse default llm config: {e}")

    def agenerate(
        self, messages: List[Dict[str, str]], llm_config: Optional[LLMConfig] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Синхронный вызов OpenAI-compatible /chat/completions.
        Возвращает (text, usage).
        """
        cfg = llm_config or self.default_llm_config
        if not cfg:
            raise RuntimeError(
                "LLM config is not provided and default config is missing."
            )

        base_url = cfg.base_url or "https://openrouter.ai/api/v1"
        api_key = (
            os.getenv("OPENROUTER_API_KEY") or ""
        )  # конфиг тоже допускает api_key, но env приоритетнее
        if not api_key and hasattr(cfg, "api_key"):
            api_key = getattr(cfg, "api_key") or ""

        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        url = f"{base_url.rstrip('/')}/chat/completions"

        payload: Dict[str, Any] = {
            "model": cfg.model_name,
            "messages": messages,
        }

        params: Optional[LLMGenerationParams] = cfg.parameters
        if params:
            payload.update(
                {
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                }
            )
            if params.max_tokens is not None:
                payload["max_tokens"] = params.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # OpenRouter рекомендует добавлять Referer и HTTP-User-Agent,
        # но сделаем опционально
        if os.getenv("OPENROUTER_REFERRER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER")
        if os.getenv("OPENROUTER_AGENT"):
            headers["HTTP-User-Agent"] = os.getenv("OPENROUTER_AGENT")

        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            msg = (
                f"LLM request failed: status={response.status_code}, "
                f"body={response.text}"
            )
            raise RuntimeError(msg)

        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"LLM response has no choices: {data}")

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        usage_slim = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        return text, usage_slim
