import os
from typing import Optional

from hello_agents import HelloAgentsLLM
from openai import OpenAI

class MyAgentLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        # 单独处理modelscope供应商的api
        if provider == "modelscope":
            print("正在使用自定义的ModelScope Provider")
            self.provider = provider

            # 解析modelscope的凭证
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")

            if not self.api_key:
                raise ValueError("API_KEY为空，请检查")
            self.model = model or os.getenv("MODEL_ID")
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)
            # 创建自定义的API调用client
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
        else:
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )