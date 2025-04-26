import os
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional

# 设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.tracing.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

from pydantic import Field

# 自定义一个调用 Ollama 本地模型的 LLM 类
class OllamaLLM(LLM):
    model_name: str = Field(..., description="The name of the model to use.")
    api_url: str = Field(default="http://localhost:11434", description="The URL of the Ollama API.")

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        # 调用 Ollama 的 REST API
        payload = {"model": self.model_name, "prompt": prompt}
        response = requests.post(f"{self.api_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]

# 初始化 Ollama LLM
ollama_llm = OllamaLLM(model_name="Qwen2.5:latest")  # 替换为你的模型名称

# 定义 PromptTemplate
prompt = PromptTemplate(
    input_variables=["question"],
    template="请回答以下问题：{question}"
)

# 创建 LLMChain
chain = LLMChain(llm=ollama_llm, prompt=prompt)

# 调用链并获取结果
question = "什么是LangChain？"
response = chain.run(question)
print(response)