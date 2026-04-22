import os

from hello_agents import SimpleAgent, HelloAgentsLLM, CalculatorTool
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 创建大模型实例
llm = HelloAgentsLLM(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("MODEL_ID")
)

# 创建Agent实例
agent = SimpleAgent(
    name="Agent助手",
    llm=llm,
    system_prompt="你是一个为用户提供各种服务的Agent智能体"
)

# 基础对话
response = agent.run("你好，请介绍一下自己")
print(response)

# 添加计算器工具
calculator = CalculatorTool()

print()
print()

response = agent.run("帮我计算一下 3 - 5 * 2 + 7")
print(response)

# 查看历史记录
print(f"历史消息数：{len(agent.get_history())}, 对话历史：{agent.get_history()}")



