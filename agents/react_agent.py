"""ReAct Agent实现 - 推理与行动结合的智能体"""
from typing import Optional, List, Tuple
import re
from ..core.llm import AgentsLLM
from ..core.agent import Agent
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具
来获取信息，最终给出准确的答案。
## 可用工具
{tools}
## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:
Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:- `{{tool_name}}[{{tool_input}}]` - 调用指定工具- `Finish[最终答案]` - 当你有足够信息给出最终答案时
## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数
## 当前任务
**Question:** {question}
## 执行历史
{history}
现在开始你的推理和行动:
"""


class ReactAgent(Agent):
    """
        重写的ReAct Agent - 推理与行动结合的智能体
    """
    def __init__(
            self,
            name: str,
            llm: AgentsLLM,
            tool_registry: Optional[ToolRegistry] = None,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_steps: int = 5,
            custom_prompt: str = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的工具注册表）
            system_prompt: 系统提示词
            config: 配置对象
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题：{input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n=========== 第 {current_step} 步 ==================")

            # 1. 构建提示词
            tool_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tool_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages=messages, **kwargs).content

            if not response_text:
                print("❌ 错误：LLM未能返回有效响应。")
                break

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程终止。")
                break

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                # 保存到历史记录
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            # 5. 执行工具调用
            if action:
                tool_name, tool_input = self._parse_action(action)
                if not tool_name or tool_input is None:
                    self.current_history.append("Observation: 无效的Action格式，请检查。")
                    continue
                # 调用工具
                print(f"🎬 行动: {tool_name}[{tool_input}]")
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                print(f"👀 观察: {observation}")

                # 更新历史
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")

        # 达到最大步数
        print("⏰ 已达到最大步数，流程终止。")
        final_answer = f"当前已达到最大执行步数：{self.max_steps}， 任务未完成"

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析LLM输出，提取思考和行动"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None

        return thought, action

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析行动文本，提取工具名称和输入"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """解析行动输入"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""