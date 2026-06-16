"""
Liquid AI model handlers for BFCL evaluation.

This module provides handlers for Liquid models using OpenAI-compatible API endpoints.
"""

import ast
import json
import os
import re
import time
from typing import Any, Dict, List

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from bfcl_eval.utils import extract_test_category_from_id, is_agentic
from openai import OpenAI
from overrides import override


LIQUID_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of <|tool_call_start|>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]<|tool_call_end|>
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""

LIQUID_SYSTEM_PROMPT_WITHOUT_TOOLS = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of <|tool_call_start|>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]<|tool_call_end|>
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

# Non-restrictive tool-call FORMAT guidance for agentic categories (web_search, memory).
# Those categories carry BFCL's own system prompt (the {answer,context} format, and for
# memory the agent role + the available memory tools described inline -- they have NO
# `function` field, so tools= is empty and the model learns its tools from the system
# prompt). Prepending LIQUID_SYSTEM_PROMPT_WITHOUT_TOOLS ("you are given a set of possible
# functions ... only return the function calls") buries that and makes the model report
# "no applicable tools are available". So for agentic we keep BFCL's prompt and only append
# the format hint, mirroring LFM2Handler (local_inference/lfm2.py).
LIQUID_TOOL_CALL_FORMAT_INSTRUCTION = (
    "When you decide to invoke any of the available function(s), you MUST put the "
    "call(s) in the format of <|tool_call_start|>[func_name1(params_name1=params_value1, "
    "params_name2=params_value2...), func_name2(params)]<|tool_call_end|>."
)


def fmt(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)

def convert_json_call_to_py(tool: dict) -> str:
    try:
        line = f"{tool['name']}("
        for i, (k, v) in enumerate(tool["arguments"].items()):
            if i > 0:
                line += ", "
            line += f"{k}={fmt(v)}"
        line += ")"
        return line
    except Exception as e:
        return ""


def convert_jsonl_calls_to_py(tool_dicts: list) -> str:
    tool_calls_python = []
    for tool in tool_dicts:
        tool_calls_python.append(convert_json_call_to_py(tool))
    tool_calls_python = [call for call in tool_calls_python if call != ""]
    return f"[{', '.join(tool_calls_python)}]"



def extract_think_block(text: str) -> str | None:
    """
    Returns the content inside <think>...</think>, or None if not present.
    """
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    return m.group(1).strip() if m else ""

def parse_liquid_response(response: str | None) -> str:
    """
    Parse the response from LiquidAI and return the function call content.
    Extracts content from <|tool_call_start|>...<|tool_call_end|> tags if present.
    """
    if response is None:
        return "No Response"
    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception:
            return ""
    
    if "<|tool_call_start|>" in response and "<|tool_call_end|>" in response:
        match = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", response, re.DOTALL)
        if match:
            answer = match.group(1)
        else:
            answer = response
        answer = re.sub(r"<\|tool_call_start\|>|<\|tool_call_end\|>", "", answer).strip()
        return answer
    else:
        return response


def _eval_node(node: ast.AST):
    """Convert a limited subset of AST nodes into Python objects."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [_eval_node(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        return {
            _eval_node(k): _eval_node(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Name):
        # For things like order=ascending / descending
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # Handle negative numbers
        return -_eval_node(node.operand)
    else:
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def _get_function_name(node: ast.expr) -> str:
    """Extract function name from AST node, handling dotted names."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return _get_function_name(node.value) + "." + node.attr
    else:
        raise ValueError(f"Invalid function name node: {node}")


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract all tool calls from Liquid format response.
    
    Format: <|tool_call_start|>[func_name(arg1="val", arg2=123), func2(...)]<|tool_call_end|>
    
    Returns a list of {"name": str, "arguments": dict}.
    """
    if not text:
        return []

    calls: List[Dict[str, Any]] = []

    # Find ALL tool call blocks
    tool_call_pattern = r'<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>'
    matches = re.findall(tool_call_pattern, text, flags=re.DOTALL)
    
    for match in matches:
        try:
            # Parse the function calls
            parsed = ast.parse(f"x = [{match}]").body[0].value.elts
            for call in parsed:
                try:
                    if not isinstance(call, ast.Call):
                        continue
                    function_name = _get_function_name(call.func)
                    args = {kw.arg: _eval_node(kw.value) for kw in call.keywords}
                    calls.append({'name': function_name, 'arguments': args})
                except Exception as e:
                    # Log but continue processing other calls
                    print(f"Warning: Failed to parse individual call: {e}")
                    continue
        except Exception as e:
            # Log but continue processing other matches
            print(f"Warning: Failed to parse tool call block: {e}")
            continue
    
    return calls


def _is_tool_call_response_format(items: list) -> bool:
    """Check if the response is in the expected tool call format."""
    if not isinstance(items, list) or not items:
        return False
    for it in items:
        if not isinstance(it, dict):
            return False
        if set(it.keys()) != {"name", "arguments"}:
            return False
    return True


class LiquidHandler(OpenAICompletionsHandler):
    """
    Handler for Liquid models in prompting mode.
    Uses OpenAI-compatible API with custom response parsing for Liquid format.
    
    Supports generation parameters via CLI args:
        --max-tokens: Maximum tokens to generate (default: 4096)
        --repetition-penalty: Repetition penalty (default: 1.0)
        --min-p: Min-p sampling parameter (default: 0.0)
    """
    
    def __init__(self, model_name, temperature, registry_name=None, is_fc_model=False, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        port = os.getenv("PORT", 8000)
        
        self.client = OpenAI(
            base_url=os.getenv("LIQUID_API_BASE_URL", f"http://localhost:{port}/v1"),
            api_key=os.getenv("LIQUID_API_KEY", "none"),
        )
        
        # These will be set by build_handler() after construction
        # Default values here, overridden by CLI args
        self.max_tokens = 4096
        self.min_p = 0.0
        self.repetition_penalty = 1.0
        self.preserve_thinking = None
    
    def _build_extra_body(self) -> dict:
        """Build extra_body dict for vLLM-specific parameters."""
        extra_body = {}
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.min_p and self.min_p > 0:
            extra_body["min_p"] = self.min_p
        if self.preserve_thinking:
            extra_body["chat_template_kwargs"] = {"preserve_thinking": True}
        return extra_body

    @override
    def _query_prompting(self, inference_data: dict) -> Any:
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]
        
        # Convert function format without mutating original
        formatted_functions = []
        for func in function:
            if isinstance(func, dict) and "name" in func:
                func_copy = {k: v for k, v in func.items() if k != "response"}
                formatted_functions.append({"type": "function", "function": func_copy})
            elif isinstance(func, dict) and "function" in func:
                inner = {k: v for k, v in func["function"].items() if k != "response"}
                formatted_functions.append({"type": "function", "function": inner})
            else:
                formatted_functions.append(func)
        
        inference_data["inference_input_log"] = {"message": repr(message), "tools": formatted_functions}
        
        kwargs = {
            "messages": message,
            "tools": formatted_functions,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        return self.generate_with_backoff(**kwargs)

    def _parse_query_response_prompting_v5(self, api_response: Any) -> dict:
        message = api_response.choices[0].message
        raw_content = message.content or ""
        think_block = extract_think_block(raw_content)
        if "<think>" in raw_content and "</think>" in raw_content:
            cleaned_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).lstrip("\n")
        else:
            cleaned_content = raw_content
        tool_calls = getattr(message, "tool_calls", None) or []
        decoded_calls = None
        if tool_calls:
            model_responses = []
            for func_call in tool_calls:
                json_call = {}
                json_call["name"] = func_call.function.name
                arg = func_call.function.arguments if isinstance(func_call.function.arguments, str) else {}
                try:
                    arg = json.loads(arg)
                except:
                    pass
                json_call["arguments"] = arg
                model_responses.append(json_call)
            model_responses = "<|tool_call_start|>" + convert_jsonl_calls_to_py(model_responses) + "<|tool_call_end|>"
            decoded_calls = extract_tool_calls(model_responses)
            tool_call_ids = [func_call.id for func_call in tool_calls]
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": model_responses,
            }
        else:
            model_responses = cleaned_content
            tool_call_ids = []
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_content,
            }

        if think_block:
            model_responses_message_for_chat_history["thinking"] = think_block

        final_results = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

        if decoded_calls is not None:
            final_results["model_responses_decoded"] = decoded_calls
            final_results["tool_calls_decoded"] = decoded_calls

        return final_results




    def _parse_query_response_prompting_v4(self, api_response: Any) -> dict:
        content_tobe_parsed = api_response.choices[0].message.content
        think_block = extract_think_block(content_tobe_parsed)
        if "<think>" in content_tobe_parsed and "</think>" in content_tobe_parsed:
            content_tobe_parsed = re.sub(r"<think>.*?</think>", "", content_tobe_parsed, flags=re.DOTALL)
            content_tobe_parsed = content_tobe_parsed.lstrip('\n')
        model_responses = parse_liquid_response(content_tobe_parsed)
        prompt_tokens = api_response.usage.prompt_tokens
        completion_tokens = api_response.usage.completion_tokens
        
        # Also extract decoded tool calls for multi-turn
        decoded_calls = extract_tool_calls(content_tobe_parsed)
        
        model_responses_message_for_chat_history = {
            "role": "assistant",
            "content": content_tobe_parsed,
        }
        if think_block:
            model_responses_message_for_chat_history["thinking"] = think_block

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "model_responses_decoded": decoded_calls,
            "tool_calls_decoded": decoded_calls,
            "input_token": prompt_tokens,
            "output_token": completion_tokens
        }

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        
        if api_response.choices[0].finish_reason == "tool_calls":
            return self._parse_query_response_prompting_v5(api_response)
        else:
            return self._parse_query_response_prompting_v4(api_response)
        """
        return self._parse_query_response_prompting_v4(api_response)
        """



    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Override the default system prompt processing to use LIQUID_SYSTEM_PROMPT_WITHOUT_TOOLS.
        
        Unlike the parent class which calls system_prompt_pre_processing_chat_model to embed
        functions in the system prompt, this handler passes tools via the tools= parameter
        in _query_prompting, so we use a system prompt without embedded function definitions.
        """
        functions: list = test_entry["function"]
        prompts: list[dict] = test_entry["question"][0]
        
        
        functions = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

        # Agentic categories (web_search, memory) carry BFCL's own system prompt with the
        # tools described inline; prepending the restrictive LIQUID_SYSTEM_PROMPT_WITHOUT_TOOLS
        # buries that and makes the model report "no applicable tools are available". Mirror
        # LFM2Handler: non-agentic -> prepend the restrictive instruction; agentic -> keep
        # BFCL's prompt and only APPEND the non-restrictive tool-call format hint.
        is_agentic_category = is_agentic(extract_test_category_from_id(test_entry["id"]))
        system_injection = (
            LIQUID_TOOL_CALL_FORMAT_INSTRUCTION
            if is_agentic_category
            else LIQUID_SYSTEM_PROMPT_WITHOUT_TOOLS
        )
        if prompts and prompts[0].get("role") == "system":
            if is_agentic_category:
                prompts[0]["content"] = prompts[0]["content"] + "\n\n" + system_injection
            else:
                prompts[0]["content"] = system_injection + "\n\n" + prompts[0]["content"]
        else:
            prompts.insert(0, {"role": "system", "content": system_injection})

        # Return functions separately - they will be passed via tools= parameter in _query_prompting
        return {"message": [], "function": functions}

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        response_message = []
        decoded = model_response_data["tool_calls_decoded"]
        
        for execution_result, tool_call in zip(execution_results, decoded):
            response_message.append({'name': tool_call["name"], 'result': execution_result})
        
        inference_data["message"].append({
            "role": "tool",
            "content": repr(response_message),
        })
        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """Decode model output for AST evaluation."""
        extracted = extract_tool_calls(result)
        if extracted:
            decoded_output = []
            for call in extracted:
                decoded_output.append({call["name"]: call["arguments"]})
            return decoded_output
        try:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)
        except Exception:
            return []

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """Decode model output for execution."""
        extracted = extract_tool_calls(result)
        if extracted:
            calls = []
            for call in extracted:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in call["arguments"].items())
                calls.append(f"{call['name']}({args_str})")
            return calls
        try:
            return default_decode_execute_prompting(result)
        except Exception:
            return []


class LiquidFCAPIHandler(OpenAICompletionsHandler):
    """
    API handler for Liquid function-calling models using OpenAI-compatible endpoints.

    This handler formats prompts according to Liquid's chat template:
    - System messages with tool injection via function descriptions
    - Tool calls expected in format: <|tool_call_start|>[func_name(args)]<|tool_call_end|>
    
    Supports generation parameters via CLI args:
        --max-tokens: Maximum tokens to generate (default: 4096)
        --repetition-penalty: Repetition penalty (default: 1.0)
        --min-p: Min-p sampling parameter (default: 0.0)
    """

    def __init__(self, model_name, temperature, registry_name=None, is_fc_model=True, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        port = os.getenv("PORT", 8000)

        self.client = OpenAI(
            base_url=os.getenv("LIQUID_API_BASE_URL", f"http://localhost:{port}/v1"),
            api_key=os.getenv("LIQUID_API_KEY", "none"),
        )

        # Enable FC path
        self.is_fc_model = True
        
        # These will be set by build_handler() after construction
        # Default values here, overridden by CLI args
        self.max_tokens = 4096
        self.min_p = 0.0
        self.repetition_penalty = 1.0
        self.preserve_thinking = False
    
    def _build_extra_body(self) -> dict:
        """Build extra_body dict for vLLM-specific parameters."""
        extra_body = {}
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.min_p and self.min_p > 0:
            extra_body["min_p"] = self.min_p
        if self.preserve_thinking:
            extra_body["chat_template_kwargs"] = {"preserve_thinking": True}
        return extra_body

    def _format_liquid_prompt(self, messages: List[Dict], tools: List[Dict] = None) -> List[Dict]:
        """
        Format messages according to Liquid's chat template requirements.
        Returns properly formatted messages for the API call.
        """
        if not messages:
            return []

        formatted_messages = []
        msgs = list(messages)

        # Extract system message if present
        system_prompt = ""
        if msgs and msgs[0].get("role") == "system":
            system_prompt = msgs[0].get("content", "")
            msgs = msgs[1:]

        # Build system message with tools if provided
        if system_prompt:
            system_content = system_prompt
        else:
            system_content = LIQUID_SYSTEM_PROMPT.format(functions=tools)

        formatted_messages.append({"role": "system", "content": system_content})

        # Add remaining messages
        for msg in msgs:
            role = msg.get("role", "")
            if role == "tool":
                tool_msg: Dict[str, Any] = {"role": "tool"}
                content = msg.get("content", "")
                if isinstance(content, str):
                    wrapped = content
                else:
                    wrapped = json.dumps(content)
                tool_msg["content"] = wrapped

                if "tool_call_id" in msg:
                    tool_msg["tool_call_id"] = msg["tool_call_id"]
                if "name" in msg:
                    tool_msg["name"] = msg["name"]

                formatted_messages.append(tool_msg)
            else:
                formatted_messages.append({k: v for k, v in msg.items()})

        return formatted_messages

    # ----------------------------
    # Function Calling (FC) methods
    # ----------------------------
    @override
    def _query_FC(self, inference_data: dict):
        """Query the model in function calling mode."""
        messages = inference_data["message"]
        tools = inference_data.get("tools", [])

        formatted_messages = self._format_liquid_prompt(messages, tools)

        inference_data["inference_input_log"] = {
            "message": repr(formatted_messages),
            "tools": tools
        }
        
        kwargs = {
            "messages": formatted_messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        return self.generate_with_backoff(**kwargs)

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """Parse the response from function calling mode."""
        try:
            content = api_response.choices[0].message.content or ""
        except Exception:
            content = ""

        think_block = extract_think_block(content)
        if "<think>" in content and "</think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).lstrip("\n")

        # Extract tool calls from the response
        extracted_tool_calls = extract_tool_calls(content)

        if extracted_tool_calls and _is_tool_call_response_format(extracted_tool_calls):
            # Build OpenAI-compatible tool_calls for the chat history
            tool_calls_for_history = []
            tool_call_ids = []
            for i, call in enumerate(extracted_tool_calls):
                tool_id = f"call_{i}"
                tool_call_ids.append(tool_id)
                args_json_str = json.dumps(call["arguments"], ensure_ascii=False)
                tool_calls_for_history.append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": args_json_str
                    }
                })

            # Return raw model text for evaluation
            model_responses = content

            # Keep original content so model sees its previous tool call output in multi-turn
            model_response_message_for_chat_history = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls_for_history,
            }
        else:
            model_responses = content
            model_response_message_for_chat_history = {
                "role": "assistant",
                "content": content,
            }
            tool_call_ids = []

        if think_block:
            model_response_message_for_chat_history["thinking"] = think_block

        # Token usage
        usage = getattr(api_response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
        else:
            input_tokens = output_tokens = 0

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_response_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": input_tokens,
            "output_token": output_tokens,
            "model_responses_decoded": extracted_tool_calls,
        }

    # ----------------------------
    # Prompting methods
    # ----------------------------
    @override
    def _query_prompting(self, inference_data: dict):
        """Query the model in prompting mode."""
        messages = inference_data["message"]
        function_list = inference_data.get("function", [])

        formatted_messages = self._format_liquid_prompt(messages, function_list)

        inference_data["inference_input_log"] = {"message": repr(formatted_messages)}

        kwargs = {
            "messages": formatted_messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        return self.generate_with_backoff(**kwargs)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """Parse the response from prompting mode."""
        try:
            content = api_response.choices[0].message.content or ""
        except Exception:
            content = ""

        think_block = extract_think_block(content)
        if "<think>" in content and "</think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).lstrip("\n")

        extracted_tool_calls = extract_tool_calls(content)

        if extracted_tool_calls and _is_tool_call_response_format(extracted_tool_calls):
            hist_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": extracted_tool_calls
            }
        else:
            hist_msg = {"role": "assistant", "content": content}

        if think_block:
            hist_msg["thinking"] = think_block

        # Token usage
        usage = getattr(api_response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
        else:
            input_tokens = output_tokens = 0

        return {
            "model_responses": content,
            "model_responses_message_for_chat_history": hist_msg,
            "model_responses_decoded": extracted_tool_calls,
            "input_token": input_tokens,
            "output_token": output_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        """Add assistant message to the conversation history in prompting mode."""
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: List[str], model_response_data: dict
    ) -> dict:
        """Add execution results to the conversation history in prompting mode."""
        payload = []
        decoded = model_response_data.get("model_responses_decoded", [])

        for i, exec_res in enumerate(execution_results):
            if i < len(decoded):
                d = decoded[i]
                name = d.get("name") if isinstance(d, dict) else str(d)
            else:
                name = f"function_{i}"
            payload.append({"name": name, "execution_result": exec_res})

        tool_content = json.dumps(payload, indent=2)
        inference_data["message"].append({"role": "tool", "content": tool_content})
        return inference_data

    @override
    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: List[str], model_response_data: dict
    ) -> dict:
        """Append OpenAI-compatible tool messages for the FC loop."""
        assistant_msg = model_response_data.get("model_responses_message_for_chat_history", {})
        tool_calls = assistant_msg.get("tool_calls", [])

        for i, exec_res in enumerate(execution_results):
            if i < len(tool_calls):
                tool_call = tool_calls[i]
                tool_id = tool_call.get("id", f"call_{i}")
                tool_name = tool_call.get("function", {}).get("name", f"function_{i}")
            else:
                tool_id = f"call_{i}"
                tool_name = f"function_{i}"

            raw_content = exec_res if isinstance(exec_res, str) else json.dumps(exec_res)

            inference_data["message"].append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": raw_content,
            })

        return inference_data

    # ----------------------------
    # Decoder methods
    # ----------------------------
    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """Decode model output for AST evaluation."""
        extracted = extract_tool_calls(result)
        if extracted:
            decoded_output = []
            for call in extracted:
                decoded_output.append({call["name"]: call["arguments"]})
            return decoded_output
        # No tool call tags - check if natural language response
        if "<|tool_call_start|>" not in result and "<|tool_call_end|>" not in result:
            return []
        try:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)
        except Exception:
            return []

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """Decode model output for execution."""
        extracted = extract_tool_calls(result)
        if extracted:
            calls = []
            for call in extracted:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in call["arguments"].items())
                calls.append(f"{call['name']}({args_str})")
            return calls
        # No tool call tags - check if natural language response
        if "<|tool_call_start|>" not in result and "<|tool_call_end|>" not in result:
            return []
        try:
            return default_decode_execute_prompting(result)
        except Exception:
            return []

