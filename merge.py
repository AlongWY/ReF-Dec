import os
from argparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM

TEMPLATE = r"""
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = messages | selectattr("role", "equalto", "user") | list %}

{%- for message in lmessages | rejectattr("role", "equalto", "tool") | rejectattr("role", "equalto", "tool_results") | selectattr("tool_calls", "undefined") %}
    {%- if (message["role"] == "user") != (loop.index0 % 2 == 0) %}
        {{- raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif %}
{%- endfor %}

{{- bos_token }}
{%- for message in messages %}
    {%- if message["role"] == "user" %}
        {{- "[INST] " }}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
        {%- endif %}
        {{- message["content"] + "[/INST]" }}
    {%- elif message["role"] == "tool_calls" or message.tool_calls is defined %}
        {%- if message.tool_calls is defined %}
            {%- set tool_calls = message.tool_calls %}
        {%- else %}
            {%- set tool_calls = message.content %}
        {%- endif %}
        {{- "[TOOL_CALLS] [" }}
        {%- for tool_call in tool_calls %}
            {%- set out = tool_call.function|tojson %}
            {{- out }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["role"] == "assistant" %}
        {{- " " + message["content"] }}
    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS] {"content": ' + content|string + "}[/TOOL_RESULTS]" }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}
"""


def main():
    parser = ArgumentParser()
    parser.add_argument("--base-model", type=str, default="LLM4Binary/llm4decompile-6.7b-v1.5")
    parser.add_argument("--peft-path", type=str, default="ylfeng/ReF-Decompile-lora")
    parser.add_argument("--output-dir", type=str, default="ReF-Decompile")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        tokenizer = AutoTokenizer.from_pretrained(args.peft_path)
        tokenizer.chat_template = TEMPLATE
        tokenizer.additional_special_tokens = None
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

        # resize the vocabulary
        if len(tokenizer) > base_model.get_input_embeddings().weight.size(0):
            base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        lora_model = PeftModelForCausalLM.from_pretrained(base_model, args.peft_path)

        merged_model = lora_model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
