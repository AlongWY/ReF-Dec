import asyncio
import argparse
import os
import json
import re
import struct
import tempfile
from tqdm import tqdm
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import os.path
import subprocess

C_INCLUDE = """
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long ull;
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_data",
            "description": "Parse data from a label in preprocessed assembly using a guessed data type. Original addresses are replaced with labels like D1, D2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_label": {
                        "type": "string",
                        "description": "A predefined tag within the assembly instructions that points to a data storage area (e.g., D1, D2).",
                    },
                    "data_type": {
                        "type": "string",
                        "description": "The guessed data type for the specified label, which can be described as a base type (e.g., i8) or an array type (e.g., i8[8]). Base types are strictly limited to the following: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
                    },
                },
                "required": ["data_label", "data_type"],
                "additionalProperties": False,
            },
        },
    }
]


async def evaluate_func(c_func, c_test, c_func_decompile):
    with tempfile.TemporaryDirectory() as tempdir:
        c_include = C_INCLUDE
        for line in c_func.split("\n"):
            if "#include" in line:
                c_include += line + "\n"
                c_func = c_func.replace(line, "")
        for line in c_test.split("\n"):
            if "#include" in line:
                c_include += line + "\n"
                c_test = c_test.replace(line, "")
        c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test

        # Define the C file and executable names
        c_file = os.path.join(tempdir, "combine.c")
        executable = os.path.join(tempdir, "combine")
        if os.path.exists(executable):
            os.remove(executable)

        with open(c_file, "w") as f:
            f.write(c_combine)

        # Compile the C program to an executable
        try:
            proc = await asyncio.create_subprocess_exec(
                "gcc",
                c_file,
                "-o",
                executable,
                "-lm",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            print("Timeout reached, terminating the compiler...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("Process did not terminate, killing the compiler...")
                proc.kill()
                await proc.wait()
        except Exception:
            pass

        if not os.path.exists(executable):
            return 0, 0, None

        # Run the compiled executable
        proc = await asyncio.create_subprocess_exec(
            executable,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            flag_run = int(0 == await asyncio.wait_for(proc.wait(), timeout=10))
        except asyncio.TimeoutError:
            print("Timeout reached, terminating the process...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("Process did not terminate, killing it...")
                proc.kill()
                await proc.wait()
            flag_run = 0
        except Exception:
            flag_run = 0

        return 1, flag_run, None


async def render_rodata(data_label, data_type, data_size, data_value):
    if data_type == "string":
        value = data_value
        result = f"{{{data_label}: {value}}}"
    if data_size:
        if data_type == "byte":
            value = [f"0x{v:02x}" for v in data_value]
            value = ",".join(value)
            result = f"{{{data_label}: [{value}]}}"
        elif data_type == "word":
            value = [f"0x{v:04x}" for v in data_value]
            value = ",".join(value)
            result = f"{{{data_label}: [{value}]}}"
        elif data_type == "dword":
            value = [f"0x{v:08x}" for v in data_value]
            value = ",".join(value)
            result = f"{{{data_label}: [{value}]}}"
        elif data_type == "qword":
            value = [f"0x{v:016x}" for v in data_value]
            value = ",".join(value)
            result = f"{{{data_label}: [{value}]}}"
        else:
            value = json.dumps(data_value)
            result = f"{{{data_label}: {value}}}"
    else:
        if data_type == "byte":
            value = f"0x{data_value:02x}"
            result = f"{{{data_label}: {value}}}"
        elif data_type == "word":
            value = f"0x{data_value:04x}"
            result = f"{{{data_label}: {value}}}"
        elif data_type == "dword":
            value = f"0x{data_value:08x}"
            result = f"{{{data_label}: {value}}}"
        elif data_type == "qword":
            value = f"0x{data_value:016x}"
            result = f"{{{data_label}: {value}}}"
        else:
            value = data_value
            result = f"{{{data_label}: {value}}}"

    return result


STRUCT_MAPPING = {
    "i8": ("b", 1),
    "u8": ("B", 1),
    "i16": ("h", 2),
    "u16": ("H", 2),
    "i32": ("i", 4),
    "u32": ("I", 4),
    "i64": ("q", 8),
    "u64": ("Q", 8),
    "f32": ("f", 4),
    "f64": ("d", 8),
    "byte": ("B", 1),
    "word": ("H", 2),
    "dword": ("I", 4),
    "qword": ("Q", 8),
}


async def read_data(data_type, data_size, rodata, bias):
    if data_type == "string":
        end_offset = rodata[bias:].find(b"\x00")
        if end_offset != -1:
            return json.dumps(rodata[bias : bias + end_offset].decode("utf-8"))
    else:
        try:
            type_fmt, type_size = STRUCT_MAPPING[data_type]
            parse_size = data_size if data_size is not None else 1
            byte_len = type_size * parse_size
            data = rodata[bias : bias + byte_len]
            value = struct.unpack(f"<{parse_size}{type_fmt}", data)
            if len(value) == 1 and data_size is None:
                return value[0]
            else:
                return list(value)
        except Exception as e:
            print(data_type, data_size, type_fmt, type_size, parse_size)
            print(e)


async def model_decompile(
    model="model",
    rodata=None,
    rodata_addr=None,
    rodata_parsed=None,
    address_mapping=None,
    timeout=9999,
    max_tokens=1024,
    temperature=0.01,
    stream=False,
    prompt=None,
    messages=None,
    client: AsyncOpenAI = None,
    index=None,
    opt_level=None,
    enable_tool=False,
    **chat_params,
):
    assert client is not None, "client is required"

    if messages is None:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    chat_completion_resp: ChatCompletion = await client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        tools=TOOLS if enable_tool else None,
        **chat_params,
    )

    if chat_completion_resp.choices[0].message.tool_calls:
        tool_calls = chat_completion_resp.choices[0].message.tool_calls
        tool_results = []
        for tool in tool_calls:
            function = tool.function
            arguments = json.loads(function.arguments)
            data_label = arguments["data_label"]
            data_type = arguments["data_type"]

            m = re.match(r"(?P<type>\w+)(?P<size>\[\d+\])?", data_type)
            if not m:
                continue

            parsed_data_type = m.group("type")
            parsed_data_size = m.group("size")

            found = address_mapping[data_label]
            if found and found["addr"] < rodata_addr + len(rodata):
                try:
                    addr = found["addr"]
                    if parsed_data_size is not None:
                        parsed_data_size = int(parsed_data_size[1:-1])

                    value = await read_data(
                        parsed_data_type,
                        parsed_data_size,
                        rodata,
                        addr - rodata_addr,
                    )
                    result = await render_rodata(
                        data_label, parsed_data_type, parsed_data_size, value
                    )
                    tool_results.append(result)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(data_label, data_type, found, e)

        messages.append({"role": "assistant", "tool_calls": tool_calls})
        for item in tool_results:
            messages.append({"role": "tool", "content": item})

        chat_completion_resp: ChatCompletion = await client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            tools=TOOLS if enable_tool else None,
            **chat_params,
        )
        content = chat_completion_resp.choices[0].message.content
    else:
        content = chat_completion_resp.choices[0].message.content

    try:
        # regex find last ```c...```
        codes = re.findall(r"```\w*\n(?P<code>.*?)```", content, re.DOTALL)
        return codes[-1] if codes else content
    except Exception as e:
        return content

    return content


async def run_decompile(
    sem,
    item,
    enable_tool,
    client: AsyncOpenAI = None,
    model="model",
    with_label=False,
):
    async with sem:
        c_func = item["c_func"]
        c_test = item["c_test"]
        rodata = item["rodata_data"]
        rodata_addr = item["rodata_addr"]
        rodata_parsed = item["rodata_parsed"]
        address_mapping = item["address_mapping"]
        input_asm_prompt = item["asm_labeled"] if with_label else item["asm"]

        messages = [
            {
                "role": "user",
                "content": f"What is the c source code of the assembly code below:\n\n{input_asm_prompt}",
            },
        ]

        try:
            if item.get("c_func_decompile", None) is not None:
                c_func_decompile, flag_compile, flag_run = (
                    item["c_func_decompile"],
                    item["compile"],
                    item["run"],
                )
            else:
                c_func_decompile = await model_decompile(
                    client=client,
                    model=model,
                    rodata=bytes.fromhex(rodata + "00" * 32)
                    if rodata is not None
                    else None,
                    rodata_addr=rodata_addr,
                    rodata_parsed=rodata_parsed.get("func0"),
                    address_mapping=address_mapping,
                    messages=messages,
                    temperature=0.0,
                    index=item["task_id"],
                    opt_level=item["type"],
                    enable_tool=enable_tool,
                )
                flag_compile, flag_run, err_info = await evaluate_func(
                    c_func, c_test, c_func_decompile
                )

        except Exception as e2:
            import traceback

            traceback.print_exc()
            return {
                **item,
                "compile": 0,
                "run": 0,
                "c_func_decompile": None,
                "c_func_re_decompile": None,
            }

        return {
            **item,
            "run": flag_run,
            "compile": flag_compile,
            "c_func_decompile": c_func_decompile,
        }


async def eval_model(
    client,
    data_all,
    num_semaphore=16,
    model_name="model",
    model_tag="model",
    output_file="result.jsonl",
    result_file=None,
    enable_tool=False,
    with_label=False,
):
    semaphore = asyncio.Semaphore(num_semaphore)
    num_compile = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
    num_run = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}

    tasks = [
        asyncio.create_task(
            run_decompile(
                semaphore,
                item,
                client=client,
                model=model_name,
                enable_tool=enable_tool,
                with_label=with_label,
            )
        )
        for item in data_all
    ]

    # Use tqdm to create a progress bar for the asyncio.gather
    results = []
    pbar = tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=model_tag,
        dynamic_ncols=True,
    )
    for f in pbar:
        try:
            result = await f
            results.append(result)

            opt_state = result["type"]
            num_compile[opt_state] += result["compile"]
            num_run[opt_state] += result["run"]
        except Exception as e:
            raise e

        pbar.set_postfix(num_run)

    with open(output_file, "a") as f:
        total_run = sum(num_run.values())
        total_run_rate = total_run / len(data_all)

        total_compile = sum(num_compile.values())
        total_compile_rate = total_compile / len(data_all)

        level_num = len(data_all) // 4

        data = {
            "model": model_name,
            # rates
            "total_run_rate": total_run_rate,
            "total_compile_rate": total_compile_rate,
            "run_rate": {k: v / level_num for k, v in num_run.items()},
            "compile_rate": {k: v / level_num for k, v in num_compile.items()},
            # numbers
            "total_run": total_run,
            "total_compile": total_compile,
            "num_run": num_run,
            "num_compile": num_compile,
        }

        f.write(json.dumps(data) + "\n")

    if result_file:
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        type=str,
        required=False,
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        default="None",
    )
    parser.add_argument(
        "--enable-tool",
        action="store_true",
        default=False,
        help="Enable tool",
    )
    parser.add_argument(
        "--with-label",
        action="store_true",
        default=False,
        help="Use labeled data",
    )

    args = parser.parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    model_list = await client.models.list()

    model_name = model_list.data[0].id
    model_tag = model_name.replace("/", "_")

    with open("data/decompile-eval-gcc-rodata.json", "r") as f:
        data_all = json.load(f)
    result_file = os.path.join("results", f"{model_tag}.json")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if not os.path.exists(result_file):
        await eval_model(
            client=client,
            data_all=data_all,
            model_name=model_name,
            model_tag=model_tag,
            result_file=result_file,
            enable_tool=args.enable_tool,
            with_label=args.with_label,
            output_file=os.path.join("results", "results.jsonl"),
        )


if __name__ == "__main__":
    asyncio.run(main())
