import re
import json
import argparse
import struct
import asyncio
import subprocess
from elftools.elf.elffile import ELFFile
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

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


JUMP_REGEX = r"(?P<op>j\w+)(?P<blank>\s+)(?P<addr>[0-9A-Fa-f]+)\s+<(?P<func>[\w\.@]+)\+(?P<bias>0x[0-9A-Fa-f]+)>"
CALL_REGEX = (
    r"(?P<op>call\w*)(?P<blank>\s+)(?P<addr>[0-9A-Fa-f]+)\s+<(?P<func>[\w\.@]+)>"
)
ADDRESSL_REGEX = r"(?P<op>\w+)(?P<blank>\s+)(?P<bias>0x[0-9A-Fa-f]+)\(%rip\),%(?P<reg>\w+)\s+#\s+(?P<addr>[0-9A-Fa-f]+).*"
ADDRESSR_REGEX = r"(?P<op>\w+)(?P<blank>\s+)%(?P<reg>\w+),(?P<bias>0x[0-9A-Fa-f]+)\(%rip\)\s+#\s+(?P<addr>[0-9A-Fa-f]+).*"
ADDRESSVA_REGEX = r"(?P<op>\w+)(?P<blank>\s+)(?P<val>\$0x[0-9A-Fa-f]+),(?P<bias>0x[0-9A-Fa-f]+)\(%rip\)\s+#\s+(?P<addr>[0-9A-Fa-f]+).*"
ADDRESSAV_REGEX = r"(?P<op>\w+)(?P<blank>\s+)(?P<bias>0x[0-9A-Fa-f]+)\(%rip\),(?P<val>\$0x[0-9A-Fa-f]+)\s+#\s+(?P<addr>[0-9A-Fa-f]+).*"

JUMP_REGEX = re.compile(JUMP_REGEX)
CALL_REGEX = re.compile(CALL_REGEX)
ADDRESSL_REGEX = re.compile(ADDRESSL_REGEX)
ADDRESSR_REGEX = re.compile(ADDRESSR_REGEX)
ADDRESSVA_REGEX = re.compile(ADDRESSVA_REGEX)
ADDRESSAV_REGEX = re.compile(ADDRESSAV_REGEX)


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


async def disassemble(binary_file: str, func_name: str):
    # Disassemble the binary to get the assembly code
    proc = await asyncio.create_subprocess_exec(
        "objdump",
        "-d",
        f"--disassemble={func_name}",
        binary_file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        asm, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        print("Timeout reached, terminating the objdump...")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            print("Process did not terminate, killing the objdump...")
            proc.kill()
            await proc.wait()
        return None
    except Exception:
        return None
    asm = asm.decode()
    return asm


async def format_asm(asm, func_name, convert=True):
    # IMPORTANT replace func0 with the function name
    if "<" + func_name + ">:" not in asm:
        raise ValueError("compile fails")
    # IMPORTANT replace func0 with the function name
    asm = (
        "<" + func_name + ">:" + asm.split("<" + func_name + ">:")[-1].split("\n\n")[0]
    )

    asm_sp = asm.split("\n")

    # 1. found num of jump instructions
    # 2. founf num of address instructions

    jump_insts = JUMP_REGEX.findall(asm)
    addr_insts = (
        ADDRESSL_REGEX.findall(asm)
        + ADDRESSR_REGEX.findall(asm)
        + ADDRESSVA_REGEX.findall(asm)
        + ADDRESSAV_REGEX.findall(asm)
    )

    num_jumps = len(jump_insts)
    num_address = len(addr_insts)

    jump_labels = [i for i in range(num_jumps)]
    addr_labels = [i for i in range(num_address)]

    jumping_mapping = {}
    address_mapping = {}

    asm_clean = ["<" + func_name + ">:"]
    for tmp in asm_sp:
        parts = tmp.split("\t")
        if len(parts) < 3:
            continue

        if len(parts) == 3:
            addr, op_code, asm_code = parts

            addr = addr.strip().removesuffix(":")

            # match the jump instructions

            mj = JUMP_REGEX.match(asm_code)
            # match the call instructions
            mc = CALL_REGEX.match(asm_code)

            # match the address instructions
            # addsd  0xec0(%rip),%xmm0        # 200a <_fini+0xe7c>
            # addsd  %xmm0,0xec0(%rip)        # 200b <_fini+0xe7c>
            ml = ADDRESSL_REGEX.match(asm_code)
            mr = ADDRESSR_REGEX.match(asm_code)
            mva = ADDRESSVA_REGEX.match(asm_code)
            mav = ADDRESSAV_REGEX.match(asm_code)

            if mj and convert:
                addr_ref = int(mj.group("addr"), 16)
                if addr_ref not in jumping_mapping:
                    jumping_mapping[addr_ref] = {
                        "label": f"L{jump_labels[len(jumping_mapping)]}",
                        "addr": addr_ref,
                    }
                asm_code = (
                    mj.group("op")
                    + mj.group("blank")
                    + jumping_mapping[addr_ref]["label"]
                )
                asm_clean.append((addr.strip(), asm_code))
            elif mc and convert:
                asm_code = (
                    mc.group("op") + mc.group("blank") + "<" + mc.group("func") + ">"
                )
                asm_clean.append((addr.strip(), asm_code))
            elif ml and convert:
                # 转换成 addsd  D1(%rip),%xmm0
                bias = int(ml.group("bias"), 16)
                addr_ref = int(ml.group("addr"), 16)

                if addr_ref not in address_mapping:
                    address_mapping[addr_ref] = {
                        "label": f"D{addr_labels[len(address_mapping)]}",
                        "addr": addr_ref,
                        "bias": [bias],
                    }
                else:
                    address_mapping[addr_ref]["bias"].append(bias)

                data_label = address_mapping[addr_ref]["label"]
                asm_code = (
                    ml.group("op")
                    + ml.group("blank")
                    + data_label
                    + r"(%rip),%"
                    + ml.group("reg")
                )
                asm_clean.append((addr.strip(), asm_code))
            elif mr and convert:
                bias = int(mr.group("bias"), 16)
                addr_ref = int(mr.group("addr"), 16)

                if addr not in address_mapping:
                    address_mapping[addr_ref] = {
                        "label": f"D{addr_labels[len(address_mapping)]}",
                        "addr": addr_ref,
                        "bias": [bias],
                    }
                else:
                    address_mapping[addr_ref]["bias"].append(bias)

                data_label = address_mapping[addr_ref]["label"]
                asm_code = (
                    mr.group("op")
                    + mr.group("blank")
                    + "%"
                    + mr.group("reg")
                    + ","
                    + data_label
                    + r"(%rip)"
                )
                asm_clean.append((addr.strip(), asm_code))
            elif mva and convert:
                bias = int(mva.group("bias"), 16)
                addr_ref = int(mva.group("addr"), 16)

                if addr_ref not in address_mapping:
                    address_mapping[addr_ref] = {
                        "label": f"D{addr_labels[len(address_mapping)]}",
                        "addr": addr_ref,
                        "bias": [bias],
                    }
                else:
                    address_mapping[addr_ref]["bias"].append(bias)

                data_label = address_mapping[addr_ref]["label"]
                asm_code = (
                    mva.group("op")
                    + mva.group("blank")
                    + mva.group("val")
                    + ","
                    + data_label
                    + r"(%rip)"
                )
                asm_clean.append((addr.strip(), asm_code))
            elif mav and convert:
                bias = int(mav.group("bias"), 16)
                addr_ref = int(mav.group("addr"), 16)

                if addr_ref not in address_mapping:
                    address_mapping[addr_ref] = {
                        "label": f"D{addr_labels[len(address_mapping)]}",
                        "addr": addr_ref,
                        "bias": [bias],
                    }
                else:
                    address_mapping[addr_ref]["bias"].append(bias)

                data_label = address_mapping[addr_ref]["label"]
                asm_code = (
                    mav.group("op")
                    + mav.group("blank")
                    + data_label
                    + r"(%rip),"
                    + mav.group("val")
                )
                asm_clean.append((addr.strip(), asm_code.strip()))
            else:
                asm_clean.append((addr.strip(), asm_code.strip()))
        else:
            idx = min(len(parts) - 1, 2)
            tmp_asm = "  ".join(parts[idx:])  # remove the binary code
            tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
            asm_clean.append(tmp_asm)

    asm_combined = ""
    for item in asm_clean:
        if isinstance(item, tuple):
            addr, asm_code = item
            addr = int(addr, 16)
            if addr in jumping_mapping:
                asm_combined += (
                    jumping_mapping[addr]["label"] + ":\n  " + asm_code + "\n"
                )
            else:
                asm_combined += "  " + asm_code + "\n"
        else:
            asm_combined += item + "\n"

    address_ref = {}
    for addr, item in address_mapping.items():
        address_ref[item["label"]] = item

    for addr, item in jumping_mapping.items():
        address_ref[item["label"]] = item

    return asm_combined.strip(), address_ref


async def extract_function_rodata(elf: ELFFile):
    # 获取 .rodata 段
    rodata_section = elf.get_section_by_name(".rodata")
    if not rodata_section:
        # rodata_addr, rodata_data.hex(), rodata_values, num_missing
        return None, None, {}, 0
    rodata_data = rodata_section.data()
    rodata_addr = rodata_section["sh_addr"]
    return rodata_addr, rodata_data


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


async def model_decompile(
    # for decompile
    assembly,
    address_mapping,
    rodata_data,
    rodata_addr,
    enable_tool=False,
    # model config
    model="model",
    timeout=9999,
    max_tokens=1024,
    temperature=0.01,
    stream=False,
    client: AsyncOpenAI = None,
    **chat_params,
):
    assert client is not None, "client is required"
    messages = [
        {
            "role": "user",
            "content": f"What is the c source code of the assembly code below:\n\n{assembly}",
        },
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
            if found and found["addr"] < rodata_addr + len(rodata_data):
                try:
                    addr = found["addr"]
                    if parsed_data_size is not None:
                        parsed_data_size = int(parsed_data_size[1:-1])

                    value = await read_data(
                        parsed_data_type,
                        parsed_data_size,
                        rodata_data,
                        addr - rodata_addr,
                    )
                    result = await render_rodata(
                        data_label, parsed_data_type, parsed_data_size, value
                    )
                    tool_results.append(result)
                    print(data_label, data_type, found)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(data_label, data_type, found, e)

        messages.append(chat_completion_resp.choices[0].message.model_dump())
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

    messages.append(
        {
            "role": "assistant",
            "content": content,
        }
    )
    try:
        # regex find last ```c...```
        codes = re.findall(r"```\w*\n(?P<code>.*?)```", content, re.DOTALL)
        return codes[-1] if codes else content
    except Exception as e:
        return content

    return content


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
        default=True,
        help="Enable tool",
    )

    args = parser.parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    model_list = await client.models.list()
    model_name = model_list.data[0].id

    binary_file = "lib.so"
    function_name = "func0"

    raw_asm = await disassemble(binary_file, function_name)
    assembly, address_mapping = await format_asm(raw_asm, function_name)

    with open(binary_file, "rb") as f:
        elf = ELFFile(f)
        rodata_addr, rodata_data = await extract_function_rodata(elf)

    c_func_decompile = await model_decompile(
        assembly=assembly,
        address_mapping=address_mapping,
        rodata_data=rodata_data,
        rodata_addr=rodata_addr,
        enable_tool=args.enable_tool,
        client=client,
        model=model_name,
        temperature=0.0,
    )

    print(assembly)
    print(c_func_decompile)


if __name__ == "__main__":
    asyncio.run(main())
