import asyncio
import json
import re
import os.path
import tempfile
import subprocess
import clang.cindex
from tqdm import tqdm
from collections import Counter
from rodata import parse_ro_data
import random

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


def format_struct_definition(
    struct_type_name, struct_decl, parsed_structs: list, parsed_structs_set: set
):
    struct_type_name = struct_type_name.removeprefix("struct ")
    if struct_type_name in parsed_structs_set:
        return
    parsed_structs_set.add(struct_type_name)
    struct_def = f"struct {struct_type_name} {{\n"

    for field in struct_decl.get_children():
        if field.kind == clang.cindex.CursorKind.FIELD_DECL:
            field_name = field.spelling
            field_type_name = field.type.spelling
            field_canonical_type = field.type.get_canonical()

            if (
                field_canonical_type.get_declaration().kind
                == clang.cindex.CursorKind.STRUCT_DECL
            ):
                nested_struct_decl = field_canonical_type.get_declaration()
                if nested_struct_decl.spelling not in parsed_structs_set:
                    format_struct_definition(
                        field_type_name,
                        nested_struct_decl,
                        parsed_structs,
                        parsed_structs_set,
                    )
                struct_def += f"  struct {field_type_name} {field_name};\n"
            else:
                struct_def += f"  {field_type_name} {field_name};\n"

    struct_def += "};"
    struct_def += f"\ntypedef struct {struct_type_name} {struct_type_name};"
    parsed_structs.append(struct_def)


async def parse_function_structs(filename, func_name):
    index = clang.cindex.Index.create()
    tu = index.parse(filename)
    parsed_structs = []
    parsed_structs_set = set()
    for cursor in tu.cursor.get_children():
        if (
            cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
            and cursor.spelling == func_name
        ):
            # 函数返回值类型
            if (
                cursor.result_type.get_declaration().kind
                == clang.cindex.CursorKind.STRUCT_DECL
            ):
                struct_type_name = cursor.result_type.spelling
                struct_decl = cursor.result_type.get_declaration()

                format_struct_definition(
                    struct_type_name,
                    struct_decl,
                    parsed_structs,
                    parsed_structs_set,
                )

            for arg in cursor.get_arguments():
                struct_type_name = arg.type.spelling
                canonical_type = arg.type.get_canonical()

                if (
                    canonical_type.get_declaration().kind
                    == clang.cindex.CursorKind.STRUCT_DECL
                ):
                    struct_decl = canonical_type.get_declaration()
                    format_struct_definition(
                        struct_type_name,
                        struct_decl,
                        parsed_structs,
                        parsed_structs_set,
                    )

    return parsed_structs


async def compile_with_optimization(code_file, binary_file, optimization_level):
    # Compile the C program to get the binary
    proc = await asyncio.create_subprocess_exec(
        "gcc",
        "-shared",
        "-fPIC",
        f"-{optimization_level}",
        "-lm",
        "-o",
        binary_file,
        code_file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        print("Timeout reached, terminating the compiler...")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            print("Process did not terminate, killing the compiler...")
            proc.kill()
            await proc.wait()
    except Exception as e:
        print(e)

    if not os.path.exists(binary_file):
        # print(stderr.decode())
        raise ValueError("Compilation failed")


async def decodedlines(binary_file: str):
    # Compile the C program to get the binary
    proc = await asyncio.create_subprocess_exec(
        "dwarfdump",
        "-ls",
        binary_file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        print("Timeout reached, terminating the compiler...")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            print("Process did not terminate, killing the compiler...")
            proc.kill()
            await proc.wait()
    except Exception as e:
        print(e)

    output = stdout.decode()

    # 正则表达式用于匹配行号
    lines = set()
    line_number_pattern = re.compile(r"\[\s*(\d+),")
    # 遍历每一行，匹配并提取行号
    for line in output.splitlines():
        match = line_number_pattern.search(line)
        if match:
            lines.add(int(match.group(1)))
    return lines


async def disassemble(binary_file, func_name):
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


async def format_asm(asm, func_name, convert=True, shuffle=True):
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

    if shuffle:
        jump_labels = random.sample([i for i in range(num_jumps * 10)], num_jumps)
        addr_labels = random.sample([i for i in range(num_address * 10)], num_address)
    else:
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


async def compile_code(code, optimization_level, func_name="func0", shuffle=True):
    with tempfile.TemporaryDirectory() as tmpdir_path:
        # Write the code to a file if it is not provided
        code_file = os.path.join(tmpdir_path, "code.c")
        binary_file = os.path.join(tmpdir_path, "code.out")
        optimization_level = optimization_level.strip("-")

        with open(code_file, "w") as f:
            f.write(code)

        # Parse the code to get the struct definitions
        parsed_structs = await parse_function_structs(code_file, func_name)

        # compile the code
        await compile_with_optimization(code_file, binary_file, optimization_level)

        # Parse the DWARF information to get the variable locations
        rodata_addr, rodata_data, rodata_parsed, num_missing = await parse_ro_data(
            code_file, binary_file
        )

        # Disassemble the binary to get the assembly code
        raw_asm = await disassemble(binary_file, func_name)

        # Format the assembly code
        asm_unlabeled, _ = await format_asm(
            raw_asm, func_name, convert=False, shuffle=shuffle
        )
        asm_labeled, address_mapping = await format_asm(
            raw_asm, func_name, convert=True, shuffle=shuffle
        )
        return (
            asm_unlabeled,
            asm_labeled,
            address_mapping,
            rodata_addr,
            rodata_data,
            rodata_parsed,
            num_missing,
            parsed_structs,
        )


async def main():
    # https://github.com/albertan017/LLM4Decompile/blob/main/decompile-eval/decompile-eval.json
    with open("data/decompile-eval-gcc-raw.json") as f:
        data = json.load(f)

        labeld_data = []
        counter = Counter()
        pbar = tqdm(data)
        for item in pbar:
            (
                asm,
                asm_labeled,
                address_mapping,
                rodata_addr,
                rodata_data,
                rodata_parsed,
                num_missing,
                structs,
            ) = await compile_code(item["c_func"], item["type"], "func0", shuffle=False)

            labeld_data.append(
                {
                    "task_id": item["task_id"],
                    "type": item["type"],
                    "c_func": item["c_func"],
                    "c_test": item["c_test"],
                    "asm": asm,
                    "asm_labeled": asm_labeled,
                    "num_missing": num_missing,
                    "address_mapping": address_mapping,
                    "rodata_addr": rodata_addr,
                    "rodata_data": rodata_data,
                    "rodata_parsed": rodata_parsed,
                    "structs": structs,
                }
            )

            if num_missing > 0:
                counter.update([num_missing])
                pbar.set_postfix({str(k): v for k, v in counter.items()}, refresh=True)

    with open("data/decompile-eval-gcc-rodata.json", "w") as f:
        json.dump(labeld_data, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
