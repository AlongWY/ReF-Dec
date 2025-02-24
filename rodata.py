import itertools
import json
import asyncio
import struct
import re
from collections import defaultdict
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_MODE_32, CS_OP_MEM, CS_OP_REG, x86
from elftools.elf.elffile import ELFFile
from clang.cindex import Index, CursorKind, TokenKind

TYPE_MAPPING = {
    "char": "i8",
    "unsigned char": "u8",
    "int": "i32",
    "unsigned int": "u32",
    "long": "i64",
    "unsigned long": "u64",
    "float": "f32",
    "double": "f64",
}

STRUCT_MAPPING = {
    "i8": "b",
    "u8": "B",
    "i16": "h",
    "u16": "H",
    "i32": "i",
    "u32": "I",
    "i64": "q",
    "u64": "Q",
    "f32": "f",
    "f64": "d",
    "byte": "B",
    "word": "H",
    "dword": "I",
    "qword": "Q",
}

MACRO_MAPPING = dict(
    CHAR_BIT={"type": "u8", "value": 8},
    MB_LEN_MAX={"type": "u8", "value": 16},
    CHAR_MIN={"type": "u8", "value": -128},
    CHAR_MAX={"type": "u8", "value": 127},
    SCHAR_MIN={"type": "i8", "value": -128},
    SCHAR_MAX={"type": "i8", "value": 127},
    SHRT_MIN={"type": "i16", "value": -32768},
    SHRT_MAX={"type": "i16", "value": 32767},
    INT_MIN={"type": "i32", "value": -2147483648},
    INT_MAX={"type": "i32", "value": 2147483647},
    LONG_MIN={"type": "i64", "value": -9223372036854775808},
    LONG_MAX={"type": "i64", "value": 9223372036854775807},
    LLONG_MIN={"type": "i64", "value": -9223372036854775808},
    LLONG_MAX={"type": "i64", "value": 9223372036854775807},
    UCHAR_MAX={"type": "u8", "value": 255},
    USHRT_MAX={"type": "u16", "value": 65535},
    UINT_MAX={"type": "u32", "value": 4294967295},
    ULONG_MAX={"type": "u64", "value": 18446744073709551615},
    ULLONG_MAX={"type": "u64", "value": 18446744073709551615},
    PTRDIFF_MIN={"type": "i64", "value": -9223372036854775808},
    PTRDIFF_MAX={"type": "i64", "value": 9223372036854775807},
    SIZE_MAX={"type": "u64", "value": 18446744073709551615},
    SIG_ATOMIC_MIN={"type": "i32", "value": -2147483648},
    SIG_ATOMIC_MAX={"type": "i32", "value": 2147483647},
    WCHAR_MIN={"type": "i32", "value": -2147483648},
    WCHAR_MAX={"type": "i32", "value": 2147483647},
    WINT_MIN={"type": "u32", "value": 0},
    WINT_MAX={"type": "u32", "value": 4294967295},
    FLT_RADIX={"type": "i32", "value": 2},
    DECIMAL_DIG={"type": "i32", "value": 21},
    FLT_DECIMAL_DIG={"type": "i32", "value": 9},
    DBL_DECIMAL_DIG={"type": "i32", "value": 17},
    LDBL_DECIMAL_DIG={"type": "i32", "value": 21},
    FLT_MIN={"type": "f32", "value": 1.17549e-38},
    FLT_MAX={"type": "f32", "value": 3.40282e38},
    FLT_EPSILON={"type": "f32", "value": 1.19209e-07},
    DBL_MIN={"type": "f64", "value": 2.22507e-308},
    DBL_MAX={"type": "f64", "value": 1.79769e308},
    DBL_EPSILON={"type": "f64", "value": 2.22045e-16},
    LDBL_MIN={"type": "f64", "value": 3.3621e-4932},
    LDBL_MAX={"type": "f64", "value": 1.18973e4932},
    LDBL_EPSILON={"type": "f64", "value": 1.0842e-19},
    FLT_TRUE_MIN={"type": "f32", "value": 1.4013e-45},
    DBL_TRUE_MIN={"type": "f64", "value": 4.94066e-324},
    LDBL_TRUE_MIN={"type": "f64", "value": 3.6452e-4951},
    FLT_DIG={"type": "i32", "value": 6},
    DBL_DIG={"type": "i32", "value": 15},
    LDBL_DIG={"type": "i32", "value": 18},
    FLT_MANT_DIG={"type": "i32", "value": 24},
    DBL_MANT_DIG={"type": "i32", "value": 53},
    LDBL_MANT_DIG={"type": "i32", "value": 64},
    FLT_MIN_EXP={"type": "i32", "value": -125},
    DBL_MIN_EXP={"type": "i32", "value": -1021},
    LDBL_MIN_EXP={"type": "i32", "value": -16381},
    FLT_MIN_10_EXP={"type": "i32", "value": -37},
    DBL_MIN_10_EXP={"type": "i32", "value": -307},
    LDBL_MIN_10_EXP={"type": "i32", "value": -4931},
    FLT_MAX_EXP={"type": "i32", "value": 128},
    DBL_MAX_EXP={"type": "i32", "value": 1024},
    LDBL_MAX_EXP={"type": "i32", "value": 16384},
    FLT_MAX_10_EXP={"type": "i32", "value": 38},
    DBL_MAX_10_EXP={"type": "i32", "value": 308},
    LDBL_MAX_10_EXP={"type": "i32", "value": 4932},
    FLT_ROUNDS={"type": "i32", "value": 1},
    FLT_EVAL_METHOD={"type": "i32", "value": 0},
    FLT_HAS_SUBNORM={"type": "i32", "value": 1},
    DBL_HAS_SUBNORM={"type": "i32", "value": 1},
    LDBL_HAS_SUBNORM={"type": "i32", "value": 1},
)

FUNCTION_IDENTITY = {
    "abs": [
        {"type": "dword", "value": 0x7FFFFFFF},
        {"type": "qword", "value": 0x7FFFFFFFFFFFFFFF},
    ],
    "labs": [
        {"type": "dword", "value": 0x7FFFFFFF},
        {"type": "qword", "value": 0x7FFFFFFFFFFFFFFF},
    ],
    "llabs": [
        {"type": "dword", "value": 0x7FFFFFFF},
        {"type": "qword", "value": 0x7FFFFFFFFFFFFFFF},
    ],
    "fabs": [
        {"type": "dword", "value": 0x7FFFFFFF},
        {"type": "qword", "value": 0x7FFFFFFFFFFFFFFF},
    ],
    "fabsf": [
        {"type": "dword", "value": 0x7FFFFFFF},
        {"type": "qword", "value": 0x7FFFFFFFFFFFFFFF},
    ],
    "fabsl": [
        {"type": "dword", "value": 0x7FFFFFFF},
        {"type": "qword", "value": 0x7FFFFFFFFFFFFFFF},
    ],
}


def extract_code_literal(token_value, cursor_type):
    # 从数字开始往后切，包括 16 进制和浮点数, 不要去掉任何字符
    m = re.match(
        r"(?P<sign>\+|\-)?"
        r"(?P<preffix>0x|0b)?"
        r"(?P<int_part>[0-9a-f]+)"
        r"(?P<frac_part>\.[0-9a-f]+)?"
        r"(?P<exp>((e|p)(\+|\-)?[0-9]+))?"
        r"(?P<suffix>[ulf]+)?",
        token_value,
    )
    if not m:
        raise ValueError(f"Invalid numeric format: {token_value}")

    sign, preffix, int_part, frac_part, exp, suffix = (
        m["sign"],
        m["preffix"],
        m["int_part"],
        m["frac_part"],
        m["exp"],
        m["suffix"],
    )

    # 处理基数和符号
    sign = sign if sign else ""
    base = (
        16
        if preffix == "0x"
        else (2 if preffix == "0b" else (8 if int_part.startswith("0") else 10))
    )

    int_bits = 32
    int_sign = "i"
    float_bits = 32
    if suffix is not None:
        if "u" in suffix:
            int_sign = "u"
        if "l" in suffix:
            int_bits = 64
            float_bits = 64

    if sign == "-" and int_sign == "u":
        raise ValueError(f"Invalid numeric format: {token_value}")

    if cursor_type is not None and len(cursor_type) > 0 and cursor_type in TYPE_MAPPING:
        value_type = TYPE_MAPPING[cursor_type]
        if value_type in ["f32", "f64"]:
            value = float(
                sign
                + int_part
                + (f".{frac_part}" if frac_part else "")
                + (exp if exp else "")
            )
        else:
            value = int(sign + int_part, base)
    else:
        match (frac_part, exp):
            case (None, None):
                value = int(sign + int_part, base)
                value_type = f"{int_sign}{int_bits}"
            case (frac_part, None):
                value = float(sign + int_part + frac_part)
                if value > 3.40282e38:
                    value_type = "f64"
                else:
                    value_type = f"f{float_bits}"
            case (None, exp):
                value = float(sign + int_part + exp)
                if value > 3.40282e38:
                    value_type = "f64"
                else:
                    value_type = f"f{float_bits}"
            case (frac_part, exp) if exp.startswith("p"):
                base = float(sign + int_part + frac_part)
                value = base * 2 ** int(exp[1:])
                if value > 3.40282e38:
                    value_type = "f64"
                else:
                    value_type = f"f{float_bits}"
            case (frac_part, exp) if exp.startswith("e"):
                value = float(sign + int_part + frac_part + exp)
                if value > 3.40282e38:
                    value_type = "f64"
                else:
                    value_type = f"f{float_bits}"

    return {
        # type 是 float 或 double
        "type": value_type,
        "value": value,
    }


async def extract_code_rodata(code_file):
    index = Index.create()
    translation_unit = index.parse(
        code_file, args=["-x", "c", "-std=c11", "-I/usr/include"]
    )

    done_literals = set()
    rodata_per_function = {}
    current_function = None

    def visit_node(cursor, function_context):
        nonlocal current_function

        # 如果节点是函数声明，进入新的函数上下文
        if cursor.kind == CursorKind.FUNCTION_DECL:
            current_function = cursor.spelling
            rodata_per_function[current_function] = []

            for child in cursor.get_children():
                visit_node(child, current_function)

        elif function_context:
            # 检查字符串字面量
            if (
                cursor.kind == CursorKind.STRING_LITERAL
                and cursor.spelling not in done_literals
            ):
                rodata_per_function[function_context].append(
                    {
                        "type": "string",
                        "value": json.loads(cursor.spelling),
                        # "location": node.location,
                    }
                )
            # 检查小数
            elif cursor.kind == CursorKind.FLOATING_LITERAL:
                tokens = list(cursor.get_tokens())
                for token in tokens:
                    if (
                        token.kind == TokenKind.LITERAL
                        and token.spelling not in done_literals
                    ):
                        try:
                            rodata_per_function[function_context].append(
                                extract_code_literal(
                                    token.spelling.lower(), cursor.type.spelling
                                )
                            )
                        except ValueError as ve:
                            continue
            elif cursor.kind in [
                CursorKind.FOR_STMT,
                CursorKind.COMPOUND_STMT,
                CursorKind.VAR_DECL,
            ]:
                for token in cursor.get_tokens():
                    if (
                        token.kind == TokenKind.LITERAL
                        and token.spelling not in done_literals
                    ):
                        try:
                            rodata_per_function[function_context].append(
                                extract_code_literal(
                                    token.spelling.lower(), cursor.type.spelling
                                )
                            )
                        except ValueError as ve:
                            continue

                    if token.kind == TokenKind.IDENTIFIER:
                        if token.spelling in FUNCTION_IDENTITY:
                            for item in FUNCTION_IDENTITY[token.spelling]:
                                rodata_per_function[function_context].append(item)

                        if token.spelling in MACRO_MAPPING:
                            rodata_per_function[function_context].append(
                                MACRO_MAPPING[token.spelling]
                            )
            for child in cursor.get_children():
                visit_node(child, function_context)
        else:
            for child in cursor.get_children():
                visit_node(child, function_context)

    # 开始遍历 AST，从根节点开始，无初始函数上下文
    visit_node(translation_unit.cursor, None)

    # 归并函数中出现的常量，里面可能会有值一样的
    for func_name, items in rodata_per_function.items():
        if len(items) == 0:
            continue

        rodata_per_function[func_name] = [
            dict(t) for t in {tuple(d.items()) for d in items}
        ]

    return {k: v for k, v in rodata_per_function.items() if v}


def get_op_width(op_str):
    ops = [op.strip() for op in op_str.split(",")]
    width = []
    for op in ops:
        if op.startswith("byte ptr"):
            width.append(1)
        elif op.startswith("word ptr"):
            width.append(2)
        elif op.startswith("dword ptr"):
            width.append(4)
        elif op.startswith("qword ptr"):
            width.append(8)
        elif op.startswith("xmmword ptr"):
            width.append(16)
        elif op.startswith("ymmword ptr"):
            width.append(32)
        elif op.startswith("zmmword ptr"):
            width.append(64)
        else:
            width.append(0)

    return width


def default2dict():
    return defaultdict(
        lambda: {
            "op": [],
            "width": [],
            "syn": set(),
            "reg": [],
        }
    )


async def extract_function_ref(elf: ELFFile):
    # 获取符号表，提取函数的地址范围
    functions_addresses = {}
    symtab = elf.get_section_by_name(".symtab")
    if symtab:
        for symbol in symtab.iter_symbols():
            if symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_size"] > 0:
                func_name = symbol.name
                start_addr = symbol["st_value"]
                end_addr = start_addr + symbol["st_size"]
                functions_addresses[func_name] = (start_addr, end_addr)
    else:
        return {}

    # 获取 .text 段
    text_section = elf.get_section_by_name(".text")
    if not text_section:
        return {}

    text_data = text_section.data()
    text_addr = text_section["sh_addr"]

    # 确定架构
    arch = elf.get_machine_arch()
    if arch == "x64":
        md = Cs(CS_ARCH_X86, CS_MODE_64)
    elif arch == "x86":
        md = Cs(CS_ARCH_X86, CS_MODE_32)
    else:
        return {}

    md.detail = True

    # 存储每个函数中对 .rodata 段的引用
    functions_refs = defaultdict(default2dict)

    for func_name, (start_addr, end_addr) in functions_addresses.items():
        # 计算函数在 .text 段中的偏移
        func_offset = start_addr - text_addr
        func_size = end_addr - start_addr
        func_code = text_data[func_offset : func_offset + func_size]

        # todo: 一个地址可能有多个引用，需要处理
        for insn in md.disasm(func_code, start_addr):
            op_sizes = get_op_width(insn.op_str)

            actual_address = None
            for op, op_size in zip(insn.operands, op_sizes):
                if op.type == CS_OP_MEM:
                    mem = op.mem
                    if mem.disp != 0 and md.reg_name(mem.base) == "rip":
                        rip_relative_offset = mem.disp
                        rip_value = insn.address + insn.size
                        actual_address = rip_value + rip_relative_offset
                        functions_refs[func_name][actual_address]["op"].append(
                            insn.mnemonic
                        )
                        functions_refs[func_name][actual_address]["width"].append(
                            op_size
                        )

                        if insn.mnemonic.startswith("p") and insn.mnemonic[4:] == "s":
                            # 无符号
                            functions_refs[func_name][actual_address]["syn"].add(False)
                        elif insn.mnemonic.startswith("p") and insn.mnemonic[4:] == "u":
                            # 无符号
                            functions_refs[func_name][actual_address]["syn"].add(True)

                        if not hasattr(insn, "detail") or insn.detail is None:
                            continue

                        eflags = insn.detail.x86.eflags
                        if (
                            eflags & x86.X86_EFLAGS_MODIFY_SF
                            or eflags & x86.X86_EFLAGS_MODIFY_OF
                        ):
                            functions_refs[func_name][actual_address]["syn"].add(True)

            if actual_address is None:
                continue

            functions_refs[func_name][actual_address]["reg"].append([])
            for op, op_size in zip(insn.operands, op_sizes):
                if op.type == CS_OP_REG:
                    functions_refs[func_name][actual_address]["reg"][-1].append(
                        md.reg_name(op.reg)
                    )

    # 聚合所有的 syn
    for func_name, addrs in functions_refs.items():
        for actual_address in addrs:
            functions_refs[func_name][actual_address]["syn"] = list(
                functions_refs[func_name][actual_address]["syn"]
            )

    return functions_refs


def partial_match(item_value, rodata_slice, tolerance=0):
    # 计算有多少个字节不同
    mismatches = sum(
        1 for a, b in itertools.zip_longest(item_value, rodata_slice) if a != b
    )
    # 如果不匹配的字节数在容许范围内，则返回 True
    return mismatches <= tolerance


async def extract_function_rodata(elf: ELFFile, functions_refs, code_rodata):
    # 获取 .rodata 段
    rodata_section = elf.get_section_by_name(".rodata")
    if not rodata_section:
        # rodata_addr, rodata_data.hex(), rodata_values, num_missing
        return None, None, {}, 0
    rodata_data = rodata_section.data()
    rodata_addr = rodata_section["sh_addr"]

    # 利用 code_rodata 中的信息构建 byte 到值的映射
    # 从 float 和 double 转换成字节
    # 将 string_literal 的值转换成字节
    rodata_map = defaultdict(dict)
    for func_name, items in code_rodata.items():
        for item in items:
            if item["type"] == "f32":
                # 值得注意的是，字面量的 f32 事实上也可能实际存储为 f64
                # remove the trailing f
                # item["value"] = item["value"].rstrip("f").rstrip("F")
                rovalue = [
                    ("f32", struct.pack("<f", item["value"])),
                    ("f64", struct.pack("<d", item["value"])),
                ]
            elif item["type"] == "f64":
                # 值得注意的是，字面量的 f64 事实上也可能实际存储为 f32
                # remove the trailing l / L
                # item["value"] = item["value"].rstrip("l").rstrip("L")

                # 如果值很大，不会被存为 f32
                if item["value"] > 3.40282e38:
                    rovalue = struct.pack("<d", item["value"])
                else:
                    rovalue = [
                        ("f32", struct.pack("<f", item["value"])),
                        ("f64", struct.pack("<d", item["value"])),
                    ]
            elif item["type"] == "string":
                rovalue = item["value"]
            elif item["type"] == "byte":  # u8
                rovalue = struct.pack("<B", item["value"])
            elif item["type"] == "word":  # u16
                rovalue = struct.pack("<H", item["value"])
            elif item["type"] == "dword":  # u32
                rovalue = struct.pack("<I", item["value"])
            elif item["type"] == "qword":  # u64
                rovalue = struct.pack("<Q", item["value"])
            # elif item["type"] == "xmmword":
            # rovalue = struct.pack("<16B", item["value"])
            # elif item["type"] == "ymmword":
            # rovalue = struct.pack("<32B", item["value"])
            # elif item["type"] == "zmmword":
            # rovalue = struct.pack("<64B", item["value"])
            else:
                continue
            if isinstance(rovalue, list):
                for ro_type, ro_key in rovalue:
                    rodata_map[func_name][ro_key] = item
                    rodata_map[func_name][ro_key]["addition"] = rovalue
            else:
                rodata_map[func_name][rovalue] = item

    # 对 rodata_map 中的值进行排序，以便后续查找，以 rovalue 的长度进行排序，越长的越靠前
    for func_name in rodata_map:
        rodata_map[func_name] = dict(
            sorted(rodata_map[func_name].items(), key=lambda x: len(x[0]), reverse=True)
        )

    # 构建地址到值的映射，记录类型
    rodata_values = defaultdict(dict)

    # 利用 rodata_map 中的信息，利用 functions_refs 中的地址，查找 .rodata 段中的值
    num_missing = 0
    for func_name, addrs in functions_refs.items():
        founded_addr = set()
        for addr, item in addrs.items():
            found = False
            offset = addr - rodata_addr
            for item_value, func_item in rodata_map[func_name].items():
                rodata_read = rodata_data[offset : offset + len(item_value)]
                if item_value == rodata_read:
                    if func_item["type"] == "string" and "lea" in item["op"]:
                        item["type"] = "string"
                        item["value"] = func_item["value"]

                        rodata_values[func_name][addr] = {
                            "type": "string",
                            "value": func_item["value"],
                        }
                    elif func_item["type"] != "string":
                        if "addition" in func_item:
                            # 移除其他的值
                            for addition_type, addition_key in func_item["addition"]:
                                if addition_key == item_value:
                                    func_item["type"] = addition_type
                                else:
                                    rodata_map[func_name].pop(addition_key, None)
                            # 移除 addtion
                            func_item.pop("addition", None)

                        max_width = max(item["width"]) if len(item["width"]) > 0 else len(item_value)
                        # print(item_value, func_item["type"], max_width, len(item_value))
                        if max_width > len(item_value):
                            # convert to array
                            array_size = max_width // len(item_value)
                            array_struct = STRUCT_MAPPING[func_item["type"]]
                            try:
                                array_value = struct.unpack(
                                    f"<{array_size}{array_struct}",
                                    rodata_data[offset : offset + max_width],
                                )
                                item["type"] = f"{func_item['type']}[{array_size}]"
                                item["value"] = list(array_value)
                            except struct.error:
                                item["type"] = func_item["type"]
                                item["value"] = func_item["value"]
                        else:
                            item["type"] = func_item["type"]
                            item["value"] = func_item["value"]

                        rodata_values[func_name][addr] = {
                            "type": item["type"],
                            "value": item["value"],
                        }

                        if item["type"] in ["i32", "f32", "f64"]:
                            item["syn"].append(True)

                    # rodata_values[func_name][addr] = func_item
                    founded_addr.add(addr)
                    found = True
                    break

            if found:
                continue

        not_founded = addrs.keys() - founded_addr

        # 将 lea 指令的地址转换成字符串
        for addr in not_founded:
            if "lea" in addrs[addr]["op"]:
                offset = addr - rodata_addr
                end_offset = rodata_data[offset:].find(b"\x00")
                if end_offset != -1:
                    found_value = rodata_data[offset : offset + end_offset]
                    rodata_values[func_name][addr] = {
                        "type": "string",
                    }
                    rodata_values[func_name][addr]["value"] = json.dumps(
                        found_value.decode("utf-8")
                    )
                founded_addr.add(addr)

        # 将其他未知的地址转换成字节码
        for addr, item in addrs.items():
            if addr in founded_addr:
                continue

            offset = addr - rodata_addr
            max_width = max(item["width"])
            rodata_read = rodata_data[offset : offset + max_width]

            # 转成 bytes to hex
            rodata_read = [int(b) for b in rodata_read]

            rodata_values[func_name][addr] = {
                "type": f"byte[{max_width}]",
                "value": rodata_read,
            }

        not_founded = addrs.keys() - founded_addr
        for addr in not_founded:
            num_missing += 1

    # rodata_values 中的 addr 转换成 hex
    for func_name, items in rodata_values.items():
        rodata_values[func_name] = {
            hex(addr): item for addr, item in items.items()
        }
    
    return rodata_addr, rodata_data.hex(), rodata_values, num_missing


async def parse_ro_data(code_file, binary_file):
    # 存储每个函数中的常量
    functions_constants = await extract_code_rodata(code_file)

    # 使用 pyelftools 解析二进制文件
    with open(binary_file, "rb") as f:
        elf = ELFFile(f)

        functions_refs = await extract_function_ref(elf=elf)
        (
            rodata_addr,
            rodata_data,
            rodata_parsed,
            num_missing,
        ) = await extract_function_rodata(
            elf=elf, functions_refs=functions_refs, code_rodata=functions_constants
        )

        return rodata_addr, rodata_data, rodata_parsed, num_missing
