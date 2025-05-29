from collections import defaultdict
import re
import random
import orjson as json
import traceback
from tqdm import tqdm


def get_tool_description(function_name, label_tag, type_tag):
    description = [
        "Parse data from a label in preprocessed assembly using a guessed data type. Original addresses are replaced with labels like D1, D2.",
        "Extract information from placeholders in optimized machine code, where original memory references have already been replaced with symbolic identifiers such as D1, D2.",
        "Interpret the content of markers within compiled binaries, employing inferred variable types; original locations have already been denoted by tags like D1, D2.",
        "Decode values from symbols in post-processed assembly language, where initial addresses have already been abstracted into labels such as D1, D2.",
        "Translate data segments from assembly mnemonics using assumed type definitions; actual addresses have already been symbolized with placeholders like D1, D2.",
        "Retrieve datasets from marked sections in assembly output, where direct address calls have already been substituted with alias labels similar to D1, D2.",
        "Convert data entries from tags in refined assembly scripts using heuristic data typing; original pointers have already been exchanged for labels akin to D1, D2.",
        "Unpack data structures from labeled areas in processed assembly instructions, where original reference points have already been replaced with symbols like D1, D2.",
        "Disassemble data items from tagged zones in prepped assembly code, inferring data categories, and numeric addresses have already been swapped for labels like D1, D2.",
        "Read data elements from designated regions in assembly source, assuming appropriate data types, and original addresses have already been replaced with labels such as D1, D2.",
        "Map data fields from label-marked sections in transformed assembly listings, guessing at data formats, and original positions have already been replaced with labels like D1, D2.",
        "从优化过的机器代码中的占位符提取信息，其中原始内存引用已经被替换为如D1、D2等符号标识符。",
        "解释编译后的二进制文件中的标记内容，使用推测的变量类型；原始位置已经被如D1、D2等标签表示。",
        "从后处理汇编语言中的符号解码值，其中初始地址已经被抽象成如D1、D2等标签。",
        "使用假设的类型定义从汇编助记符中翻译数据段；实际地址已经被象征性地用如D1、D2等占位符代替。",
        "从汇编输出中标记的部分检索数据集，其中直接地址调用已经被类似于D1、D2的别名标签所替代。",
        "从精炼过的汇编脚本中的标签转换数据项，使用启发式数据类型；原始指针已经被换成如D1、D2等标签。",
        "从处理过的汇编指令中标记区域解包数据结构，其中原始参考点已经被如D1、D2等符号替换。",
        "从预处理过的汇编代码中标记区解汇编数据项，推断数据类别，并且数字地址已经被如D1、D2等标签交换。",
        "从汇编源码中指定的区域内读取数据元素，假设适当的数据类型，并且原始地址已经被如D1、D2等标签替换。",
        "从转换过的汇编列表中标记部分映射数据字段，猜测数据格式，并且原始位置已经被如D1、D2等标签替换。",
    ]

    label_desc = [
        "The specified label in the preprocessed assembly where data is stored (e.g., D1, D2).",
        "The designated marker in the preprocessed assembly code indicating the storage location for data (e.g., D1, D2).",
        "A predefined tag within the assembly instructions that points to a data storage area (e.g., D1, D2).",
        "The reference label used in the assembly code after preprocessing to denote data locations (e.g., D1, D2).",
        "An identifier in the prepared assembly language code that represents a data holding point (e.g., D1, D2).",
        "The labeled section in the processed assembly program where variables are kept (e.g., D1, D2).",
        "A specific address label in the assembly source code marking the place of data residence (e.g., D1, D2).",
        "The named segment in the compiled assembly instructions where information is stored (e.g., D1, D2).",
        "The indexed label in the post-preprocessing assembly text that refers to data storage (e.g., D1, D2).",
        "A unique symbol in the assembly coding that identifies the location of stored data items (e.g., D1, D2).",
        "The coded label in the assembly directive set that signifies the position of data elements (e.g., D1, D2). ",
        "在预处理后的汇编代码中标记存储数据位置的指定标识（例如，D1, D2）。",
        "在汇编指令中预先定义指向数据存储区域的标签（例如，D1, D2）。",
        "预处理后汇编代码中用来表示数据位置的引用标签（例如，D1, D2）。",
        "准备好的汇编语言代码中代表数据存放点的标识符（例如，D1, D2）。",
        "经过处理的汇编程序中标注变量存放位置的标记部分（例如，D1, D2）。",
        "汇编源代码中标识数据驻留位置的具体地址标签（例如，D1, D2）。",
        "编译后的汇编指令中存放信息的命名段落（例如，D1, D2）。",
        "后预处理汇编文本中指代数据存储位置的索引标签（例如，D1, D2）。",
        "汇编编码中唯一识别已存储数据项位置的符号（例如，D1, D2）。",
        "汇编指令集中表示数据元素位置的编码标签（例如，D1, D2）。",
    ]

    type_desc = [
        "The guessed data type for the specified label, which can be described as a base type (e.g., i8) or an array type (e.g., i8[8]). Base types are strictly limited to the following: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "The inferred data type for the given label can be characterized as either a fundamental type (such as i8) or an array type (like i8[8]). Fundamental types are confined to: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "For the indicated label, the presumed data type may be specified as a primitive type (for instance, i8) or an aggregate type in the form of an array (i.e., i8[8]). Primitive types are restricted to the set: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "The deduced data type associated with the specific label is expressible as either a scalar type (e.g., i8) or an array type (e.g., i8[8]). Scalar types are limited to this list: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "When labeling a data type, it can be categorized into basic types (like i8) or array-based types (such as i8[8]). Basic types are strictly chosen from: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "The anticipated data type for the particular label encompasses either a simple type (e.g., i8) or an array-oriented type (e.g., i8[8]). Simple types adhere to this enumeration: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "With respect to the designated label, the expected data type might be a foundational type (example, i8) or an array type (example, i8[8]). Foundational types are exclusively these: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "The predicted data type for the selected label could be defined as a core type (e.g., i8) or an array structure (e.g., i8[8]). Core types are bound by this selection: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "In terms of the labeled data type, one can anticipate it to be a primary type (such as i8) or an array type (like i8[8]). Primary types are limited to: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "The assumed data type corresponding to the label in question is either an elementary type (e.g., i8) or an array type (e.g., i8[8]). Elementary types are specifically these: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "The estimated data type for the assigned label should be understood as a root type (e.g., i8) or an array type (e.g., i8[8]). Root types are precisely limited to: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.",
        "对于给定标签，推测的数据类型可以描述为基础类型（例如，i8）或数组类型（如，i8[8]）。基础类型严格限制为以下几种：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "针对指定标签，推断出的数据类型可被描述为基本类型（比如，i8）或以数组形式存在的聚合类型（即，i8[8]）。基本类型仅限于：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "与特定标签关联的推导数据类型可以用标量类型（例如，i8）或数组类型（例如，i8[8]）来表达。标量类型限定为以下列表：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "在标注数据类型时，它可以分类为基本类型（如，i8）或基于数组的类型（比如，i8[8]）。基本类型严格选择自：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "特定标签的预期数据类型包括简单类型（例如，i8）或以数组为导向的类型（例如，i8[8]）。简单类型遵循此枚举：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "关于指定的标签，期望的数据类型可能是基础类型（例子，i8）或数组结构类型（例子，i8[8]）。基础类型专门是这些：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "针对选定标签预测的数据类型可以定义为核心类型（例如，i8）或数组结构（例如，i8[8]）。核心类型受此选择限制：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "就标记的数据类型而言，可以预期其为初级类型（例如，i8）或数组类型（如，i8[8]）。初级类型限于：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "与问题标签对应假设的数据类型要么是元素类型（例如，i8），要么是数组类型（例如，i8[8]）。元素类型具体为：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
        "对于分配的标签，估计的数据类型应理解为基础类型（例如，i8）或数组类型（例如，i8[8]）。基础类型精确地限制为：i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword 和 string。",
    ]

    instructions = [
        "What is the c source code of the assembly code below"
        + random.choice([":", "?"]),
        "Translate the below assembly code into C programming language:",
        "Converte this assembly snippet into its equivalent C code:",
        "Please show me the C implementation that matches the assembly code listed below:",
        "What would the C source look like for the assembly code provided here"
        + random.choice([":", "?"]),
        "Demonstrate the C code that generates the assembly output below:",
        "下面的汇编指令对应的C源代码是什么" + random.choice(["：", ":"]),
        "请将以下的汇编代码转换成C语言代码" + random.choice(["：", ":"]),
        "请将这段汇编代码片段转换成等效的C代码" + random.choice(["：", ":"]),
        "请显示与下面列出的汇编代码相匹配的C语言实现" + random.choice(["：", ":"]),
        "提供的汇编代码用C语言表示会是什么样子" + random.choice(["：", ":"]),
        "这段汇编代码用C语言表示是什么样的" + random.choice(["？", "：", ":", "?"]),
        "请从给定的汇编指令推导出C源代码" + random.choice(["：", ":"]),
    ]

    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": random.choice(description),
            "parameters": {
                "type": "object",
                "properties": {
                    label_tag: {
                        "type": "string",
                        "description": random.choice(label_desc),
                    },
                    type_tag: {
                        "type": "string",
                        "description": random.choice(type_desc),
                    },
                },
                "required": [label_tag, type_tag],
                "additionalProperties": False,
            },
        },
    }, random.choice(instructions)


def convert_rodata(
    rodata: dict, address_mapping: dict, rodata_range_left, rodata_range_right
):
    result = []
    address_mapping = {hex(v["addr"]): v["label"] for v in address_mapping.values()}

    for addr, item in rodata.items():
        if addr not in address_mapping:
            continue

        if not address_mapping[addr].startswith("D"):
            continue

        m = re.match(r"(?P<type>\w+)(?P<size>\[\d+\])?", item["type"])
        if not m:
            continue

        data_type = m.group("type")
        data_size = m.group("size")

        if data_size == "[1]":
            data_size = None
            item["type"] = data_type

            if len(item["value"]) == 1:
                item["value"] = item["value"][0]
            else:
                raise ValueError(f"Invalid data size: {item['value']}")
        elif data_size is not None:
            data_size = int(data_size[1:-1])
            if data_size != len(item["value"]):
                raise ValueError(f"Invalid data size: {item['value']}")

        if data_type == "string":
            result.append(
                {
                    "address": addr,
                    "label": address_mapping[addr],
                    "type": item["type"],
                    "value": item["value"],
                }
            )
        if data_size:
            if data_type == "byte":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": item["type"],
                        "value": [f"0x{v:02x}" for v in item["value"]],
                    }
                )
            elif data_type == "word":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": item["type"],
                        "value": [f"0x{v:04x}" for v in item["value"]],
                    }
                )
            elif data_type == "dword":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": item["type"],
                        "value": [f"0x{v:08x}" for v in item["value"]],
                    }
                )
            elif data_type == "qword":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": item["type"],
                        "value": [f"0x{v:016x}" for v in item["value"]],
                    }
                )
            else:
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": item["type"],
                        "value": item["value"],
                    }
                )
        else:
            if data_type == "byte":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": data_type,
                        "value": f"0x{item['value']:02x}",
                    }
                )
            elif data_type == "word":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": data_type,
                        "value": f"0x{item['value']:04x}",
                    }
                )
            elif data_type == "dword":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": data_type,
                        "value": f"0x{item['value']:08x}",
                    }
                )
            elif data_type == "qword":
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": data_type,
                        "value": f"0x{item['value']:016x}",
                    }
                )
            else:
                result.append(
                    {
                        "address": addr,
                        "label": address_mapping[addr],
                        "type": data_type,
                        "value": item["value"],
                    }
                )

    return result


def render_rodata(item, use_label=False):
    m = re.match(r"(?P<type>\w+)(?P<size>\[\d+\])?", item["type"])
    if not m:
        raise ValueError(f"Invalid type: {item['type']}")

    data_type = m.group("type")
    data_size = m.group("size")
    data_value = item["value"]

    if use_label:
        info = f'"{item["label"]}"'
    else:
        info = f'"{item["address"]}"'

    if item["type"] == "string":
        value = data_value
        result = f"{{{info}: {value}}}"
    elif data_size:
        if data_type in ["byte", "word", "dword", "qword"]:
            value = ",".join(data_value)
            result = f"{{{info}: [{value}]}}"
        else:
            value = json.dumps(
                data_value  # , ensure_ascii=False
            ).decode("utf-8")
            result = f"{{{info}: {value}}}"
    else:
        if data_type in ["byte", "word", "dword", "qword"]:
            value = data_value
            result = f"{{{info}: {value}}}"
        else:
            value = json.dumps(
                data_value  # , ensure_ascii=False
            ).decode("utf-8")
            result = f"{{{info}: {value}}}"

    return result


def main():
    count_skip = 0
    count_load = 0
    count_struct = 0

    count = defaultdict(int)
    type_count = defaultdict(int)
    with (
        open("data/train_real_compilable.jsonl", "r") as file_input,
        open("data/train.jsonl", "wb") as file_output,
    ):
        pbar = tqdm(enumerate(file_input), desc="Processing data")
        for idx, line in pbar:
            if sum(count.values()) >= 150000 * 4:
                print("Reached 150k")
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print("Error in line:", idx)
                continue

            function_def = item["function_def"]
            function_name = item["function_name"]
            function_structs = item["function_structs"]

            if function_def.count("\n") < 3:
                count_skip += 1
                continue

            has_rodata = False
            has_struct = False

            for opt in ["O0", "O1", "O2", "O3"]:
                if opt not in item:
                    continue

                LABEL_TAG = random.choice(["label", "data_label", "tag", "data_tag"])
                TYPE_TAG = random.choice(
                    ["type", "data_type", "datatype", "type_guessed", "type_infered"]
                )
                TOOL_NAME = random.choice(
                    [
                        "parse_data",
                        "unpack_data",
                        "disassemble_data",
                        "read_data",
                        "extract_symbolic_data",
                        "retrieve_data",
                    ]
                )
                TOOL_DEF, INSTRUCTION = get_tool_description(
                    TOOL_NAME, LABEL_TAG, TYPE_TAG
                )

                asm = item[opt]["asm"].strip()
                asm_labeled = item[opt]["asm_labeled"].strip()

                conversation_tool_label = {
                    "tools": json.dumps(
                        [TOOL_DEF["function"]]  # , ensure_ascii=False
                    ).decode("utf-8"),
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{INSTRUCTION}\n\n{asm_labeled}",
                        }
                    ],
                }

                # process rodata
                try:
                    function_name = item["function_name"]
                    address_mapping = item[opt]["address_mapping"]
                    rodata_parsed = item[opt]["rodata_parsed"]

                    if (item[opt].get("rodata_addr", None) is not None) and (
                        item[opt].get("rodata_data", None) is not None
                    ):
                        rodata_range_left, rodata_range_right = (
                            item[opt]["rodata_addr"],
                            item[opt]["rodata_addr"]
                            + len(item[opt]["rodata_data"]) // 2,
                        )
                    else:
                        rodata_range_left, rodata_range_right = (0, 0)

                    if (
                        function_name in rodata_parsed
                        and len(rodata_parsed[function_name]) > 0
                    ):
                        rodata_parsed = convert_rodata(
                            rodata_parsed[function_name],
                            address_mapping,
                            rodata_range_left,
                            rodata_range_right,
                        )

                        rodata_parsed = {v["label"]: v for v in rodata_parsed}.values()

                        # 应该按照在汇编中出现的顺序
                        index_mapping = {}
                        m = re.findall(r"D\d+", asm_labeled)
                        for match in m:
                            if match not in index_mapping:
                                index_mapping[match] = len(index_mapping)

                        rodata_parsed = sorted(
                            rodata_parsed, key=lambda x: index_mapping[x["label"]]
                        )

                        for v in rodata_parsed:
                            type_count[v["type"]] += 1

                        if len(rodata_parsed) > 0:
                            # tool calls
                            conversation_tool_label["messages"].append(
                                {
                                    "role": "tool_calls",
                                    "content": json.dumps(
                                        [
                                            {
                                                "name": TOOL_NAME,
                                                "arguments": {
                                                    LABEL_TAG: v["label"],
                                                    TYPE_TAG: v["type"],
                                                },
                                            }
                                            for v in rodata_parsed
                                        ],
                                        # ensure_ascii=False,
                                    ).decode("utf-8"),
                                }
                            )

                            # tool results
                            for v in rodata_parsed:
                                conversation_tool_label["messages"].append(
                                    {
                                        "role": "tool",
                                        "content": render_rodata(v, use_label=True),
                                    }
                                )
                            has_rodata = True
                except ValueError as e:
                    count_skip += 1
                    continue
                except Exception as e:
                    count_skip += 1
                    traceback.print_exc()
                    continue

                # construct the conversation
                conversation_tool_label["messages"].append(
                    {
                        "role": "assistant",
                        "content": function_def,
                    }
                )

                if len(function_structs) > 0:
                    has_struct = True

                assert conversation_tool_label["messages"][-1]["role"] == "assistant"

                file_output.write(json.dumps(conversation_tool_label) + b"\n")
                count[opt] += 1

            if has_rodata:
                count_load += 1
            if has_struct:
                count_struct += 1

            pbar.set_postfix(
                {
                    "skip": count_skip,
                    "load": count_load,
                    "struct": count_struct,
                    **count,
                }
            )

    print("Load:", count_load, "Struct:", count_struct)

    for k, v in count.items():
        print(k + ":", v)

    for k, v in type_count.items():
        print(k, v)


if __name__ == "__main__":
    main()
