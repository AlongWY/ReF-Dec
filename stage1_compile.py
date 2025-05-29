from asyncio import subprocess
import argparse
import orjson as json
import os
import asyncio
import re
import tempfile
import aiofiles
from multiprocessing import cpu_count
from tqdm.asyncio import tqdm
from compiler import (
    compile_with_optimization,
    parse_function_structs,
    parse_ro_data,
    disassemble,
    format_asm,
)

os.environ["USE_TORCH"] = "FALSE"
from datasets import load_from_disk


async def compile_code(tmpdir, code_file, function_name, optimization_level):
    binary_file = os.path.join(tmpdir, f"code{optimization_level}.out")

    # compile the code
    await compile_with_optimization(code_file, binary_file, optimization_level)

    # Parse the DWARF information to get the variable locations
    rodata_addr, rodata_data, rodata_parsed, num_missing = await parse_ro_data(
        code_file, binary_file
    )

    # Disassemble the binary to get the assembly code
    raw_asm = await disassemble(binary_file, function_name)

    # Format the assembly code
    asm_unlabeled, _ = await format_asm(raw_asm, function_name, False)
    asm_labeled, address_mapping = await format_asm(raw_asm, function_name, True)

    return (
        asm_unlabeled,
        asm_labeled,
        address_mapping,
        rodata_addr,
        rodata_data,
        rodata_parsed,
        num_missing,
        optimization_level,
    )


async def build_item(idx, synth_deps, function_def, function_name):
    # 在 { 之前移除掉 static 和 inline
    function_def, remain = function_def.split("{", maxsplit=1)
    function_def = (
        function_def.replace("static", "")
        .replace("inline", "")
        .replace("\n", " ")
        .strip()
    )
    remain, right_bracket = remain.rsplit("}", maxsplit=1)
    # remain 移除 # 1 "filename.c" 类似的注释
    remain = re.sub(r"#\s+\d+\s+\"[^\"]+\"", "", remain)
    function_def += " {" + remain + "\n}"

    # replace multiple \n to one \n
    function_def = re.sub("\n+", "\n", function_def)

    with tempfile.TemporaryDirectory(
        dir="/run/user/10000", prefix="exebench-"
    ) as tmpdir:
        code_file = os.path.join(tmpdir, "code.c")
        function_file = os.path.join(tmpdir, "function.c")

        with open(function_file, "w") as f:
            f.write(function_def)

        # format function
        proc_compile = await asyncio.create_subprocess_exec(
            "clang-format",
            f"--style={{ColumnLimit: {len(function_def)}}}",
            function_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        formatted_function, _ = await proc_compile.communicate()
        formatted_function = formatted_function.decode("utf-8")

        if formatted_function.count("\n") < 3:
            return None

        # 代码中一条 stmt 不能跨行
        if synth_deps is not None:
            synth_deps = synth_deps.strip()
            full_code = synth_deps + "\n" + formatted_function.strip() + "\n\n"
        else:
            full_code = formatted_function.strip() + "\n\n"

        with open(code_file, "w") as f:
            f.write(full_code)

        parsed_structs = await parse_function_structs(code_file, function_name)

        tasks = [
            asyncio.create_task(
                compile_code(tmpdir, code_file, function_name, optimization)
            )
            for optimization in ["O0", "O1", "O2", "O3"]
        ]

        asm_codes = {
            "idx": idx,
            "synth_deps": synth_deps,
            "function_def": formatted_function,
            "function_name": function_name,
            "function_structs": parsed_structs,
        }
        error = None
        for example in asyncio.as_completed(tasks):
            try:
                (
                    asm_unlabeled,
                    asm_labeled,
                    address_mapping,
                    rodata_addr,
                    rodata_data,
                    rodata_parsed,
                    num_missing,
                    optimization_level,
                ) = await example
                if asm_unlabeled is not None and asm_labeled is not None:
                    asm_codes[optimization_level] = {
                        "asm": asm_unlabeled,
                        "asm_labeled": asm_labeled,
                        "num_missing": num_missing,
                        "address_mapping": address_mapping,
                        "rodata_addr": rodata_addr,
                        "rodata_data": rodata_data,
                        "rodata_parsed": rodata_parsed,
                    }
            except Exception as e:
                error = e
                pass

        if len(asm_codes) == 5:
            raise error

        return asm_codes


async def producer_func(queue: asyncio.Queue, dataset, num_workers):
    for item in dataset:
        await queue.put(item)

    for _ in range(num_workers):
        await queue.put(None)


async def consumer_func(queue: asyncio.Queue, file, state, pbar):
    while True:
        item = await queue.get()  # 从队列中获取任务
        if item is not None:
            try:
                code_asm = None
                code_asm = await build_item(
                    idx=item["idx"],
                    synth_deps=item["synth_deps"],
                    function_def=item["func_def"],
                    function_name=item["fname"],
                )
                if code_asm is not None:
                    await file.write(json.dumps(code_asm) + b"\n")
                    state["Done"] += 1
                    if state["Done"] % 500 == 0:
                        await file.flush()
            except Exception as e:
                state["Err"] += 1
                if not isinstance(e, ValueError):
                    import traceback
                    import json as json_dbg

                    traceback.print_exc()
                    print(f"Error: {e}")
                    print(json_dbg.dumps(item, indent=2))
                state["Last Err"] = f"[{item['idx']}]: {e}"
            pbar.set_postfix(state)
            pbar.update(1)
        else:
            break


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=300000, required=False)
    parser.add_argument("-s", "--skip", type=int, default=0, required=False)
    parser.add_argument("--num_workers", type=int, default=64, required=False)
    parser.add_argument(
        "--data_dir", type=str, default="data/exebench", required=False
    )
    parser.add_argument(
        "--split", type=str, default="train_real_compilable", required=False
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_real_compilable.jsonl",
        required=False,
    )
    args = parser.parse_args()

    # check file exists and skip existing files
    ids_built = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in tqdm(f):
                try:
                    example = json.loads(line)
                    ids_built.add(example["idx"])
                except json.JSONDecodeError:
                    print(f"Error: {line}")
                    exit(1)

    dataset = load_from_disk(os.path.join(args.data_dir, args.split))
    dataset = dataset.filter(
        lambda item: item["func_def"].count("\n") >= 3,
        num_proc=cpu_count(),
        desc="Filtering functions with more than 3 lines",
    )
    # add index column
    dataset = dataset.map(
        lambda item, idx: {"idx": idx},
        with_indices=True,
        batched=True,
        num_proc=cpu_count(),
        desc="Adding index",
    )
    print(f"Total examples: {len(dataset)}")

    ids_to_build = sorted(set(range(args.num)) - ids_built)
    ids_to_build = ids_to_build[args.skip :]
    dataset = dataset.select(ids_to_build, keep_in_memory=True)
    num_examples = len(dataset)
    queue = asyncio.Queue(maxsize=max(args.num_workers * 1024, 8192))
    state = {"Done": 0, "Err": 0}
    print(f"Processing examples: {num_examples}")
    pbar = tqdm(total=len(dataset), dynamic_ncols=True)

    async with aiofiles.open(args.output, "ab") as file:
        workers = [
            asyncio.create_task(consumer_func(queue, file, state, pbar))
            for i in range(args.num_workers)
        ]
        await producer_func(queue, dataset, args.num_workers)
        await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())