import json
import asyncio


async def main():
    c_funcs = set()
    c_funcs_list = []
    with open("data/decompile-eval-gcc-rodata.json", "r") as f:
        data_all = json.load(f)

        for item in sorted(data_all, key=lambda x:x['task_id']):
            c_func = item['c_func']
            c_func = c_func.replace("func0", f"func{item['task_id']:03d}")
            if c_func not in c_funcs:
                c_funcs.add(c_func)
                c_funcs_list.append(c_func)
    
    with open("test.c", "w") as f:
        f.write("\n\n\n".join(c_funcs_list))
    


if __name__ == "__main__":
    asyncio.run(main())
