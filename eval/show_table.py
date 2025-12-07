import io
import json
import asyncio
from pathlib import Path
import polars as pl
from great_tables import GT, md, html
from statistics import mean


def to_markdown(df: pl.DataFrame) -> str:
    buf = io.StringIO()
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        print(df, file=buf)
    buf.seek(0)
    return buf.read()


async def main():
    data = []
    top_dir = Path("results")
    for model in [
        "deepseek-chat",
        "qwen3-max-preview",
        "qwen3-max",
        "qwen-plus",
        "qwen3-coder-plus",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "ylfeng_ReF-Decompile_messages_sem32",
    ]:
        with open(top_dir / f"{model}.jsonl", "r") as f, open(top_dir / f"{model}_raw.jsonl", "r") as f_raw:
            num_cycles = []
            num_run = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
            num_run_raw = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
            total_num = 0
            for line in f:
                item = json.loads(line)
                num_cycles.append(
                    len([m for m in item["messages"] if m["role"] == "assistant"])
                )
                opt_state = item["type"]
                num_run[opt_state] += item["run"]
                total_num += 1
            
            for line in f_raw:
                item = json.loads(line)
                opt_state = item["type"]
                num_run_raw[opt_state] += item["run"]

            if model == "ylfeng_ReF-Decompile_messages_sem32":
                model = "ReF-Decompile"

            total_run_rate = sum(num_run.values()) / total_num
            total_run_rate_raw = sum(num_run_raw.values()) / total_num
            level_num = total_num // 4
            data.append(
                {
                    "model": model,
                    "O0 (RAW)": num_run_raw["O0"] / level_num,
                    "O1 (RAW)": num_run_raw["O1"] / level_num,
                    "O2 (RAW)": num_run_raw["O2"] / level_num,
                    "O3 (RAW)": num_run_raw["O3"] / level_num,
                    "AVG (RAW)": total_run_rate_raw,

                    "O0 (ReF)": num_run["O0"] / level_num,
                    "O1 (ReF)": num_run["O1"] / level_num,
                    "O2 (ReF)": num_run["O2"] / level_num,
                    "O3 (ReF)": num_run["O3"] / level_num,
                    "AVG (ReF)": total_run_rate,

                    "MIN Cycles #": min(num_cycles),
                    "MAX Cycles #": max(num_cycles),
                    "AVG Cycles #": mean(num_cycles),
                }
            )

    dataframe = pl.from_dicts(data)
    print(to_markdown(dataframe))
    gt_tbl = GT(dataframe)
    print(gt_tbl.as_latex())


if __name__ == "__main__":
    asyncio.run(main())
