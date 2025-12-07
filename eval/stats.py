import json
import asyncio
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns


async def main():
    num_cycles = []
    time_elapsed = []
    start_times = []
    end_times = []

    len2time = []

    with open("results/ylfeng_ReF-Decompile_messages_sem32.jsonl", "r") as f:
        data_all = [json.loads(line) for line in f]

        for item in data_all:
            num_cycle = len([m for m in item["messages"] if m["role"] == "assistant"])
            num_cycles.append(num_cycle)

            elapsed = (
                item["messages"][-1]["end_time"] - item["messages"][-1]["start_time"]
            )
            time_elapsed.append(elapsed)

            len2time.append((len(item["messages"][-1]["content"]), elapsed))

            start_times.append(item["messages"][-1]["start_time"])
            end_times.append(item["messages"][-1]["end_time"])

    print(mean(num_cycles), min(num_cycles), max(num_cycles))
    print(mean(time_elapsed), min(time_elapsed), max(time_elapsed))
    print(max(end_times) - min(start_times))

    len2time = sorted(len2time, key=lambda x: x[0])

    xs = []
    ys = []
    for x, y in len2time:
        xs.append(x)
        ys.append(y)

    ax = sns.scatterplot(x=xs, y=ys)

    plt.savefig("stats.pdf")


if __name__ == "__main__":
    asyncio.run(main())
