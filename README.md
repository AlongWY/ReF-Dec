# ReF Decompile: Relabeling and Function Call Enhanced Decompile

Code for [ReF Decompile: Relabeling and Function Call Enhanced Decompile](https://arxiv.org/abs/2502.12221)

## Deploy

```base
python merge.py --output-dir ReF-Decompile
vllm serve ReF-Decompile --port 8000 --enable-auto-tool-choice --tool-call-parser mistral
python eval.py --base_url http://127.0.0.1:8000/v1
```

## Results

<table class="table">
    <tr class="header">
        <th rowspan="2">Model/Metrics</th>
        <th colspan="5">Re-executability Rate (%)</th>
        <th colspan="5">Readability (#)</th>
    </tr>
    <tr class="header">
        <td>O0</td><td>O1</td><td>O2</td><td>O3</td><td>AVG</td>
        <td>O0</td><td>O1</td><td>O2</td><td>O3</td><td>AVG</td>
    </tr>
    <tr style="text-align: center;"><td colspan="11"><strong>Rule Based Decompiler</strong></td></tr>
    <tr>
        <td>ghidra</td>
        <td>34.76</td><td>16.46</td><td>15.24</td><td>14.02</td><td>20.12</td>
        <td>2.98</td><td>2.41</td><td>2.52</td><td>2.38</td><td>2.57</td>
    </tr>
    <tr style="text-align: center;"><td colspan="11"><strong>Refine-Based Method</strong></td></tr>
    <tr>
        <td>GPT-4o</td>
        <td>46.95</td><td>34.15</td><td>28.66</td><td>31.10</td><td>35.22</td>
        <td>2.82</td><td>2.35</td><td>2.29</td><td>2.31</td><td>2.44</td>
    </tr>
    <tr>
        <td><a href="https://github.com/albertan017/LLM4Decompile">LLM4Decompile-Ref</a></td>
        <td>74.39</td><td>46.95</td><td>47.56</td><td>42.07</td><td>52.74</td>
        <td>4.08</td><td>3.38</td><td>3.34</td><td>3.19</td><td>3.50</td>
    </tr>
    <tr style="text-align: center;"><td colspan="11"><strong>End-to-End Method</strong></td></tr>
    <tr>
        <td><a href="https://github.com/albertan017/LLM4Decompile">LLM4Decompile-End</a></td>
        <td>69.51</td><td>44.51</td><td>39.63</td><td>38.41</td><td>48.02</td>
        <td>4.07</td><td>3.46</td><td>3.40</td><td>3.23</td><td>3.54</td>
    </tr>
    <tr>
        <td><a href="https://github.com/AlongWY/sccdec">FAE Decompile</a></td>
        <td>67.68</td><td>48.78</td><td>45.73</td><td>42.07</td><td>51.07</td>
        <td>3.94</td><td>3.46</td><td>3.40</td><td>3.25</td><td>3.51</td>
    </tr>
    <tr>
        <td><a href="https://github.com/AlongWY/sccdec">FAE Decompile+SCC</a></td>
        <td>70.24</td><td>48.54</td><td>47.56</td><td>43.29</td><td>52.41</td>
        <td>3.97</td><td>3.48</td><td>3.41</td><td>3.23</td><td>3.52</td>
    </tr>
    <tr>
        <td><a href="https://github.com/AlongWY/ReF-Dec">ReF Decompile</a></td>
        <td>85.37</td><td>56.10</td><td>51.83</td><td>52.43</td><td>61.43</td>
        <td>4.13</td><td>3.60</td><td>3.54</td><td>3.49</td><td>3.69</td>
    </tr>
</table>


## Resources

+ [Code](https://github.com/AlongWY/ReF-Dec)
+ [Model](https://huggingface.co/ylfeng/ReF-Decompile-lora)
+ [Paper](https://arxiv.org/abs/2502.12221)

## Reference

```bibtex
@misc{feng2025refdecompilerelabelingfunction,
      title={ReF Decompile: Relabeling and Function Call Enhanced Decompile}, 
      author={Yunlong Feng and Bohan Li and Xiaoming Shi and Qingfu Zhu and Wanxiang Che},
      year={2025},
      eprint={2502.12221},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2502.12221}, 
}
```
