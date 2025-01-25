# Quick analysis of VLMs to do math in images

One needs Ollama and python 3.10 to run the notebooks

Download ollama and run ```ollama serve```.

Then install oython libraries:
```
pip install ollama
pip install notebook
```


Only open source modes provided by Ollama are tesed for now. 

Results are not great.

Timing for comparison with NNs are included.

The scripts were run on hardware:

```
Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 
32 GB RAM
Ubuntu 20.04.6 LTS
64-bit
```


The models run are:
```
llava-llama3 (8b)
llava        (7b)
moondream    (1.8b)
bakllava     (7b)
```

The images tested are created in a computer mimicking handwritten digits:

![Operation 1](op1.png?raw=true) ![Operation 2](op2.png?raw=true) ![Operation 3](op3.png?raw=true)

