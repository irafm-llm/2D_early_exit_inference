#### [Paper (PDF)](docs/2D%20early%20exit.pdf)

The project introduces a two-dimensional (2D) early exit strategy that coordinates layer-wise and sentence-wise exiting for classification tasks in large language models. By processing input
incrementally sentence-by-sentence while progressively activating deeper layers, our method achieves multiplicative computational savings that exceed those from optimizing either dimension
independently. Experimental evaluation across four state-of-the-art LLMs (Llama 3.1, Llama 3.2, Gemma, Qwen; 3B-8B parameters) on three sentiment classification datasets demonstrates additional speed-ups of 1.4–2.3× over optimal layer-wise early exit for simpler tasks with vanilla models, with graceful degradation on complex multi-class problems. The approach is model-agnostic, requires only lightweight classification adapters, and is orthogonal to complementary efficiency methods such as quantization and pruning. 
