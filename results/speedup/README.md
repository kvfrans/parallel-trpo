# parallel-trpo

A parallel implementation of TRPO.

I'm working towards the ideas at [this openAI research request](https://openai.com/requests-for-research/#parallel-trpo). Code is working, and I'm in the middle of collecting data and writing a paper.

Currently has about a 2x speedup with 4 multiprocesses running on a 4 core macbook pro