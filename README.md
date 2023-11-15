# LLM_Challenge

# Project Repository
These repository consists 3 submissions for neurips23 llm efficiency challenge  in Nvidia 4090 track

This repository contains the project files organized as follows:

<details>
  <summary>4090_track</summary>
  
  - [submission_1](./4090_track/submission_1)
    - [training](./4090_track/submission_1/training)
    - [evaluation](./4090_track/submission_1/evaluation)

  - [submission_2](./4090_track/submission_2)
    - [training](./4090_track/submission_2/training)
    - [evaluation](./4090_track/submission_2/evaluation)

  - [submission_3](./4090_track/submission_3)
    - [training](./4090_track/submission_3/training)
    - [evaluation](./4090_track/submission_3/evaluation)
</details>

# LLM_Challenge

# Project Repository

These repository consists 3 submissions for neurips23 llm efficiency challenge in Nvidia 4090 track

This repository contains the project files organized as follows:

<details>
  <summary>4090_track</summary>
  
  - [submission_1](./4090_track/submission_1)
    - [training](./4090_track/submission_1/training)
    - [evaluation](./4090_track/submission_1/evaluation)

- [submission_2](./4090_track/submission_2)

  - [training](./4090_track/submission_2/training)
  - [evaluation](./4090_track/submission_2/evaluation)

- [submission_3](./4090_track/submission_3) - [training](./4090_track/submission_3/training) - [evaluation](./4090_track/submission_3/evaluation)
</details>

# Training using docker container

### Build docker container

```bash
docker build -f Dockerfile -t llm_challenge:train .
```

### Run main.py in docker container

```bash
docker run --gpus all -ti llm_challenge:train python3 main.py
```

