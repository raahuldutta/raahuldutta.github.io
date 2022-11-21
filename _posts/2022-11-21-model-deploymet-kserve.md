---
title: Model Deployment with KServe and KServe Features 
author: raahul
date: 2022-11-21 18:32:00 -0500
---

On a Friday Evening, Mid-September 2019 - Coworkers were leaving the office. The evening was welcoming a bright night. And I was working to deploy the LSTM models on production. I knew - “don't deploy anything on Friday”. But I rejected the guidance. I applied a small change on production Friday morning. Unfortunately, the model was not serving any requests due to poor infrastructure. At midnight - I was able to deploy the models perfectly. While returning home - I was sharing the bus with the returned party hoppers. You feel the pain right?

Now, It's 2022. The little hummingbird has fluttered her wings 7.5 billion times in the last three years. And We have now a simple pluggable solution for many burning machine learning problems like:
- Cost: Is the model over or under-scaled? Are resources being used efficiently?
- Monitoring: Are all the endpoints healthy? What is the performance profile and request trace?
- Rollouts: Is this rollout safe? How do I roll back? Can I test a change without swapping traffic?
- Protocol Standards: How do I make a prediction? GRPC? HTTP? Kafka
- How do I handle batch predictions?
- How do I leverage standardized Data Plane protocol so that I can move the model across MLServing Platforms?
- How do I serve Tensorflow, Xgboost, and Pytorch on the same infrastructure?
- How do I explain predictions?
- How do I wire up custom pre and post-processing?

And the Answer is KServe.
- KServe is a Model Inferencing Platform on Kubernetes.
- Run Anywhere Kubernetes runs. Provides Performant, Standardized inference protocol across ML Frameworks.
- Support modern serverless inference workload with Autoscaling including a scale to zero on GPU.
- Simple and Pluggable Production Serving including Prediction, pre/post-processing, monitoring, and explainable.
- Advanced deployments with Canary rollout, transformers, experiments, ensembles, and Model-Mesh.



![aws](/posts/202201119/decision_tree.jpeg){: width="616" height="557"}

If you are doing HPC (High Performance Job) like Drug Discovery or High Precision Job, then we suggest following the P (historically called Performance-Heavy) Instance Family. Else we recommend following the G (historically called Graphics-Heavy) Instance Family. I am providing the cost chart for the noted GPU instances.

## P3 and P4 Instances Cost

![aws](/posts/202201119/p3_p4.jpeg){: width="616" height="557"}

## G4 and G5 Instances Cost

![aws](/posts/202201119/g4_g5.jpeg){: width="616" height="557"}


![aws](/posts/202201119/dalle.png){: width="616" height="557"}


This image is not related to the GPU instance - I got the picture from Dall-E, it's not an actual view. The concept was generated with the diffusion model.

## Always Don't Select GPU on the Price Ground

Please don't select the GPU always as per the pricing basis. We executed a little experiment- We trained a Scibert Transformer model with 100K data points.
The result on a `g4dn.2xlarge` machine:

![aws](/posts/202201119/g4dn_price.png){: width="616" height="557"}

> The cost is : (0.752 * 311.25) / 3600  = $0.06
{: .prompt-info }

We executed on a `g5.xlarge` machine too. The result is on the same configuration :

![aws](/posts/202201119/g5_price.png){: width="616" height="557"}

> The cost is : (1.006 * 197.31) / 3600  = $0.05
{: .prompt-info }

### So if we use a g5.xlarge machine then we can save 20% of our budget.


And other benefits of a g5 family over a g4 family are: 

- `NVIDIA Ampere Architecture` is modern architecture, it supports all the precision formats.
- We should follow the `Mixed Precision Training` in Pytoch based project.There are different types of floating datatypes - `FP32`, `FP16`, `TF32`, `BF16`

![aws](/posts/202201119/vidia.png){: width="616" height="557"}

source: NVIDIA Blog

We executed the apple-to-apple comparison with `fp16` datatype because `tf32` and `bf16` need the `Ampere Architecture` which is available under `g5 instances` family.
 

## PS : What is TF32 and BF16?

### BF16

```
If you have access to a Ampere or newer hardware you can use bf16 for your training and evaluation. While bf16 has a worse precision than fp16, it has a much much bigger dynamic range. Therefore, if in the past you were experiencing overflow issues while training the model, bf16 will prevent this from happening most of the time. Remember that in fp16 the biggest number you can have is `65535` and any number above that will overflow. A bf16 number can be as large as `3.39e+38` (!) which is about the same as fp32 - because both have 8-bits used for the numerical range.
```
 

### TF32

```
The Ampere hardware uses a magical data type called tf32. It has the same numerical range as fp32 (8-bits), but instead of 23 bits precision it has only 10 bits (same as fp16) and uses only 19 bits in total.
It’s magical in the sense that you can use the normal fp32 training and/or inference code and by enabling tf32 support you can get up to 3x throughput improvement.
When this is done CUDA will automatically switch to using tf32 instead of fp32 where it’s possible. This, of course, assumes that the used GPU is from the Ampere series.
Like all cases with reduced precision this may or may not be satisfactory for your needs, so you have to experiment and see. According to NVIDIA research the majority of machine learning training shouldn’t be impacted and showed the same perplexity and convergence as the fp32 training.
```

## And I found the detailed AWS GPU Instance details for further read

| Architecture | NVIDIA GPU | Instance type | Instance name | Number of GPUs | GPU Memory (per GPU) | GPU Interconnect (NVLink / PCIe) | Thermal<br>Design Power (TDP) from nvidia-smi | Tensor Cores (mixed-precision) | Precision Support                  | CPU Type                           | Nitro based |
| ------------ | ---------- | ------------- | ------------- | -------------- | -------------------- | -------------------------------- | --------------------------------------------- | ------------------------------ | ---------------------------------- | ---------------------------------- | ----------- |
| Ampere       | A100       | P4            | p4d.24xlarge  | 8              | 40 GB                | NVLink gen 3 (600 GB/s)          | 400W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Ampere       | A10G       | G5            | g5.xlarge     | 1              | 24 GB                | NA (single GPU)                  | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.2xlarge    | 1              | 24 GB                | NA (single GPU)                  | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.4xlarge    | 1              | 24 GB                | NA (single GPU)                  | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.8xlarge    | 1              | 24 GB                | NA (single GPU)                  | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.16xlarge   | 1              | 24 GB                | NA (single GPU)                  | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.12xlarge   | 4              | 24 GB                | PCIe                             | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.24xlarge   | 4              | 24 GB                | PCIe                             | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Ampere       | A10G       | G5            | g5.48xlarge   | 8              | 24 GB                | PCIe                             | 300W                                          | Tensor Cores (Gen 3)           | FP64, FP32, FP16, INT8, BF16, TF32 | AMD EPYC                           | Yes         |
| Turing       | T4G        | G5            | g5g.xlarge    | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | AWS Graviton2                      | Yes         |
| Turing       | T4G        | G5            | g5g.2xlarge   | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | AWS Graviton2                      | Yes         |
| Turing       | T4G        | G5            | g5g.4xlarge   | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | AWS Graviton2                      | Yes         |
| Turing       | T4G        | G5            | g5g.8xlarge   | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | AWS Graviton2                      | Yes         |
| Turing       | T4G        | G5            | g5g.16xlarge  | 2              | 16 GB                | PCIe                             | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | AWS Graviton2                      | Yes         |
| Turing       | T4G        | G5            | g5g.metal     | 2              | 16 GB                | PCIe                             | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | AWS Graviton2                      | Yes         |
| Turing       | T4         | G4            | g4dn.xlarge   | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Turing       | T4         | G4            | g4dn.2xlarge  | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Turing       | T4         | G4            | g4dn.4xlarge  | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Turing       | T4         | G4            | g4dn.8xlarge  | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Turing       | T4         | G4            | g4dn.16xlarge | 1              | 16 GB                | NA (single GPU)                  | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Turing       | T4         | G4            | g4dn.12xlarge | 4              | 16 GB                | PCIe                             | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Turing       | T4         | G4            | g4dn.metal    | 8              | 16 GB                | PCIe                             | 70W                                           | Tensor Cores (Gen 2)           | FP32, FP16, INT8                   | Intel Xeon Scalable (Cascade Lake) | Yes         |
| Volta        | V100       | P3            | p3.2xlarge    | 1              | 16 GB                | NA (single GPU)                  | 300W                                          | Tensor Cores (Gen 1)           | FP64, FP32, FP16                   | Intel Xeon (Broadwell)             | No          |
| Volta        | V100       | P3            | p3.8xlarge    | 4              | 16 GB                | NVLink gen 2 (300 GB/s)          | 300W                                          | Tensor Cores (Gen 1)           | FP64, FP32, FP16                   | Intel Xeon (Broadwell)             | No          |
| Volta        | V100       | P3            | p3.16xlarge   | 8              | 16 GB                | NVLink gen 2 (300 GB/s)          | 300W                                          | Tensor Cores (Gen 1)           | FP64, FP32, FP16                   | Intel Xeon (Broadwell)             | No          |
| Volta        | V100\*     | P3            | p3dn.24xlarge | 8              | 32 GB                | NVLink gen 2 (300 GB/s)          | 300W                                          | Tensor Cores (Gen 1)           | FP64, FP32, FP16                   | Intel Xeon (Skylake)               | Yes         |
| Kepler       | K80        | P2            | p2.xlarge     | 1              | 12 GB                | NA (single GPU)                  | 149W                                          | No                             | FP64, FP32                         | Intel Xeon (Broadwell)             | No          |
| Kepler       | K80        | P2            | p2.8xlarge    | 8              | 12 GB                | PCIe                             | 149W                                          | No                             | FP64, FP32                         | Intel Xeon (Broadwell)             | No          |
| Kepler       | K80        | P2            | p2.16xlarge   | 16             | 12 GB                | PCIe                             | 149W                                          | No                             | FP64, FP32                         | Intel Xeon (Broadwell)             | No          |
| Maxwell      | M60        | G3            | g3s.xlarge    | 1              | 8 GB                 | PCIe                             | 150W                                          | No                             | FP32                               | Intel Xeon (Broadwell)             | No          |
| Maxwell      | M60        | G3            | g3.4xlarge    | 1              | 8 GB                 | PCIe                             | 150W                                          | No                             | FP32                               | Intel Xeon (Broadwell)             | No          |
| Maxwell      | M60        | G3            | g3.8xlarge    | 2              | 8 GB                 | PCIe                             | 150W                                          | No                             | FP32                               | Intel Xeon (Broadwell)             | No          |
| Maxwell      | M60        | G3            | g3.16xlarge   | 4              | 8 GB                 | PCIe                             | 150W                                          | No                             | FP32                               | Intel Xeon (Broadwell)             | No          |





## Reference

- [Choosing the right GPU for deep learning on AWS](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86)
- [AWS re:Invent 2021 - How to select Amazon EC2 GPU instances for deep learning (sponsored by NVIDIA)](https://youtu.be/4bVrIbgGWEA)


[chirpy-homepage]: https://github.com/cotes2020/jekyll-theme-chirpy/
