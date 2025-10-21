<a href="index.html" class="back-home">Back to Home</a>

# The Ollama Odyssey: Lessons from Running High-Throughput LLM Serving in Production

**Author**: Emiliano Frigo<br>
**Role**: Data Engineer - Data Architecture in Operations<br>
**Company**: [Mutt Data](https://www.muttdata.ai/)

---

## Table of Contents

- [Introduction](#introduction)
- [Our Ollama Setup](#our-ollama-setup)
- [Understanding Ollama's Architecture](#understanding-ollamas-architecture)
  - [Server-Client Architecture](#server-client-architecture)
  - [Model Loading and Unloading: How It Should Work](#model-loading-and-unloading-how-it-should-work)
  - [GPU Memory Management](#gpu-memory-management)
  - [Multi-GPU Layer Distribution (Pipeline Parallelism)](#multi-gpu-layer-distribution-pipeline-parallelism)
  - [Conversation Processing Flow](#conversation-processing-flow)
  - [Why This Architecture Mattered for Our Production Use Case](#why-this-architecture-mattered-for-our-production-use-case)
- [The Problems We Encountered](#the-problems-we-encountered)
  - [1. Constant Hangs and Timeouts](#1-constant-hangs-and-timeouts)
  - [2. Opaque Logging and Debugging](#2-opaque-logging-and-debugging)
  - [3. The GPU Interconnection Bottleneck](#3-the-gpu-interconnection-bottleneck)
  - [4. Context Window Truncation and Instability](#4-context-window-truncation-and-instability)
  - [5. The Qwen Model Problem](#5-the-qwen-model-problem)
  - [6. Model Unloading Reliability](#6-model-unloading-reliability)
- [Our Band-Aid Solution: Subprocess Timeouts](#our-band-aid-solution-subprocess-timeouts)
- [Researching Alternatives: Self-Hosted Inference Options](#researching-alternatives-self-hosted-inference-options)
- [The Final Decision: Moving to APIs](#the-final-decision-moving-to-apis)
- [Lessons Learned](#lessons-learned)

---

## Introduction

At **Mutt Data**, we built an AI-powered system for one of our clients to generate legal declarations for immigration cases (VAWA and Visa T). The pipeline processes complex case data through multiple stages: PDF extraction, semantic chunking with FAISS vector search, multi-model LLM generation, and quality evaluation using BERTScore metrics.

**The Challenge**: Each pipeline run made 10-15 LLM calls across 5-8 different models (base generation, fallback models, multiple judges, guardrails rewriting). We needed infrastructure that could:
- Switch models dynamically mid-execution without restarting services
- Run autonomously on AWS EC2 for hours without manual intervention
- **Use uncensored models** to generate declarations covering sensitive abuse content (domestic violence, trafficking)
- **Maintain data privacy** for highly sensitive legal case information protected by attorney-client privilege

That last requirement was non-negotiable: we couldn't send client data to external APIs due to compliance requirements. Self-hosting wasn't a cost optimization—it was a **compliance necessity**.

**Why We Chose Ollama**:
- Simple single-binary installation with dynamic model loading/unloading
- Support for uncensored open-source models (Qwen3-Uncensored, Mistral-Venice)
- Clean Python API for switching between models during execution
- Promised to handle the complexity of GPU memory management

What followed was a **months-long battle** with constant hangs, unpredictable context window truncation, opaque debugging, and GPU memory issues that ultimately led us to abandon GPU-accelerated inference entirely.

## Our Ollama Setup

We deployed Ollama natively on **AWS EC2 g5.12xlarge** instances:
- 4x NVIDIA A10G GPUs (24GB VRAM each)
- 192 GB system RAM
- Ubuntu 22.04 with NVIDIA driver 535

**Key configuration** (systemd service):
```bash
Environment="OLLAMA_LOAD_TIMEOUT=600s"      # 10-minute model loading timeout
Environment="OLLAMA_MAX_LOADED_MODELS=2"    # Limit memory pressure
Environment="OLLAMA_KEEP_ALIVE=-1"          # Keep models in memory indefinitely
Environment="OLLAMA_FLASH_ATTENTION=1"      # Enable attention optimization
```

We also implemented GPU monitoring (CloudWatch metrics every 30 seconds) to track memory utilization and diagnose when models weren't unloading properly.

## Understanding Ollama's Architecture

Before diving into the problems we encountered, it's helpful to understand how Ollama works under the hood. This context will illuminate why certain issues emerged and why they were so difficult to resolve.

### Server-Client Architecture

![Ollama Basic Architecture](/assets/images/overall_arch_ollama.png)
*Figure 1: Ollama's Client-Server architecture showing the three core components and their HTTP-based communication.*

Ollama employs a classic **Client-Server (CS) architecture** with two main layers:

**The Client:**
- Interacts with users via the command line
- Can also be started through desktop application (Electron-based) or Docker
- All methods invoke the same executable file

**The Server (two core components):**

1. **Ollama HTTP Server (Go)**: A lightweight HTTP server built with the Gin framework that exposes REST API endpoints (`/api/generate`, `/api/chat`, etc.). This layer handles request routing, model lifecycle management, and response formatting.

2. **Inference Backend (llama.cpp)**: The actual LLM inference happens in llama.cpp, a high-performance C++ library optimized for running transformer models. Ollama communicates with llama.cpp through CGo bindings.

**How it works in practice:**
- You make an HTTP request to the Ollama server: `POST /api/generate`
- The server checks if the requested model is loaded in memory
- If not loaded, it loads the model into GPU/CPU memory
- The request is forwarded to the llama.cpp backend server
- Generated tokens stream back through the Go server to your client

Communication between the client and server, as well as between ollama-http-server and llama.cpp, all occurs via HTTP. It's worth noting that llama.cpp is an independent open-source project known for its **cross-platform and hardware-friendliness**—it can run without a GPU, even on devices like the Raspberry Pi.

This architecture makes Ollama incredibly easy to use—just a single binary and a REST API. But it also introduces complexity in managing state across these layers, especially when dealing with GPU memory.

### Model Loading and Unloading: How It Should Work

When you request a model in Ollama, here's what happens:

**Loading a model:**
1. Ollama checks if the model is already in memory
2. If not, it reads the model file from disk (usually `~/.ollama/models/`)
3. Model layers are loaded into available GPU VRAM (or CPU RAM if no GPU)
4. If the model exceeds a single GPU's capacity, Ollama automatically splits layers across multiple GPUs
5. The model stays "warm" in memory based on the `keep_alive` setting

**The `keep_alive` parameter:**
- `keep_alive=0`: Unload immediately after generation completes
- `keep_alive=5m`: Keep loaded for 5 minutes of inactivity
- `keep_alive=-1`: Keep loaded indefinitely (our configuration)

**Unloading a model:**
- When `keep_alive` expires or you explicitly request unloading
- Ollama should deallocate GPU memory and free VRAM
- The model can be loaded again when needed

In theory, this gives you dynamic model management with automatic memory cleanup. In practice, as we'll see in the problems section, the reliability of this mechanism can vary.

### GPU Memory Management

Ollama attempts to automatically manage GPU memory, but this automation comes with trade-offs:

**Memory allocation strategy:**
- When loading a model, Ollama calculates required VRAM based on model size and context window
- Larger context windows (`OLLAMA_NUM_CTX`) exponentially increase memory requirements
- With multiple models loaded (`OLLAMA_MAX_LOADED_MODELS=2`), memory becomes a juggling act
- Ollama doesn't expose real-time VRAM usage through its API, making it difficult to predict when you'll hit limits

The relationship between context window size and VRAM usage is non-linear, and memory fragmentation can accumulate over time, sometimes requiring service restarts.

### Multi-GPU Layer Distribution (Pipeline Parallelism)

When a model exceeds a single GPU's capacity, Ollama uses **pipeline parallelism** to split it across GPUs:

**How layer splitting works:**
```
Example: Qwen3-32B model with 32 transformer layers on 2 GPUs

GPU 0: Layers 0-15  (first half of the model)
GPU 1: Layers 16-31 (second half of the model)
```

**The process flow:**
1. Input tokens enter GPU 0
2. GPU 0 processes layers 0-15, produces intermediate activations
3. **Activations transfer from GPU 0 → GPU 1 over PCIe** (the bottleneck!)
4. GPU 1 processes layers 16-31, produces final output
5. Output transfers back to GPU 0 over PCIe

**Why this matters:**

The transfer speed between GPUs is critical for performance:
- **NVLink (A100, H100 instances)**: ~600 GB/s bandwidth → efficient multi-GPU inference
- **PCIe 4.0 (g5.12xlarge instances)**: ~32 GB/s bandwidth → **18x slower transfers!**

This bandwidth difference creates a significant bottleneck when models are split across GPUs with PCIe interconnections, as we'll see in the problems section.

### Conversation Processing Flow

![Ollama Conversation Flow](/assets/images/ollama_conversation_processing_flow.png)
*Figure 2: Detailed conversation processing flow showing the preparation stage (model download/verification) and interactive stage (actual chat/generation).*

The conversation process between a user and Ollama can be broken down into two main stages:

**1. Preparation Stage:**
- The user initiates a conversation by executing a CLI command like `ollama run llama3.2`
- The CLI client sends an HTTP request (`POST /api/show`) to ollama-http-server to retrieve model information from local storage
- If the model is not found locally (404 not found), the CLI sends a `POST /api/pull` request to download the model from the remote registry
- Once downloaded, model information is retrieved again and confirmed

**2. Interactive Conversation Stage:**
- The CLI sends an empty message to the `/api/generate` endpoint for initial setup
- The actual conversation begins with a `POST /api/chat` request to ollama-http-server
- The ollama-http-server relies on the llama.cpp engine to perform inference:
  - First, it sends a `GET /health` request to llama.cpp to confirm its health status
  - Then it sends a `POST /completion` request to receive the generated response
  - The response streams back through ollama-http-server to the CLI for display

### Why This Architecture Mattered for Our Production Use Case

In our production pipeline, we weren't using the CLI—we were making programmatic API calls to `/api/generate` from Python code running autonomously on EC2 instances. This meant:
- **No manual intervention** when models failed to load or hung during inference
- **Limited visibility** into which stage failed (HTTP server vs. llama.cpp backend)
- **Cascading failures** when model unloading didn't properly free GPU memory between requests

**Key takeaway:** Ollama's architecture prioritizes developer experience (simple installation, automatic GPU management, clean API) over production operations (observability, deterministic behavior, efficient multi-GPU). This trade-off works beautifully for local development and prototyping, but created significant challenges in our autonomous, multi-model production pipeline.

## The Problems We Encountered

### 1. Constant Hangs and Timeouts

**The Issue**: Ollama would frequently hang indefinitely during inference. The process wouldn't crash or return an error—it would simply stop responding.

**Symptoms:**
- LLM calls that normally took 30-60 seconds would never complete
- No error logs in Ollama's output (even with `OLLAMA_DEBUG=1`)
- GPU memory remained allocated but utilization dropped to 0%
- The only solution: kill the process and restart

**Impact**: Our event-driven pipeline ran autonomously on EC2. A single hang meant the entire pipeline would stall indefinitely, blocking all subsequent jobs.

### 2. Opaque Logging and Debugging

**What we saw:**
```
[GIN] 2025/09/15 - 14:23:45 | 200 | 45.234s | 127.0.0.1 | POST "/api/generate"
```

**What we needed:**
- Why did this request take 45 seconds?
- Was the model loading, or was inference slow?
- Did GPU memory get exhausted?

The lack of structured logging made it impossible to distinguish between normal operation, degraded performance, and complete hangs. Even verbose debug mode didn't illuminate root causes.

### 3. The GPU Interconnection Bottleneck

**The Hardware Reality**: The g5.12xlarge uses **PCIe interconnections** between GPUs (not NVLink). When model weights exceeded a single GPU's 24GB VRAM, Ollama split layers across GPUs.

**Performance impact:**
- **Single GPU (model fits in 24GB)**: ~30-45 seconds per generation
- **Multi-GPU split (32B model)**: ~120-180 seconds per generation (**3-4x slower!**)

**Why**: PCIe 4.0 bandwidth (~32 GB/s) vs. NVLink bandwidth (~600 GB/s). Cross-GPU tensor transfers became the bottleneck.

**Real-world consequence**: We couldn't use our best models (Qwen3-32B) because the 4x performance degradation was unacceptable.

### 4. Context Window Truncation and Instability

**The Issue**: Ollama defaults to a tiny **2048 token context window**, regardless of what the underlying model supports. For our RAG pipeline generating 3,000-5,000 word legal declarations, this was completely insufficient.

**Why This Mattered for RAG**: Our document extractions and declarations needed large contexts:
- 10-15 retrieved chunks from FAISS (500-1000 tokens each)
- Full case context (client history, abuse definitions, legal requirements)
- Multi-stage generation without losing context between steps

We had to manually configure `OLLAMA_NUM_CTX` to avoid content truncation during the first document extraction steps. We experimented with values like **35,567 tokens** and **47,234 tokens** to find the sweet spot between model quality and memory constraints.

**The Memory vs. Performance Trade-off**:
```bash
OLLAMA_NUM_CTX=2048    # Ollama default - completely unusable for our RAG use case
OLLAMA_NUM_CTX=35567   # Our first attempt - sometimes worked, sometimes OOM
OLLAMA_NUM_CTX=47234   # Higher quality - but unpredictable stability
OLLAMA_NUM_CTX=64000   # Requires ~20-30GB VRAM - frequent OOM errors
OLLAMA_NUM_CTX=128000  # Requires ~40-60GB VRAM - multi-GPU required (3-4x slower)
```

With 24GB per GPU, we were stuck between contexts too small for quality RAG and contexts that caused OOM errors or required multi-GPU splitting (which made inference 3-4x slower due to PCIe bottlenecks).

**Worst symptom**: Inconsistent behavior even with fixed `OLLAMA_NUM_CTX` values. The same prompt with the same context window setting would work once, fail the next time with an OOM error, then work again. A 15-step pipeline might succeed 10 times, then mysteriously fail on the 11th run with identical inputs and configuration.

### 5. The Qwen3 Model Problem

**The Issue**: Qwen3 models (the latest variants at the time) produced the highest-quality legal writing, and their uncensored variants were critical for our use case. Unfortunately, they were the **most unstable models** we ran on Ollama.

**Observed hang rates over 100 pipeline runs:**
```python
"llama3-13b":         ~5% hang rate
"mistral-24b-venice": ~8% hang rate
"qwen3-14b-instruct": ~25% hang rate  # Significantly worse
"qwen3-32b-instruct": ~40% hang rate  # Nearly unusable
```

**Symptoms with Qwen3 models:**
- Much longer "thinking pauses" before output (30-60s vs. 5-10s for Llama)
- Frequent mid-generation freezes, especially on complex legal reasoning
- Higher sensitivity to context window size

**Why This Was Devastating**: We were forced to choose between **quality** (Qwen3 models) and **reliability** (any other model that actually worked). This trade-off was unacceptable for a production system generating legal documents.

**What we tried**: Doubled timeouts, reduced context windows, disabled parallelism, disabled flash attention. Nothing reliably improved Qwen3 stability.

### 6. Model Unloading Reliability

**The Issue**: Ollama's model unloading mechanism (`keep_alive=0`) didn't always free GPU memory reliably.

**Behavior:**
- Sometimes unloaded immediately
- Other times, memory remained allocated for minutes
- Occasionally, memory was never freed until we restarted Ollama entirely

**Why This Mattered**: With `OLLAMA_MAX_LOADED_MODELS=2` and 24GB VRAM, we needed precise memory control. When unloading failed, we couldn't load the next model, causing pipeline failures.

## Our Band-Aid Solution: Subprocess Timeouts

To work around the hang issues, we built a **process-based timeout wrapper** using Python's `multiprocessing`:

```python
def with_timeout_and_retry(timeout_seconds: int, max_attempts: int, retry_delay: int):
    """Uses process isolation to guarantee timeout enforcement."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                process = multiprocessing.Process(
                    target=_worker_function,
                    args=(result_queue, func, args, kwargs)
                )
                process.start()
                process.join(timeout=timeout_seconds)

                if process.is_alive():  # Timeout occurred
                    process.terminate()
                    if process.is_alive():
                        process.kill()  # Force kill if terminate fails

                    if attempt < max_attempts:
                        time.sleep(retry_delay)
                        continue  # Retry
                    else:
                        raise TimeoutError(f"Timed out after {timeout_seconds}s")

                # Return result if successful
                return result_queue.get()[1]
        return wrapper
    return decorator
```

**Configuration**:
```bash
LLM_TIMEOUT=300       # 5-minute timeout per attempt
LLM_MAX_ATTEMPTS=3    # Retry up to 3 times
LLM_RETRY_DELAY=2     # 2-second delay between retries
```

**What it solved:**
- Prevented indefinite hangs by forcefully killing stuck processes
- Provided automatic retry for transient failures
- Allowed the pipeline to make progress instead of stalling forever

**What it didn't solve:**
- The underlying Ollama stability issues
- Context window truncation and unpredictable memory requirements
- GPU memory fragmentation over time
- The operational burden of managing retries and failures

This approach was a **band-aid, not a cure**. We were fighting the infrastructure instead of building features.

## Researching Alternatives: Self-Hosted Inference Options

After months of operational pain, we researched alternatives to Ollama. Several production-grade self-hosted solutions emerged:

### vLLM (UC Berkeley)

**Strengths:**
- **Continuous batching**: Process multiple requests concurrently
- **PagedAttention**: More efficient memory management (reduces fragmentation)
- **Production-grade**: Built by UC Berkeley's SkyLab team with extensive logging and Prometheus metrics
- **Better stability**: Community reports significantly better Qwen3 model stability

**The Critical Trade-off**: vLLM's limitation is that each instance serves **a single model**. To serve multiple models, you need multiple vLLM instances.

### Other Self-Hosted Alternatives We Considered

**Text Generation Inference (TGI) by Hugging Face:**
- **Strengths**: Supports dynamic batching, streaming, and quantization; excellent integration with Hugging Face ecosystem
- **Use case**: Best for teams already using Hugging Face models and infrastructure

**TensorRT-LLM (NVIDIA):**
- **Strengths**: Highly optimized for NVIDIA GPUs with TensorRT; excellent performance on A100/H100 instances
- **Use case**: Maximum performance for production workloads on NVIDIA hardware with expert GPU teams

**LocalAI:**
- **Strengths**: Multi-model support in a single instance; compatible with OpenAI API; supports text, audio, and image generation
- **Use case**: Teams needing multi-modal capabilities and easy migration from OpenAI API

**ExLlamaV2:**
- **Strengths**: Extremely fast inference for Llama-based models; excellent tensor parallelism support
- **Use case**: High-throughput serving of Llama model family with multi-GPU setups

While all these alternatives offered production-grade capabilities, **vLLM's single-model-per-instance limitation** was the common challenge for our multi-model pipeline architecture.

**Why this was a dealbreaker for us**:

Our pipeline required:
```python
# Stage 1: Extract sections
base_response = llm.generate(model="qwen-14b-chat", prompt=extraction_prompt)

# Stage 2: Generate declaration
declaration = llm.generate(model="qwen-32b-instruct", prompt=generation_prompt)

# Stage 3: Multi-judge evaluation (3 different models)
judge_scores = [
    llm.generate(model="qwen-14b-judge", prompt=eval_prompt),
    llm.generate(model="llama3-13b-judge", prompt=eval_prompt),
    llm.generate(model="mistral-24b-judge", prompt=eval_prompt),
]

# Stage 4: Guardrails rewriting
final = llm.generate(model="qwen-32b-guardrails", prompt=rewrite_prompt)
```

Running 5-6 separate vLLM servers simultaneously would require:
- **Massive memory overhead**: Each server loads its own model
- **Complex orchestration**: Managing multiple server lifecycles
- **Higher costs**: Multiple GPU instances or a much larger single instance

Given our compliance requirements (self-hosting mandatory) and architectural needs (multi-model switching), neither Ollama nor vLLM was a good fit.

## The Final Decision: Moving to APIs

After months of battling these issues, we made a difficult decision: **remove GPU support and Ollama entirely**.

**What We Removed:**
- All GPU/CUDA/NVIDIA infrastructure code (~400 lines)
- Ollama manager and utilities
- Complex retry/timeout logic
- GPU monitoring and metrics
- NVIDIA driver installation from bootstrap scripts

**What We Moved To:**
- **AWS Bedrock**: For models with strict privacy guarantees (data never leaves AWS network via PrivateLink)
- **OpenRouter**: Access to uncensored models that are **SOC2 compliant** and **do not store prompts for training** (e.g., `dolphin-mistral-24b-venice-edition:free`)
  - The Venice model is **completely free** on OpenRouter
  - Adding just **$10 credit unlocks 1,000 free model calls per day** (vs. 50 calls/day without credit)
  - This made experimentation and production use economically viable
- **CPU-only instances**: m7i.xlarge ($0.18/hour vs. g5.12xlarge at $5.67/hour)
- **Simplified architecture**: No GPU memory management, no model loading, no timeout wrappers

**Addressing Compliance Requirements:**

The combination of Bedrock and OpenRouter solved our key challenges:

1. **Data privacy**: AWS Bedrock with PrivateLink ensures data never leaves our AWS VPC
2. **Uncensored models**: OpenRouter provides access to uncensored models via API
3. **SOC2 compliance**: We specifically chose OpenRouter models certified as SOC2 compliant
4. **No training on our data**: Selected models that contractually guarantee prompts are not stored or used for training
5. **PII protection**: Additional layer of PII redaction before API calls, re-injection during post-processing

This setup wasn't perfect, but it was **dramatically more reliable** than fighting Ollama's instability while maintaining our compliance requirements.

**The Results:**

**Reliability:**
- Zero hangs or timeouts since migration
- Predictable latency (P95 under 15 seconds)
- No operational incidents related to LLM infrastructure

**Costs:**
- m7i.xlarge: $0.18/hour (~$130/month with auto-shutdown)
- g5.12xlarge: $5.67/hour (~$4,100/month if running 24/7)
- OpenRouter free models: $0/token (with $10 one-time credit for 1,000 calls/day)
- Even with occasional paid API calls, we're spending **dramatically less** overall when factoring in engineering time

**Development Velocity:**
- Removed 400+ lines of infrastructure code
- Eliminated GPU-specific testing environments
- Faster iteration on prompt engineering (no waiting for model downloads)

## Lessons Learned

### 1. Start with APIs, Optimize Later

We tried to optimize too early, and it cost us months of development time. **Stability and development velocity are more valuable than theoretical cost savings.** The migration from g5.12xlarge GPU instances ($4,100/month) to m7i.xlarge CPU instances ($130/month) plus OpenRouter's free tier proved that even with API costs, we're spending dramatically less when factoring in engineering time and zero operational overhead.

### 2. Ollama Is Great for Development, Not Production

**Ollama excels at:**
- Local experimentation and prototyping
- Developer-friendly model exploration
- Small-scale applications with manual monitoring

**Ollama struggles with:**
- Autonomous, event-driven production workloads
- Multi-model switching under high load
- Detailed observability and debugging
- Reliable model unloading and memory management
- Consistent context window handling (especially for large contexts in RAG)
- **Qwen3 model stability** (40% hang rate vs. 5% for Llama)

### 3. GPU Infrastructure Is a Specialization

Running GPU-accelerated inference in production requires expertise in:
- CUDA driver management
- Multi-GPU memory architecture (PCIe vs. NVLink)
- Kernel optimization and performance profiling

Unless this is your core competency, the operational burden outweighs the cost savings.

**Key insight**: Don't assume "4 GPUs = 4x the model size." PCIe bandwidth is the bottleneck for multi-GPU inference. We saw 3-4x performance degradation when splitting models across GPUs. NVLink architectures (A100, H100) are required for efficient multi-GPU.

### 4. Subprocess Timeouts Are a Code Smell

If you need to wrap LLM calls in subprocess-based timeout mechanisms, **your infrastructure has a reliability problem**. This pattern indicates you're fighting the tool instead of using it effectively.

### 5. The Future: Revisiting Self-Hosted Inference

We haven't abandoned self-hosting entirely. We plan to revisit self-hosted inference using production-grade solutions like **vLLM**, **TensorRT-LLM**, or **TGI** once we:

1. **Simplify our architecture**: Reduce models per pipeline run (5-8 models → 2-3)
2. **Build GPU expertise**: Invest in training on tensor parallelism and memory optimization
3. **Choose the right hardware**: Target NVLink-enabled instances (p4d, p5) instead of PCIe-limited g5
4. **Deploy strategically**: Use self-hosted inference for highest-volume model, keep APIs for specialized tasks

**What we'll do differently:**
- Accept the single-model-per-instance limitation (for vLLM/TGI) and plan architecture accordingly
- Start with one model, validate stability over weeks, then gradually expand
- Implement comprehensive monitoring from day one (Prometheus metrics, structured logs, alerting)
- Run parallel shadow deployments (self-hosted + API) to validate before cutover
- Set strict success criteria: if we can't achieve 99.5% reliability within 3 months, we stick with APIs

**Why we're optimistic about these alternatives:**
- Built for production (not development tools like Ollama)
- Better memory management should solve fragmentation and context window issues
- Community reports significantly better stability, especially with challenging models like Qwen3
- Built-in observability will help us debug issues before they become production incidents
- Mature ecosystems with extensive documentation and community support

We're particularly interested in exploring **LocalAI** for its multi-model capabilities and **TensorRT-LLM** for maximum performance on high-end NVIDIA hardware. The key is choosing the right tool for our evolved architecture rather than forcing our original design onto incompatible infrastructure.

### 6. Recommendations for Others

If you're building a system that requires **high reliability and minimal operational overhead**:

- **Start with API providers** and prove your product works first
  - For data privacy: AWS Bedrock with PrivateLink
  - For uncensored models: OpenRouter with SOC2-compliant options
  - For general use: Anthropic, OpenAI, Google Cloud
- **Avoid Ollama for production workloads** that run autonomously
- **Consider self-hosted inference (vLLM, TGI, TensorRT-LLM) only after** you have clear ROI calculations, GPU expertise, and willingness to invest in observability
- **Don't use multi-GPU splitting on PCIe-connected GPUs** (AWS g5 instances)

For teams with deep infrastructure expertise and high-volume, single-model workloads, production-grade self-hosted solutions (vLLM, TensorRT-LLM, TGI) offer compelling benefits. But don't rush into it. Start simple, validate demand, then optimize infrastructure when it's actually the bottleneck—not when you think it might be.

---

**About the Project**: This system was developed by **Mutt Data** for a client building AI-powered tools to assist immigration attorneys with case preparation. The RAG pipeline generates legal declarations for VAWA and Visa T cases, combining PDF extraction, semantic search, multi-model LLM generation, and quality evaluation.

This was a collaborative team effort involving 2 data architects and 3 ML engineers. As part of this team, I was specifically tasked with designing and managing the LLM inference infrastructure. The challenges documented in this article represent real production issues we encountered, debugged, and ultimately resolved. I'm sharing these lessons with the community in hopes that others can avoid similar pitfalls when building production LLM systems.

**About Mutt Data**: We are AI strategy partners specializing in building production-ready AI/ML systems for complex, real-world use cases. As an **AWS Advanced Consulting Partner** with Machine Learning competency, we help businesses in AdTech, MarTech, FinTech, and Telco build automated systems that maximize revenue. This project taught us invaluable lessons about the operational realities of self-hosted LLM infrastructure at scale. Learn more at [muttdata.ai](https://www.muttdata.ai/).

**Read More**:

**About Mutt Data:**
- [Mutt Data - AI Strategy Partners](https://www.muttdata.ai/)
- [Moving from Ollama to vLLM: Finding Stability for High-Throughput LLM Serving](https://pub.towardsai.net/moving-from-ollama-to-vllm-finding-stability-for-high-throughput-llm-serving-74d3dc9702c8)

**Ollama Resources:**
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Analysis of Ollama Architecture and Conversation Processing Flow](https://medium.com/@rifewang/analysis-of-ollama-architecture-and-conversation-processing-flow-for-ai-llm-tool-ead4b9f40975)

**Self-Hosted Inference Alternatives:**
- [vLLM Documentation](https://docs.vllm.ai/)
- [Text Generation Inference (TGI) by Hugging Face](https://github.com/huggingface/text-generation-inference)
- [TensorRT-LLM by NVIDIA](https://github.com/NVIDIA/TensorRT-LLM)
- [LocalAI Documentation](https://localai.io/)
- [ExLlamaV2 GitHub](https://github.com/turboderp/exllamav2)
