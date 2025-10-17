<a href="index.html" class="back-home">Back to Home</a>

# The Ollama Odyssey: Lessons from Running High-Throughput LLM Serving in Production

**Author**: Emiliano Frigo<br>
**Role**: Data Engineer - Data Architecture in Operations<br>
**Company**: [Mutt Data](https://www.muttdata.ai/)

---

## Table of Contents

- [Introduction](#introduction)
- [Our Ollama Setup](#our-ollama-setup)
- [The Problems We Encountered](#the-problems-we-encountered)
  - [1. Constant Hangs and Timeouts](#1-constant-hangs-and-timeouts)
  - [2. Opaque Logging and Debugging](#2-opaque-logging-and-debugging)
  - [3. The GPU Interconnection Bottleneck](#3-the-gpu-interconnection-bottleneck)
  - [4. Context Window Truncation and Instability](#4-context-window-truncation-and-instability)
  - [5. The Qwen Model Problem](#5-the-qwen-model-problem)
  - [6. Model Unloading Reliability](#6-model-unloading-reliability)
- [Our Band-Aid Solution: Subprocess Timeouts](#our-band-aid-solution-subprocess-timeouts)
- [Researching Alternatives: The vLLM Trade-off](#researching-alternatives-the-vllm-trade-off)
- [The Final Decision: Moving to APIs](#the-final-decision-moving-to-apis)
- [Lessons Learned](#lessons-learned)

---

## Introduction

At **Mutt Data**, we built an AI-powered system for our client **TramCase** to generate legal declarations for immigration cases (VAWA and Visa T). The pipeline processes complex case data through multiple stages: PDF extraction, semantic chunking with FAISS vector search, multi-model LLM generation, and quality evaluation using BERTScore metrics.

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

## Researching Alternatives: The vLLM Trade-off

After months of operational pain, we researched alternatives. **vLLM** emerged as the production-grade option with significant advantages:

**vLLM Benefits:**
- **Continuous batching**: Process multiple requests concurrently
- **PagedAttention**: More efficient memory management (reduces fragmentation)
- **Production-grade**: Built by UC Berkeley's SkyLab team with extensive logging and Prometheus metrics
- **Better stability**: Community reports significantly better Qwen3 model stability

**The Critical Trade-off**: vLLM's limitation is that each instance serves **a single model**. To serve multiple models, you need multiple vLLM instances.

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

### 5. The Future: vLLM Done Right

We haven't abandoned self-hosting entirely. We plan to revisit self-hosted inference using **vLLM** once we:

1. **Simplify our architecture**: Reduce models per pipeline run (5-8 models → 2-3)
2. **Build GPU expertise**: Invest in training on tensor parallelism and memory optimization
3. **Choose the right hardware**: Target NVLink-enabled instances (p4d, p5) instead of PCIe-limited g5
4. **Deploy strategically**: Use vLLM for highest-volume model, keep APIs for specialized tasks

**What we'll do differently with vLLM:**
- Accept the single-model-per-instance limitation and plan architecture accordingly
- Start with one model, validate stability over weeks, then gradually expand
- Implement comprehensive monitoring from day one (Prometheus metrics, structured logs, alerting)
- Run parallel shadow deployments (vLLM + API) to validate before cutover
- Set strict success criteria: if we can't achieve 99.5% reliability within 3 months, we stick with APIs

**Why we're optimistic about vLLM:**
- Built for production (not a development tool like Ollama)
- PagedAttention should solve memory fragmentation and context window issues
- Community reports significantly better stability, especially with Qwen3 models
- Built-in observability will help us debug issues before they become production incidents

### 6. Recommendations for Others

If you're building a system that requires **high reliability and minimal operational overhead**:

- **Start with API providers** and prove your product works first
  - For data privacy: AWS Bedrock with PrivateLink
  - For uncensored models: OpenRouter with SOC2-compliant options
  - For general use: Anthropic, OpenAI, Google Cloud
- **Avoid Ollama for production workloads** that run autonomously
- **Consider vLLM only after** you have clear ROI calculations, GPU expertise, and willingness to invest in observability
- **Don't use multi-GPU splitting on PCIe-connected GPUs** (AWS g5 instances)

For teams with deep infrastructure expertise and high-volume, single-model workloads, vLLM offers compelling benefits. But don't rush into it. Start simple, validate demand, then optimize infrastructure when it's actually the bottleneck—not when you think it might be.

---

**About the Project**: This system was developed by **Mutt Data** for **TramCase**, a company building AI-powered tools to assist immigration attorneys with case preparation. The RAG pipeline generates legal declarations for VAWA and Visa T cases, combining PDF extraction, semantic search, multi-model LLM generation, and quality evaluation.

**About Mutt Data**: We are AI strategy partners specializing in building production-ready AI/ML systems for complex, real-world use cases. As an **AWS Advanced Consulting Partner** with Machine Learning competency, we help businesses in AdTech, MarTech, FinTech, and Telco build automated systems that maximize revenue. This project taught us invaluable lessons about the operational realities of self-hosted LLM infrastructure at scale. Learn more at [muttdata.ai](https://www.muttdata.ai/).

**Read More**:
- [Mutt Data - AI Strategy Partners](https://www.muttdata.ai/)
- [Moving from Ollama to vLLM: Finding Stability for High-Throughput LLM Serving](https://pub.towardsai.net/moving-from-ollama-to-vllm-finding-stability-for-high-throughput-llm-serving-74d3dc9702c8)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ollama GitHub](https://github.com/ollama/ollama)
