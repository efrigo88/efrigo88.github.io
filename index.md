---
layout: default
title: Home
---

# Welcome

<div style="display: flex; align-items: flex-start; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;">
  <div style="flex: 0 0 auto;">
    <img src="/assets/images/me.png" alt="Emiliano Frigo" style="max-width: 300px; width: 100%; border-radius: 8px;">
  </div>
  <div style="flex: 1 1 300px;">
    <p>Hi, I'm <strong>Emiliano Frigo</strong>, a Data Engineer specializing in Data Architecture in Operations, with hands-on experience in AI Engineering.</p>
    <p>My primary focus is on <strong>data architecture</strong>, designing scalable systems and pipelines, and more recently on <strong>AI engineering</strong>: building and operating LLM-based systems in production. This blog is where I explore new territories—starting with my first venture into the world of AI/ML infrastructure when I got assigned to an AI project.</p>
  </div>
</div>

---

## Latest Posts

### [The Ollama Odyssey: Lessons from Running High-Throughput LLM Serving in Production](ollama-infrastructure-challenges.html)

*Published: January 2025* • **My first AI/ML blog post**

This was my first deep dive into AI/ML infrastructure after being assigned to an AI project. Coming from a data architecture background, I documented the challenges we faced running Ollama for self-hosted LLM inference in a production RAG pipeline. This post covers:

- 6 major stability issues (constant hangs, context window problems, Qwen model instability)
- GPU infrastructure challenges (PCIe bottlenecks on g5.12xlarge)
- Our subprocess-based timeout solution (the "band-aid")
- Why we migrated to AWS Bedrock + OpenRouter
- Cost analysis and lessons learned
- Future plans with vLLM

**Key takeaway**: Start with APIs, optimize later. Stability and development velocity are more valuable than theoretical cost savings.

[Read the full post →](ollama-infrastructure-challenges.html)

---

## About Me

- **Role**: Data Engineer - Data Architecture in Operations & AI Engineering
- **Primary Focus**: Data architecture, designing scalable data systems and pipelines
- **AI Engineering**: LLM inference infrastructure, RAG pipelines, and production AI/ML systems
- **Recent Exploration**: AI/ML infrastructure (this blog documents my first AI project assignment)
- **GitHub**: [@efrigo88](https://github.com/efrigo88)

---

## Connect

- [GitHub](https://github.com/efrigo88)
- [LinkedIn](https://www.linkedin.com/in/emiliano-frigo-17222733)
