"""Generate Aurelius dissertation PDF — updated May 2026."""
from __future__ import annotations

import argparse
import datetime

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY  = HexColor("#0D1F3C")
GOLD  = HexColor("#C9A84C")
STEEL = HexColor("#2C4A7C")
LIGHT = HexColor("#EFF3FA")
WHITE = colors.white
BLACK = colors.black
GRAY  = HexColor("#555555")
LGRAY = HexColor("#AAAAAA")
PAGE_W, PAGE_H = letter

# Repository root path (fallback: current working directory)
import os
from pathlib import Path

REPO_ROOT = os.getenv("AURELIUS_ROOT", str(Path.cwd()))


# ── Styles ───────────────────────────────────────────────────────────────────
def S():
    return {
        "cover_title": ParagraphStyle("cover_title", fontName="Helvetica-Bold",
            fontSize=32, textColor=WHITE, alignment=TA_CENTER, leading=40),
        "cover_sub": ParagraphStyle("cover_sub", fontName="Helvetica",
            fontSize=15, textColor=GOLD, alignment=TA_CENTER, leading=22),
        "cover_meta": ParagraphStyle("cover_meta", fontName="Helvetica",
            fontSize=10, textColor=LGRAY, alignment=TA_CENTER, leading=16),
        "chapter": ParagraphStyle("chapter", fontName="Helvetica-Bold",
            fontSize=19, textColor=NAVY, spaceBefore=24, spaceAfter=8, leading=24),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold",
            fontSize=13, textColor=STEEL, spaceBefore=14, spaceAfter=5, leading=17),
        "subsection": ParagraphStyle("subsection", fontName="Helvetica-Bold",
            fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=4, leading=14),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
            textColor=BLACK, leading=15, spaceAfter=6, alignment=TA_JUSTIFY),
        "body_sm": ParagraphStyle("body_sm", fontName="Helvetica", fontSize=9,
            textColor=GRAY, leading=13, spaceAfter=4),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8,
            textColor=HexColor("#1A1A2E"), backColor=HexColor("#F4F6FA"),
            leading=12, leftIndent=12, rightIndent=12, spaceBefore=4, spaceAfter=4),
        "eq": ParagraphStyle("eq", fontName="Courier-Bold", fontSize=9,
            textColor=NAVY, backColor=HexColor("#F0F4FF"),
            leading=14, leftIndent=20, spaceBefore=4, spaceAfter=4),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
            fontSize=9, textColor=GRAY, alignment=TA_CENTER, spaceAfter=8),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def ch(num, title, st):
    return [
        Spacer(1, 0.12*inch),
        HRFlowable(width="100%", thickness=3, color=NAVY, spaceAfter=5),
        Paragraph(f"<font color='#C9A84C'>{num}</font>  {title}", st["chapter"]),
        HRFlowable(width="100%", thickness=1, color=STEEL, spaceBefore=2, spaceAfter=10),
    ]


def tbl(data, widths, hbg=NAVY):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), hbg),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 8.5),
        ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,1), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.4, HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",  (0,0), (-1,-1), "LEFT"),
        ("WORDWRAP",(0,0), (-1,-1), True),
    ]))
    return t


def eq(label, formula, note, st):
    return tbl(
        [[Paragraph(f"<b>{label}</b>", st["body_sm"]),
          Paragraph(formula, st["eq"]),
          Paragraph(f"<i>{note}</i>", st["body_sm"])]],
        [0.55*inch, 4.0*inch, 2.2*inch], hbg=STEEL
    )


def bar_chart_drawing(values, labels, title, bar_color=None):
    try:
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.shapes import Drawing, String
        bc_color = bar_color or NAVY
        w, h = 460, 180
        d = Drawing(w, h + 50)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.width, bc.height = 55, 35, w - 80, h - 15
        bc.data = [values]
        bc.categoryAxis.categoryNames = labels
        bc.categoryAxis.labels.fontName = "Helvetica"
        bc.categoryAxis.labels.fontSize = 7
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.labels.dy = -8
        bc.valueAxis.labels.fontName = "Helvetica"
        bc.valueAxis.labels.fontSize = 7
        bc.bars[0].fillColor = bc_color
        d.add(bc)
        d.add(String(w/2, h + 38, title, textAnchor="middle",
                     fontSize=9, fontName="Helvetica-Bold", fillColor=NAVY))
        return d
    except Exception:
        return None


def on_page(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 0.42*inch, w, 0.42*inch, fill=True, stroke=False)
        canvas.setFillColor(GOLD)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(0.5*inch, h - 0.26*inch, "AURELIUS AI — TECHNICAL DISSERTATION")
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 0.5*inch, h - 0.26*inch, f"Page {doc.page}")
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, w, 0.33*inch, fill=True, stroke=False)
        canvas.setFillColor(LGRAY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawCentredString(w/2, 0.12*inch, "CONFIDENTIAL — FOR ACADEMIC & EXECUTIVE REVIEW")
    canvas.restoreState()


# ── Build ─────────────────────────────────────────────────────────────────────
def build(output_path: str | Path) -> str:
    path = str(Path(output_path).expanduser().resolve())
    doc = SimpleDocTemplate(path, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.72*inch, bottomMargin=0.58*inch,
        title="Aurelius AI: Technical Dissertation",
        author="Christien Antonio — Aurelius Research",
        subject="Memory-Augmented Transformer with Autonomous Agent Capabilities")

    st = S()
    story = []

    # ── COVER ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.0*inch))
    cover_tbl = Table([[Paragraph("AURELIUS AI", st["cover_title"])]],
                      colWidths=[6.8*inch])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), NAVY),
        ("TOPPADDING", (0,0),(-1,-1), 30),
        ("BOTTOMPADDING",(0,0),(-1,-1), 22),
        ("LEFTPADDING", (0,0),(-1,-1), 20),
        ("RIGHTPADDING",(0,0),(-1,-1), 20),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 0.15*inch))

    sub_tbl = Table([[Paragraph(
        "A Memory-Augmented Transformer with Autonomous Agent Capabilities<br/>"
        "Technical Dissertation — Updated May 2026",
        st["cover_sub"])]], colWidths=[6.8*inch])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), STEEL),
        ("TOPPADDING",(0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING",(0,0),(-1,-1), 16),
        ("RIGHTPADDING",(0,0),(-1,-1), 16),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 0.45*inch))

    meta = [
        ["Principal Investigator", "Christien Antonio"],
        ["Affiliation", "Aurelius Research — Independent Laboratory"],
        ["Model Family", "Aurelius-150M · 1B · 2.7B (in dev) · 3B · 7B"],
        ["Primary Training Target", "~1.395B Parameter Transformer"],
        ["Architecture", "Aurelian Memory Core (AMC) + ReAct Agent Loop"],
        ["Codebase Size", "2,155 Python files · ~499,000 lines of code"],
        ["Alignment Algorithms", "80+ (CAI, PPO, GRPO, DAPO, RLVR, DPO, KTO, ORPO …)"],
        ["Inference Modules", "200+ (speculative, RAG, KIVI, Quest, DuoAttn, TEAL …)"],
        ["Security Hardening", "AUR-SEC-2026-0001 through 0027 — all closed"],
        ["Document Date", datetime.date.today().strftime("%B %d, %Y")],
        ["License", "MIT"],
    ]
    meta_tbl = Table(meta, colWidths=[2.3*inch, 4.5*inch])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,-1), HexColor("#EEF2FA")),
        ("FONTNAME", (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0),(1,-1), "Helvetica"),
        ("FONTSIZE", (0,0),(-1,-1), 9.5),
        ("TEXTCOLOR",(0,0),(0,-1), NAVY),
        ("GRID", (0,0),(-1,-1), 0.4, HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.4*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=GOLD))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph(
        "Prepared for Academic Evaluation · C-Suite Technology Briefing · Investor Due Diligence",
        st["cover_meta"]))
    story.append(PageBreak())

    # ── §0 EXECUTIVE SUMMARY ──────────────────────────────────────────────────
    story += ch("§ 0", "Executive Summary", st)

    story.append(Paragraph(
        "Aurelius is an original research project implementing a family of decoder-only transformer models "
        "augmented with a novel three-tier differentiable memory architecture — the <b>Aurelian Memory Core "
        "(AMC)</b> — and a full autonomous agent loop. The system was designed and built from scratch "
        "by a solo researcher at an estimated cloud cost of approximately $350. As of May 2026 the codebase "
        "spans <b>2,155 Python source files</b> and approximately <b>499,000 lines of code</b>, making it "
        "one of the most comprehensive independently built LLM platforms in existence.",
        st["body"]))

    story.append(Paragraph(
        "The platform now includes over <b>80 alignment algorithms</b> (from Constitutional AI and PPO through "
        "DPO, KTO, ORPO, SimPO, SPIN, and REBEL), over <b>200 inference modules</b> (speculative decoding, "
        "RAG variants, KIVI 2-bit KV quantization, Quest page-sparse attention, DuoAttention, TEAL activation "
        "sparsity, and more), a complete safety layer with 27 security findings closed under "
        "AUR-SEC-2026-0001 through 0027, and a five-layer neural brain cognitive architecture.",
        st["body"]))

    story.append(Paragraph(
        "A <b>2.7B parameter variant</b> is currently in active development on the branch "
        "<code>feat/1-scale-2_7b-config-stack</code> (562 files), and a branch audit identified 5 "
        "merge-ready feature branches representing significant capability additions.",
        st["body"]))

    story.append(tbl([
        ["Metric", "Value"],
        ["Primary model", "~1.395B parameters (1B backbone + AMC)"],
        ["Model family", "150M · 1B · 2.7B (dev) · 3B · 7B"],
        ["Vocabulary", "50,257 tokens (GPT-2 tokenizer)"],
        ["Context length (1B)", "4,096 tokens; extended by LTS retrieval"],
        ["Alignment algorithms", "80+ implemented"],
        ["Inference modules", "200+ implemented"],
        ["Memory tiers per layer", "3 — Working / Episodic / Long-Term Store"],
        ["Security findings", "27 closed (AUR-SEC-2026-0001–0027)"],
        ["Active remote branches", "42 total · 5 merge-ready"],
        ["Codebase", "2,155 Python files · ~499K lines"],
        ["Build duration", "~3 weeks (solo researcher)"],
        ["Estimated training budget", "~$350 cloud compute"],
        ["License", "MIT"],
    ], [2.8*inch, 4.0*inch]))
    story.append(Paragraph("Table 0.1 — Aurelius Key Performance Indicators (May 2026)", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 1 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 1", "Introduction and Motivation", st)

    story.append(Paragraph("1.1  The Problem with Standard Transformers", st["section"]))
    story.append(Paragraph(
        "The transformer architecture (Vaswani et al., 2017) revolutionized NLP but carries three "
        "fundamental limitations: a fixed-size context window that discards earlier tokens, no "
        "mechanism to retain knowledge across inference calls, and no built-in ability to plan "
        "multi-step actions or learn from experience at inference time. Aurelius was designed to "
        "address all three through architectural extension rather than prompt engineering.",
        st["body"]))

    story.append(Paragraph("1.2  Research Objectives", st["section"]))
    for i, obj in enumerate([
        "Design a differentiable per-layer three-tier memory hierarchy (Working → Episodic → Long-Term).",
        "Integrate a complete Observe→Think→Act→Reflect agent loop as an architectural component.",
        "Implement the full modern RLHF spectrum: Constitutional AI, PPO, GRPO, DAPO, RLVR, DPO, and beyond.",
        "Build and train the entire system at solo-researcher budget (~$350 cloud).",
        "Establish a security-first posture with formal hardening, CI/CD gates, and complete audit trails.",
    ], 1):
        story.append(Paragraph(f"<b>{i}.</b>  {obj}", st["body"]))

    story.append(Paragraph("1.3  Scope and Scale", st["section"]))
    story.append(Paragraph(
        "Aurelius is not a fine-tuned wrapper around an existing open-source model. Every component — "
        "transformer backbone, AMC memory layers, agent loop, skill library, alignment trainers, "
        "inference optimizations, safety systems, serving infrastructure, and CI/CD automation — was "
        "implemented from first principles. The codebase now contains 2,155 Python files organized "
        "into 40+ subsystem packages, backed by Rust extensions for performance-critical memory "
        "management and TypeScript/React for the frontend.",
        st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 2 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 2", "Mathematical Foundations", st)

    story.append(Paragraph(
        "This chapter presents the core mathematical formulations underlying Aurelius. "
        "All equations reflect the project's actual implementation conventions.",
        st["body"]))

    story.append(Paragraph("2.1  Attention Mechanisms", st["section"]))
    for label, formula, note in [
        ("Eq 2.1", "Attention(Q,K,V) = softmax( QK^T / sqrt(d_k) ) * V",
         "Scaled dot-product attention"),
        ("Eq 2.2", "MHA(Q,K,V) = Concat(head_1,...,head_h) * W^O",
         "Multi-head concatenation"),
        ("Eq 2.3", "GQA: n_rep = n_heads / n_kv_heads;  K,V expanded via repeat_interleave",
         "Grouped-query attention"),
    ]:
        story.append(eq(label, formula, note, st))
        story.append(Spacer(1, 0.04*inch))

    story.append(Paragraph(
        "Grouped-Query Attention (GQA) reduces KV cache memory by sharing key and value heads "
        "across groups of query heads. With n_heads=16 and n_kv_heads=8 (the 1B default), KV "
        "cache is halved versus standard MHA. During attention computation K and V are expanded "
        "via <code>repeat_interleave</code> to match query head count before SDPA.",
        st["body"]))

    story.append(Paragraph("2.2  Position Encoding — RoPE and YaRN", st["section"]))
    for label, formula, note in [
        ("Eq 2.4", "RoPE: f_q(x, m) = (W_q * x) * exp(i*m*theta_k)",
         "Rotary frequency at position m"),
        ("Eq 2.5", "theta_k = 1 / (base ^ (2k / d_head)),  k in [0..d_head/2)",
         "Frequency per dimension pair"),
        ("Eq 2.6", "YaRN: scaled_freq = base_freq * (1 - gamma*(1 - 1/scale))",
         "Context extension blending"),
        ("Eq 2.7", "gamma = clamp((wavelength/L_orig - beta_hi)/(beta_lo-beta_hi), 0, 1)",
         "Ramp: 0=extrapolate, 1=interpolate"),
    ]:
        story.append(eq(label, formula, note, st))
        story.append(Spacer(1, 0.04*inch))

    story.append(Paragraph("2.3  Normalization and Feed-Forward", st["section"]))
    for label, formula, note in [
        ("Eq 2.8", "RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma",
         "Pre-norm before each sublayer"),
        ("Eq 2.9", "SwiGLU(x) = Swish(x*W_gate) elementwise* (x*W_up)",
         "Gated feed-forward"),
        ("Eq 2.10","Swish(x) = x * sigmoid(beta * x),  beta=1",
         "Smooth gating function"),
    ]:
        story.append(eq(label, formula, note, st))
        story.append(Spacer(1, 0.04*inch))

    story.append(Paragraph("2.4  Aurelian Memory Core Equations", st["section"]))
    for label, formula, note in [
        ("Eq 2.11","g = sigmoid( Linear([h; forget_gate * mem_read]) )",
         "Surprise gate scalar"),
        ("Eq 2.12","output = h + g * mem_read",
         "Gated memory injection"),
        ("Eq 2.13","LTS score: cosine(k, q)  for content-based addressing",
         "Long-Term Store retrieval"),
        ("Eq 2.14","consolidate if cosine(e_i, e_j) > 0.65",
         "Episodic → LTS threshold"),
    ]:
        story.append(eq(label, formula, note, st))
        story.append(Spacer(1, 0.04*inch))

    story.append(Paragraph("2.5  Alignment and RL", st["section"]))
    for label, formula, note in [
        ("Eq 2.15","PPO: L_clip = E[ min(r*A, clip(r, 1-eps, 1+eps)*A) ]",
         "Clipped surrogate objective"),
        ("Eq 2.16","GRPO: A_i = (r_i - mean(r)) / (std(r) + eps)",
         "Group-relative advantage"),
        ("Eq 2.17","DAPO: clip(r,1-e_lo,1+e_hi) if A>=0 else clip(r,1-e_lo,1+e_lo)",
         "Asymmetric per-token clip"),
        ("Eq 2.18","KL(pi||pi_ref) = sum(pi * log(pi/pi_ref)).clamp(min=0)",
         "KL regularization"),
        ("Eq 2.19","GAE: A_t = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}",
         "Generalized Advantage Estimation"),
        ("Eq 2.20","DPO: L = -log sigma(beta*(log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)))",
         "Direct Preference Optimization"),
    ]:
        story.append(eq(label, formula, note, st))
        story.append(Spacer(1, 0.04*inch))

    story.append(Paragraph("2.6  KV Cache Quantization (KIVI)", st["section"]))
    for label, formula, note in [
        ("Eq 2.21","scale = (max(X) - min(X)) / (2^b - 1),  b=2",
         "KIVI 2-bit asymmetric scale"),
        ("Eq 2.22","X_q = round( (X - min(X)) / scale )",
         "Integer quantization"),
        ("Eq 2.23","KV_bytes = 2 * L * H_kv * S * D * (b/8)",
         "KV cache size formula"),
    ]:
        story.append(eq(label, formula, note, st))
        story.append(Spacer(1, 0.04*inch))

    story.append(Paragraph("2.7  MCTS Planning", st["section"]))
    story.append(eq("Eq 2.24",
        "UCB(s,a) = Q(s,a)/N(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1+N(s,a))",
        "MCTS node selection", st))
    story.append(PageBreak())

    # ── CHAPTER 3 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 3", "System Architecture", st)

    story.append(Paragraph("3.1  Five-Layer Stack", st["section"]))
    story.append(tbl([
        ["Layer", "Key Packages", "Technology", "Purpose"],
        ["Agent Layer", "agent, workflow, multiagent, skill_*", "Python / PyTorch", "Autonomous execution, planning, skills"],
        ["Alignment Layer", "alignment (80+ modules)", "Python / PyTorch", "Safety, preference learning, RLHF"],
        ["Model Layer", "model, memory, longcontext, inference (200+)", "Python / PyTorch", "Core LM, AMC memory, inference"],
        ["Infrastructure", "serving, backends, safety, security, routing", "Python / asyncio", "Serving, reliability, security"],
        ["Rust Layer", "MemoryPageTable, MmapCheckpoint, DifferentialCheckpointer", "Rust (PyO3)", "Zero-copy memory management"],
        ["Frontend", "frontend/, src/ui/, serving/web_ui.py", "TypeScript / React", "User interface, streaming chat"],
        ["CI/CD", ".github/workflows/", "GitHub Actions", "Code quality, security gating"],
    ], [1.2*inch, 2.4*inch, 1.3*inch, 1.85*inch]))
    story.append(Paragraph("Table 3.1 — System Layer Summary", st["caption"]))

    story.append(Paragraph("3.2  Subsystem Package Map", st["section"]))
    story.append(tbl([
        ["Package", "Files", "Description"],
        ["src/inference/", "200+", "All inference strategies: 8 speculative decoding variants, 10 RAG variants, quantization (AWQ, GPTQ, KIVI, FP8, INT4), beam search, constrained decoding, MCTS, DuoAttention, TEAL, Quest, continuous batching"],
        ["src/alignment/", "80+", "Full RLHF: PPO, GRPO, DAPO, RLVR, DPO, KTO, ORPO, IPO, CPO, SimPO, SPIN, REBEL, RLOO, SFT, LoRA variants, reward models, red-teaming, debate"],
        ["src/model/", "200+", "Transformer variants: MoE, RetNet, Mamba, RWKV, linear attention, SSMs, GQA, MLA, mixture-of-depths, and core AureliusTransformer"],
        ["src/agent/", "80+", "Agent runtime, skill system (evolver, registry, catalog, composer), multi-agent swarm, socratic tutor, background executor, goal hierarchy"],
        ["src/memory/", "20+", "Layered memory (5-tier), episodic, long-term, semantic, associative, progressive search, memory fusion, retrieval reranker"],
        ["src/safety/", "25+", "Jailbreak detector, prompt injection scanner, PII detector, hallucination guard, toxicity scorer, SARIF exporter"],
        ["src/workflow/", "15+", "DAG executor, event bus, checkpoint manager, parallel step, scheduler, retry/dead-letter"],
        ["src/retrieval/", "20+", "Dense/BM25/hybrid retrieval, ColBERT, Hyde embedder, multi-hop, hard negative mining"],
    ], [1.3*inch, 0.6*inch, 4.85*inch]))
    story.append(Paragraph("Table 3.2 — Key Subsystem Packages (May 2026)", st["caption"]))

    story.append(Paragraph("3.3  Request Lifecycle", st["section"]))
    story.append(tbl([
        ["Step", "Component", "Action"],
        ["1", "Auth middleware", "Bearer token; fail-closed (require_auth=True)"],
        ["2", "Rate limiter", "Sliding window per API key; async lock-guarded"],
        ["3", "RAG pipeline", "Extract last user message text; embed; retrieve; prepend context"],
        ["4", "Prefix cache", "Exact-match lookup; skip generation on cache hit"],
        ["5", "Chunked prefill", "Split long prompt into memory-bounded chunks"],
        ["6", "S-LoRA adapter", "Select adapter; LRU evict if at capacity; merge weights via try/finally"],
        ["7", "Generation", "Quest attention → generate_paged() → generate() fallback"],
        ["8", "KV offload", "MemoryManager: CUDA stream prefetch + synchronize_prefetch()"],
        ["9", "CrossLayerKV", "CrossLayerKVStack.forward(x, freqs_cis=freqs_cis)"],
        ["10", "LoRA unmerge", "finally block guarantees weight restore even on exception"],
        ["11", "Response", "Stream or batch; safety filter; format"],
    ], [0.35*inch, 1.6*inch, 4.8*inch]))
    story.append(Paragraph("Table 3.3 — End-to-End Request Lifecycle", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 4 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 4", "Model Architecture — Aurelian Memory Core", st)

    story.append(Paragraph("4.1  Model Family", st["section"]))
    story.append(tbl([
        ["Variant", "Params", "d_model", "Heads", "Layers", "d_ff", "d_mem", "Context"],
        ["Aurelius-150M", "150M",   "768",   "12", "12", "3,072",  "256",  "2,048"],
        ["Aurelius-1B*",  "~1.4B",  "1,536", "16", "24", "6,144",  "512",  "4,096"],
        ["Aurelius-2.7B†","~2.7B",  "2,048", "32", "32", "8,192",  "640",  "8,192"],
        ["Aurelius-3B",   "~3.0B",  "2,560", "32", "32", "10,240", "768",  "8,192"],
        ["Aurelius-7B",   "~7.0B",  "3,584", "40", "40", "14,336","1,024","16,384"],
    ], [1.1*inch, 0.7*inch, 0.75*inch, 0.6*inch, 0.6*inch, 0.75*inch, 0.65*inch, 0.8*inch]))
    story.append(Paragraph(
        "Table 4.1 — Model Family. *Primary training target. †Active development branch.", st["caption"]))

    story.append(Paragraph("4.2  Transformer Backbone", st["section"]))
    story.append(Paragraph(
        "All variants: decoder-only autoregressive transformer, <b>pre-RMSNorm</b>, "
        "<b>SwiGLU</b> feed-forward, <b>Grouped-Query Attention</b> with FlashAttention "
        "dispatch via <code>F.scaled_dot_product_attention</code>, <b>RoPE</b> with optional "
        "YaRN extension, and weight tying between token embedding table and LM head. "
        "No positional bias terms. A <code>max_seq_len</code> guard was added to "
        "<code>generate()</code> and <code>generate_stream()</code> to prevent RoPE buffer "
        "out-of-bounds crashes on long inputs.",
        st["body"]))

    story.append(Paragraph("4.3  The Aurelian Memory Core (AMC)", st["section"]))

    story.append(Paragraph("4.3.1  Tier 1 — Working Memory", st["subsection"]))
    story.append(Paragraph(
        "The transformer layer's hidden states constitute working memory: ephemeral, "
        "high-bandwidth, bounded to the current forward pass. No additional parameters.",
        st["body"]))

    story.append(Paragraph("4.3.2  Tier 2 — Episodic Buffer", st["subsection"]))
    story.append(Paragraph(
        "A bidirectional GRU encodes memory slots. A <b>SurpriseGate</b> computes scalar "
        "gate <i>g</i> (Eq 2.11) controlling whether the current hidden state is written "
        "to the buffer. A forget gate clears stale content. The output is the sum of "
        "hidden state and gated memory read (Eq 2.12). This teaches the model to be "
        "selective: only surprising or novel tokens update episodic memory.",
        st["body"]))

    story.append(Paragraph("4.3.3  Tier 3 — Long-Term Store (LTS)", st["subsection"]))
    story.append(Paragraph(
        "Fixed-capacity key-value memory with content-based cosine addressing. Written via "
        "graph consolidation: episodic slots with cosine similarity above 0.65 are clustered "
        "and importance-weighted top-k entries are promoted. Backed by a Rust "
        "<code>MemoryPageTable</code> with LRU eviction. Capacity: 1,024–8,192 entries.",
        st["body"]))

    story.append(Paragraph("4.4  2.7B Variant — In Development", st["section"]))
    story.append(Paragraph(
        "Branch <code>feat/1-scale-2_7b-config-stack</code> (50 commits ahead, 562 files) "
        "introduces d_model=2,048, 32 heads, 32 layers, d_ff=8,192 — a clean stepping stone "
        "between 1B and 3B. It adds tensor-parallel training configurations, revised memory "
        "budgets, extended KV quantization defaults, and updated AMC slot counts. "
        "This is the highest-priority merge-ready branch.",
        st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 5 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 5", "VRAM / RAM Efficiency System", st)

    story.append(Paragraph(
        "Serving and training billion-parameter models on constrained hardware required "
        "a multi-phase efficiency engineering program. The system implements 15+ techniques "
        "organized into three phases.",
        st["body"]))

    story.append(Paragraph("5.1  Phase 1 — Wire Existing Modules", st["section"]))
    story.append(tbl([
        ["Module", "File", "Memory Impact", "Status"],
        ["PagedKVCache", "src/longcontext/paged_kv_cache.py", "Eliminates fragmentation", "Wired"],
        ["PrefixCache",  "src/longcontext/prefix_cache.py",   "Shared prefix reuse; skip-on-hit", "Wired"],
        ["KVCacheQuantizer (INT8)", "src/longcontext/kv_cache_quantization.py", "−75% KV memory", "Wired"],
        ["ChunkedPrefillScheduler", "src/longcontext/chunk_prefill.py", "Memory-bounded prefill", "Wired"],
        ["ModelRouter",  "src/serving/model_router.py",        "Capacity-aware routing", "Wired"],
        ["RetrievalPipeline", "src/retrieval/pipeline.py",     "RAG offloads LTS lookups", "Wired"],
        ["FP8 AllReduce", "src/training/fp8_communication.py",  "−75% gradient bandwidth", "Wired"],
        ["CrossLayerKVSharing", "src/longcontext/cross_layer_kv_sharing.py", "−50% KV (share_every_n=2)", "Wired + RoPE fixed"],
    ], [1.6*inch, 2.5*inch, 1.6*inch, 0.9*inch]))
    story.append(Paragraph("Table 5.1 — Phase 1 Efficiency Modules", st["caption"]))

    story.append(Paragraph("5.2  Phase 2 — Quantization and Sparsity", st["section"]))
    story.append(tbl([
        ["Module", "Technique", "Reduction", "Key Formula"],
        ["KIVI Quantizer", "2-bit asymmetric KV quant", "−87.5% vs FP16 KV", "scale=(max-min)/(2^b-1)"],
        ["DuoAttention Manager", "Streaming heads (sink+recent) vs retrieval heads (full KV)", "~50% KV eviction", "Per-head mask at decode"],
        ["TEAL Sparsity", "Zero activations below magnitude threshold", "30-50% FLOP reduction", "mask=(|x|>threshold)"],
    ], [1.3*inch, 2.1*inch, 1.4*inch, 2.0*inch]))
    story.append(Paragraph("Table 5.2 — Phase 2 Quantization and Sparsity", st["caption"]))

    story.append(Paragraph("5.3  Phase 3 — Advanced Memory Management", st["section"]))
    story.append(tbl([
        ["Module", "Technique", "Benefit"],
        ["Quest Attention", "Page-sparse: attend only to top-K KV pages by importance score", "Sub-quadratic O(k) attention vs O(n) for long contexts"],
        ["CPU KV Offload (InfLLM)", "3-tier: GPU hot → CPU pinned → SSD; async H2D prefetch on dedicated CUDA stream", "Unbounded effective KV; overlaps prefetch with compute"],
        ["S-LoRA", "Multi-adapter serving: LRU eviction when at capacity; merge/unmerge via try/finally", "Serve many LoRA fine-tunes from one base model"],
    ], [1.4*inch, 2.9*inch, 2.45*inch]))
    story.append(Paragraph("Table 5.3 — Phase 3 Advanced Memory Modules", st["caption"]))

    story.append(Paragraph("5.4  Memory Reduction by Technique", st["section"]))
    chart = bar_chart_drawing(
        [75, 87, 50, 75, 50, 30],
        ["INT8 KV", "KIVI 2-bit", "CrossLayer\nKV", "FP8\nAllReduce", "DuoAttn\n(est)", "TEAL\nFLOPs"],
        "Memory / Compute Reduction by Technique (%)",
        bar_color=STEEL)
    if chart:
        story.append(chart)
        story.append(Paragraph(
            "Figure 5.1 — Estimated reduction % per technique vs. baseline. "
            "KIVI 2-bit vs FP16 KV cache.", st["caption"]))

    story.append(Paragraph("5.5  Critical Bug Fixes During Wiring", st["section"]))
    for item in [
        "<b>CrossLayerKVStack RoPE:</b> stack.forward() was not passing freqs_cis to layers. "
        "Added local _apply_rope() (cross_layer_kv_sharing.py is hermetic — cannot import from src.model) "
        "and threaded freqs_cis through the full stack.",
        "<b>CPU CUDA guard:</b> prefetch_to_gpu() called .cuda() unconditionally, crashing without GPU. "
        "Added if not torch.cuda.is_available() guard returning CPU tensor.",
        "<b>LoRA unmerge safety:</b> unmerge was in straight-line code, bypassed by exception return paths "
        "and permanently corrupting model weights. Wrapped generation in try/finally.",
        "<b>RAG query extraction:</b> pipeline.run() received chat-formatted text with special tokens. "
        "Fixed to extract raw text of last user message before retrieval.",
    ]:
        story.append(Paragraph(f"• {item}", st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 6 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 6", "Inference Engine — 200+ Modules", st)

    story.append(Paragraph(
        "The <code>src/inference/</code> package covers essentially every known inference optimization "
        "technique as of 2025–2026, making it one of the most complete inference reference "
        "implementations in any open research project.",
        st["body"]))

    story.append(Paragraph("6.1  Speculative Decoding Family (8 Variants)", st["section"]))
    story.append(tbl([
        ["Module", "Technique", "Key Innovation"],
        ["speculative_decoding.py", "Standard speculative", "Draft → target verify; rollback on mismatch"],
        ["eagle_speculative.py", "EAGLE", "Draft uses target hidden states; better acceptance rate"],
        ["eagle2_decoding.py", "EAGLE-2", "Dynamic draft length based on real-time acceptance rate"],
        ["eagle3_decoding.py", "EAGLE-3", "Multi-layer draft features; improved alignment to target"],
        ["medusa.py", "Medusa", "Multiple draft heads on target model; no separate draft needed"],
        ["cascade_speculative.py", "Cascade", "Hierarchical draft cascade for very long outputs"],
        ["hydra_speculative.py", "Hydra", "Multiple independent draft models; consensus selection"],
        ["self_speculative_decoding.py", "Self-speculative", "Early exit layers as draft; zero extra params"],
    ], [2.0*inch, 1.2*inch, 3.55*inch]))
    story.append(Paragraph("Table 6.1 — Speculative Decoding Variants", st["caption"]))

    story.append(Paragraph("6.2  KV Cache Optimization Stack", st["section"]))
    story.append(tbl([
        ["Module", "Technique", "Reduction"],
        ["kivi_quant.py", "2-bit asymmetric per-channel quantization", "−87.5% vs FP16"],
        ["kv_cache_quantization.py", "INT8 simulation (noise injection)", "−75% vs FP32"],
        ["h2o_kv.py", "Heavy-Hitter Oracle: evict by cumulative attention score", "Top-k retention"],
        ["pyramid_kv.py", "Fewer tokens retained at higher layers", "Layer-adaptive budget"],
        ["snapkv.py", "Compress via observation pooling before generation", "Reduce prefill KV"],
        ["quest_attention.py", "Score pages by max query-key product; attend top-K only", "Sub-quadratic"],
        ["duo_attention.py", "Streaming heads keep sink+recent; retrieval heads keep full KV", "Per-head policy"],
        ["teal_sparsity.py", "Zero activations below magnitude threshold", "30-50% FLOP saving"],
    ], [1.8*inch, 2.6*inch, 1.85*inch]))
    story.append(Paragraph("Table 6.2 — KV Cache and Sparsity Modules", st["caption"]))

    story.append(Paragraph("6.3  RAG and Retrieval Variants", st["section"]))
    story.append(Paragraph(
        "Ten RAG variants are implemented: basic dense RAG, fusion-in-decoder (FiD), speculative RAG "
        "(draft with RAG context, verify with full), RAG fusion (multiple query rewrites), attributed "
        "RAG (citation tracking), ICL retrieval (few-shot example selection), and more. The production "
        "path auto-instantiates via <code>RetrievalPipeline.from_defaults()</code>. Query extraction "
        "uses raw last-user-message text — not chat-formatted text with special tokens.",
        st["body"]))

    story.append(Paragraph("6.4  Reasoning Modules", st["section"]))
    story.append(Paragraph(
        "Inference-time reasoning: Chain of Thought (v1/v2), Tree of Thought, MCTS in latent space, "
        "CoT with Process Reward Models, Chain of Draft, Soft Thinking (smooth reasoning tokens), "
        "COCONUT (continuous chain of thought), Best-of-N reranking, Self-Consistency (K=5 paths), "
        "Majority Voting. These modules trade compute for reasoning quality at inference time.",
        st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 7 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 7", "Alignment Stack — 80+ Algorithms", st)

    story.append(Paragraph(
        "The alignment package has grown to over 80 modules covering the complete arc of modern "
        "RLHF research from 2017 (PPO) through 2025 (DAPO, DR-GRPO, STILL, REBEL).",
        st["body"]))

    story.append(Paragraph("7.1  Core RLHF Algorithms", st["section"]))
    story.append(tbl([
        ["Algorithm", "Module", "Key Innovation", "Bugs Fixed"],
        ["Constitutional AI", "constitutional_ai.py", "KL regularization toward safe reference", "KL args swapped — gradient through frozen ref; fixed"],
        ["PPO + GAE", "ppo_trainer.py", "Clipped surrogate + value function + entropy", "NameError; off-by-one logit gather; detached hook"],
        ["GRPO", "grpo.py", "Group-relative advantages; no value network", "Unclipped KL; fixed with clamp(min=0)"],
        ["DAPO", "dapo.py", "Asymmetric clip per advantage sign", "Uniform clip applied (paper not implemented); fixed"],
        ["RLVR", "rlvr.py", "Verifiable outcome rewards (math/code)", "NaN from std() on n=1; negative KL; both fixed"],
        ["DPO", "dpo.py", "Preference pairs; no reward model needed", "—"],
        ["KTO", "kto_trainer.py", "Binary feedback; Kahneman-Tversky theory", "—"],
        ["ORPO", "orpo_trainer.py", "Odds ratio; no reference model", "—"],
        ["SimPO", "simpo.py", "Reference-free; sequence-average log-probs", "—"],
        ["SPIN", "spin_trainer.py", "Self-play from SFT data alone", "—"],
        ["REBEL", "rebel.py", "Regress reward-to-go; value-based RL", "—"],
        ["RLOO", "rloo.py", "REINFORCE leave-one-out variance reduction", "—"],
    ], [1.2*inch, 1.65*inch, 2.35*inch, 1.55*inch]))
    story.append(Paragraph("Table 7.1 — Core Alignment Algorithms", st["caption"]))

    story.append(Paragraph("7.2  Algorithm Complexity Chart", st["section"]))
    chart2 = bar_chart_drawing(
        [3, 4, 2, 2, 2, 3, 2, 1, 1, 1],
        ["CAI", "PPO", "GRPO", "DAPO", "RLVR", "DPO", "KTO", "ORPO", "SimPO", "SPIN"],
        "Relative Implementation Complexity (1=simple, 4=most complex)",
        bar_color=GOLD)
    if chart2:
        story.append(chart2)
        story.append(Paragraph("Figure 7.1 — Relative implementation complexity of core alignment algorithms.", st["caption"]))

    story.append(Paragraph("7.3  Advanced Alignment Categories", st["section"]))
    story.append(tbl([
        ["Category", "Modules", "Purpose"],
        ["Constitutional", "constitutional_ai_v2/v3.py, constitutional_committee.py, constitution_dimensions.py", "Multi-principle safety; committee review; dimension scoring"],
        ["Debate", "debate_framework.py, debate_irving.py, self_play_debate.py", "Adversarial debate for truth-seeking; Irving protocol"],
        ["Reward Modeling", "reward_model.py, reward_ensemble.py, reward_calibration.py, reward_soup.py", "Ensemble, calibration, model merging for robust rewards"],
        ["Red Teaming", "red_team.py, red_team_auto.py, adversarial_training.py", "Automated adversarial prompt generation and training"],
        ["PEFT", "lora_variants.py, dora.py, ia3.py, vera.py, adalora.py, prefix_tuning.py", "LoRA family, IA3, VeRA, AdaLoRA, prefix tuning"],
        ["Process Supervision", "process_reward.py, step_dpo.py, stepwise_dpo.py", "Step-level reward and preference learning"],
    ], [1.3*inch, 2.8*inch, 2.65*inch]))
    story.append(Paragraph("Table 7.2 — Advanced Alignment Categories", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 8 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 8", "Autonomous Agent System", st)

    story.append(Paragraph("8.1  ReAct Loop Architecture", st["section"]))
    story.append(tbl([
        ["Phase", "Module", "Implementation", "Innovation"],
        ["Observe", "ToolFormerAdapter", "Cross-attention over tool embedding table", "Learned tool-context fusion into hidden state"],
        ["Think", "PlanningModule + ValueHead", "MCTS with UCB (c_puct=1.4)", "8–24 latent-space simulations per step"],
        ["Act", "ToolCallHead + SkillLibrary", "Categorical tool selection + param generation", "Gated FiLM skill conditioning; online acquisition"],
        ["Reflect", "CriticHead + ExperienceReplayBuffer", "Score + suggestion vector", "Suggestion injected back into hidden state"],
    ], [0.7*inch, 1.5*inch, 2.1*inch, 2.45*inch]))
    story.append(Paragraph("Table 8.1 — ReAct Agent Loop Phases", st["caption"]))

    story.append(Paragraph("8.2  Self-Evolving Skill Library", st["section"]))
    story.append(Paragraph(
        "Six skill modules: <code>skill_library.py</code> (retrieval via top-k dot-product), "
        "<code>skill_registry.py</code> (ranked by success_rate × log(usage+1)), "
        "<code>skill_evolver.py</code> (crystallize successful trajectories into reusable skills "
        "using momentum encoder τ=0.99), <code>skill_catalog.py</code> (structured taxonomy), "
        "<code>skill_composer.py</code> (merge skills for complex tasks), and "
        "<code>skill_trigger_engine.py</code> (condition-based auto-firing).",
        st["body"]))

    story.append(Paragraph("8.3  Multi-Agent Coordination", st["section"]))
    story.append(Paragraph(
        "<code>agent_swarm.py</code> and <code>swarm_scaler.py</code> coordinate specialized "
        "sub-agents. <code>multi_agent.py</code> implements three coordination patterns: "
        "consensus voting (all agents solve, majority wins), delegation chains (agent A delegates "
        "subtasks to B and C), and competition (multiple agents solve independently, best selected). "
        "<code>specialist_agents.py</code> defines 10 domain specialists: researcher, coder, math, "
        "critic, safety, summarizer, planner, reviewer, debugger, teacher.",
        st["body"]))

    story.append(Paragraph("8.4  Neural Brain (10-Module Cognitive Stack)", st["section"]))
    story.append(tbl([
        ["Module", "Function", "Training Signal"],
        ["Perception Encoder", "Fuse text, tool output, memory, system state", "End-to-end RL gradient"],
        ["Working Memory", "64-slot differentiable scratchpad with importance gating", "REINFORCE on task completion"],
        ["Long-Term Memory", "3-store (factual, procedural, episodic) with reranker", "Contrastive retrieval loss"],
        ["Reasoning Core", "Multi-step CoT; self-consistency K=5; recursive thought", "PRM-guided REINFORCE + KL distillation"],
        ["Planning Network", "DAG decomposition; dependency prediction; replanning", "Supervised + RL on plan execution"],
        ["Tool Controller", "Cosine similarity selection + param generation + retry", "Behavioral cloning + tool success reward"],
        ["Agent Router", "10 specialized sub-agents; dynamic assembly per task", "Multi-task behavioral cloning"],
        ["Verifier / Critic", "Error detection; confidence calibration; fix generation", "Labeled error data + ECE calibration"],
        ["Reflection Module", "Post-task trajectory analysis; memory update", "Contrastive trajectory learning"],
        ["Executive Controller", "Orchestrates all modules; budget-aware; loop detection", "PPO (correctness + efficiency + cost)"],
    ], [1.45*inch, 2.65*inch, 2.65*inch]))
    story.append(Paragraph("Table 8.2 — Neural Brain Modules", st["caption"]))

    story.append(Paragraph("8.5  New Capabilities (Cycles 209–213)", st["section"]))
    story.append(tbl([
        ["Feature", "Module", "Source Inspiration"],
        ["Agent Mode Registry", "agent_mode_registry.py", "Roo-Code — Code/Architect/Ask/Debug/Custom presets"],
        ["Workflow DAG Executor", "src/workflow/dag_executor.py", "Archon — YAML-free DAG with depends_on, loop_until"],
        ["Layered Memory (5-tier)", "src/memory/layered_memory.py", "GenericAgent — L0 Rules → L1 Insights → L2 Facts → L3 Skills → L4 Archive"],
        ["Progressive Search", "src/memory/progressive_search.py", "claude-mem — 3-layer: index → timeline → full; ~10x token savings"],
        ["Skill Evolver", "src/agent/skill_evolver.py", "GenericAgent — crystallize execution paths into skills"],
        ["Socratic Tutor", "src/agent/socratic_tutor.py", "DeepTutor — hint-based educational agent mode"],
        ["SRE Golden Signals", "src/monitoring/sre_metrics.py", "OpenSRE — latency/traffic/errors/saturation"],
    ], [1.5*inch, 2.3*inch, 2.95*inch]))
    story.append(Paragraph("Table 8.3 — New Agent Capabilities from Feature Audit", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 9 ─────────────────────────────────────────────────────────────
    story += ch("Chapter 9", "Safety, Security, and Hardening", st)

    story.append(Paragraph("9.1  AUR-SEC-2026 Hardening Pass (27 Findings Closed)", st["section"]))
    story.append(Paragraph(
        "Branch <code>fix/hardening-pass-20260429</code> (14 commits, 98 files) closed all 27 "
        "security findings from AUR-SEC-2026-0001 through 0027. It is merge-ready with no conflicts.",
        st["body"]))
    story.append(tbl([
        ["Finding(s)", "Component", "Vulnerability", "Fix"],
        ["0001–0003", "sandbox_executor.py, code_execution.py, code_eval.py",
         "Sandbox escape via object.__subclasses__() — attacker traverses class hierarchy",
         "Blocked __subclasses__, __bases__, __mro__ in execution namespace"],
        ["0004", "http_backend.py",
         "No SSRF protection — server could proxy to internal services (169.254.x.x etc)",
         "Added IP blocklist; validates against private/reserved ranges before any request"],
        ["0005", "auth_middleware.py",
         "DEFAULT_AUTH_MIDDLEWARE was fail-open (require_auth=False)",
         "Changed to require_auth=True; fail-closed by default"],
        ["0006", "shell_tool.py",
         "shell=True + denylist — bypassable via quoting, encoding tricks",
         "Replaced with shell=False + shlex.split + explicit command allow-list"],
        ["0007", "deploy.yml",
         "Secrets interpolated in run: steps — logged in plaintext by GitHub",
         "Bound to env: vars; permissions: read + packages:write only"],
        ["0008–0027", "Various",
         "Path traversal, ReDoS regexes, HMAC auth, canary tokens, audit logging, CSP, CORS",
         "All closed in hardening branch; see SECURITY_AUDIT.md for full details"],
    ], [0.75*inch, 1.5*inch, 2.2*inch, 2.3*inch]))
    story.append(Paragraph("Table 9.1 — AUR-SEC-2026 Hardening Summary", st["caption"]))

    story.append(Paragraph("9.2  Safety Module Layer (src/safety/ — 25+ modules)", st["section"]))
    story.append(tbl([
        ["Module", "Function"],
        ["jailbreak_detector.py / jailbreak_v2.py", "Pattern + embedding jailbreak detection; two independent classifiers"],
        ["prompt_injection_detector.py / scanner.py", "Detect injection in user messages and tool outputs"],
        ["pii_detector.py", "Detect and redact PII (names, emails, SSNs, credit cards)"],
        ["hallucination_guard.py", "Check factual claims against retrieved context"],
        ["toxicity_scorer.py", "Multi-dimension toxicity scoring; configurable thresholds"],
        ["output_safety_filter.py / output_sanitizer.py", "Final output pass; strip unsafe content before response"],
        ["canary_token_guard.py", "Detect canary tokens in output (data exfiltration signal)"],
        ["malicious_code_detector.py", "Pattern-match generated code against known malicious signatures"],
        ["sarif_exporter.py", "Export security findings in SARIF format for IDE/CI integration"],
    ], [2.5*inch, 4.25*inch]))
    story.append(Paragraph("Table 9.2 — Safety Module Layer", st["caption"]))

    story.append(Paragraph("9.3  Original Audit Summary (Pre-Hardening)", st["section"]))
    story.append(tbl([
        ["Domain", "Critical", "High", "Medium", "Total", "Remediated"],
        ["Security / Backends", "4", "6", "5", "15", "15 / 15"],
        ["Training / Alignment", "3", "4", "4", "11", "11 / 11"],
        ["CI/CD / Config / Plugins", "4", "4", "5", "13", "11 / 13"],
        ["TOTAL", "11", "14", "14", "39", "37 / 39"],
    ], [2.1*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.25*inch]))
    story.append(Paragraph("Table 9.3 — Original Audit Issue Summary (all Critical remediated)", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 10 ────────────────────────────────────────────────────────────
    story += ch("Chapter 10", "Memory System and RAG Pipeline", st)

    story.append(Paragraph("10.1  RAG Pipeline Integration", st["section"]))
    story.append(tbl([
        ["Stage", "Component", "Description"],
        ["Query extraction", "api_server.py", "Raw text of last user message — NOT chat-formatted with special tokens"],
        ["Embedding", "dense_retriever.py / hyde_embedder.py", "Encode query; optional HyDE for better recall"],
        ["Retrieval", "hybrid_retriever.py", "Dense + BM25 via Reciprocal Rank Fusion (RRF)"],
        ["Reranking", "cross_encoder_reranker.py / colbert_reranker.py", "Cross-encoder or ColBERT late interaction"],
        ["Injection", "api_server.py", "Prepend retrieved passages before generation"],
        ["Attribution", "attributed_rag.py", "Track which passage supports each output claim"],
    ], [1.3*inch, 2.0*inch, 3.45*inch]))
    story.append(Paragraph("Table 10.1 — RAG Pipeline Stages", st["caption"]))

    story.append(Paragraph("10.2  Five-Tier Layered Memory", st["section"]))
    story.append(tbl([
        ["Layer", "Contents", "TTL", "Access Cost"],
        ["L0 — Meta Rules", "Constitutional principles, hard constraints", "Permanent", "Always loaded"],
        ["L1 — Insight Index", "Compressed summaries of past sessions", "30 days", "Index lookup"],
        ["L2 — Global Facts", "Domain knowledge, user preferences", "90 days", "Embedding search"],
        ["L3 — Task Skills", "Crystallized skill templates (SkillEvolver)", "Permanent if ranked", "Registry lookup"],
        ["L4 — Session Archive", "Full session transcripts and trajectories", "7 days", "Full retrieval"],
    ], [1.1*inch, 2.3*inch, 0.9*inch, 1.4*inch]))
    story.append(Paragraph("Table 10.2 — Five-Tier Layered Memory (GenericAgent / Cycle 210)", st["caption"]))

    story.append(Paragraph("10.3  Progressive Search (3-Pass, ~10x Token Savings)", st["section"]))
    story.append(Paragraph(
        "Pass 1 (~100 tokens): search compact ID index. "
        "Pass 2 (~500 tokens): retrieve timeline context around matching IDs. "
        "Pass 3 (~2,000 tokens): fetch full details only for final filtered IDs. "
        "Most queries are satisfied at Pass 1 or 2, avoiding the 2K-token cost of Pass 3.",
        st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 11 ────────────────────────────────────────────────────────────
    story += ch("Chapter 11", "Training Pipeline", st)

    story.append(Paragraph("11.1  Three-Phase Curriculum", st["section"]))
    story.append(tbl([
        ["Phase", "Objective", "Loss", "Data"],
        ["1 — LM Pretraining", "Next-token prediction", "CrossEntropy(logits, labels)", "200B tokens (1B)"],
        ["2 — Memory Curriculum", "Surprise-gate selectivity + LTS consolidation",
         "CE + λ_mem·mean(g²) + λ_consol·L_consol", "10B memory-diverse tokens"],
        ["3 — Agent Fine-Tuning", "Tool use, planning, RL from task rewards",
         "CE(tools) + PPO(RL) via LoRA", "500M agent demos (3B only)"],
    ], [1.2*inch, 1.8*inch, 2.4*inch, 1.35*inch]))
    story.append(Paragraph("Table 11.1 — Three-Phase Training Curriculum", st["caption"]))

    story.append(Paragraph("11.2  Key Hyperparameters (1B)", st["section"]))
    story.append(tbl([
        ["Hyperparameter", "Value", "Notes"],
        ["Optimizer", "AdamW (β₁=0.9, β₂=0.95)", "Muon optimizer available; state now saved/loaded in checkpoints"],
        ["Learning Rate", "1.5e-4 cosine → 1e-5", "3,000 warmup steps"],
        ["Weight Decay", "0.1", "Non-bias, non-norm params only"],
        ["Gradient Clipping", "1.0 (global norm)", "ZClip variant for adaptive clipping"],
        ["Precision", "BF16 (FP32 master)", "FP8 AllReduce for gradient communication"],
        ["Distributed", "FSDP (8 × H100)", "ZeRO-2 optimizer state sharding"],
        ["Activation Checkpointing", "Enabled", "−60% activation memory"],
        ["Max Seq Len", "4,096 (guarded)", "Guard added to prevent RoPE OOB (changelog fix)"],
        ["CLI Tokenizer", "AureliusTokenizer (with fallback)", "Fixed: now loads real tokenizer before byte fallback"],
    ], [2.1*inch, 2.1*inch, 2.55*inch]))
    story.append(Paragraph("Table 11.2 — 1B Training Hyperparameters", st["caption"]))

    story.append(Paragraph("11.3  Infrastructure", st["section"]))
    story.append(tbl([
        ["Variant", "GPUs", "Strategy", "Est. Cost"],
        ["150M", "1 × H100-80GB", "DDP", "~$10"],
        ["1B (~1.4B)", "8 × H100-80GB", "FSDP", "~$350"],
        ["2.7B (dev)", "16 × H100-80GB", "FSDP + TP×2", "~$1,400"],
        ["3B", "16 × H100-80GB", "FSDP + TP×2 + PP×2", "~$2,800"],
        ["7B", "32 × H100-80GB", "FSDP + TP×4 + PP×2", "~$8,000"],
    ], [1.0*inch, 1.5*inch, 2.3*inch, 1.95*inch]))
    story.append(Paragraph("Table 11.3 — Training Infrastructure", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 12 ────────────────────────────────────────────────────────────
    story += ch("Chapter 12", "CI/CD, DevOps, and Branch Management", st)

    story.append(Paragraph("12.1  Rewritten GitHub Actions Pipeline", st["section"]))
    story.append(Paragraph(
        "The CI/CD pipeline was fully rewritten in the current changelog cycle: Python 3.12/3.13 "
        "matrix, full pytest suite, ruff lint+format, Bandit (medium+, no continue-on-error), "
        "and pip-audit. Triggers on main plus cycle/*, sec/*, feat/*, deploy/* branches.",
        st["body"]))
    story.append(tbl([
        ["Workflow", "Trigger", "Jobs", "Security Gates"],
        ["ci.yml", "push/PR main + cycle/*/sec/*/feat/*/deploy/*",
         "Python 3.12+3.13 matrix, pytest, ruff, Bandit, pip-audit, Rust cargo-audit",
         "Bandit medium+ (no continue-on-error), pip-audit"],
        ["deploy.yml", "CI success on main", "Docker build + registry push",
         "permissions: contents: read, packages: write only"],
        ["ruff-autofix.yml", "push non-main", "Ruff format+fix, create PR",
         "No direct write to main"],
    ], [1.3*inch, 1.9*inch, 2.2*inch, 1.35*inch]))
    story.append(Paragraph("Table 12.1 — GitHub Actions Workflows", st["caption"]))

    story.append(Paragraph("12.2  Branch Audit Results (May 2026)", st["section"]))
    story.append(tbl([
        ["Branch", "Ahead", "Files", "Action"],
        ["fix/hardening-pass-20260429", "14", "98", "MERGE — closes AUR-SEC-0001–0027"],
        ["feat/1-scale-2_7b-config-stack", "50", "562", "MERGE — 2.7B model config"],
        ["wip/context-drift-recovery", "55", "525", "REVIEW + MERGE"],
        ["feat/1-scale-2_7b-config", "2", "330", "MERGE (superset of config-stack)"],
        ["dependabot/npm_and_yarn/…", "1", "2", "MERGE — dependency bump"],
        ["30 stale cycle/sec branches", "—", "—", "CLOSE and archive"],
    ], [2.8*inch, 0.65*inch, 0.6*inch, 2.7*inch]))
    story.append(Paragraph("Table 12.2 — Merge-Ready Branch Queue", st["caption"]))

    story.append(Paragraph("12.3  Working Copy Discrepancy", st["section"]))
    story.append(Paragraph(
        f"The branch audit identified a divergence between <code>{REPO_ROOT}/</code> "
        "(55+ Python files with model improvements, 262 passing tests, FastAPI server, recursive MAS) "
        "and the remote repository. Recommended action: create a "
        "<code>feat/model-core-improvements</code> branch, copy updated files, and push. "
        f"The Desktop repo at <code>{REPO_ROOT}/</code> has the latest commits.",
        st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 13 ────────────────────────────────────────────────────────────
    story += ch("Chapter 13", "Serving Infrastructure", st)

    story.append(Paragraph("13.1  API Server", st["section"]))
    story.append(Paragraph(
        "Primary serving: <code>src/serving/api_server.py</code> — OpenAI-compatible chat completions, "
        "SSE streaming, function calling, and Responses API. Additional: <code>aurelius_api.py</code> "
        "(Aurelius-native), <code>aurelius_server.py</code> (FastAPI alternative).",
        st["body"]))

    story.append(Paragraph("13.2  Reliability Stack", st["section"]))
    story.append(tbl([
        ["Component", "Module", "Behavior"],
        ["Circuit Breaker", "circuit_breaker.py", "3-state FSM; thread-safe; trips after N failures"],
        ["Rate Limiter", "rate_limiter.py / rate_limiter_v2.py", "Async lock-guarded; per-key; burst multiplier"],
        ["Load Balancer", "load_balancer.py", "Weighted round-robin; health-check aware"],
        ["Request Coalescer", "request_coalescer.py", "Merge identical in-flight requests"],
        ["Load Shedder", "load_shedder.py", "Reject low-priority under high load"],
        ["Continuous Batching", "continuous_batching.py", "vLLM-style in-flight request batching"],
    ], [1.4*inch, 1.9*inch, 3.45*inch]))
    story.append(Paragraph("Table 13.1 — Serving Reliability Components", st["caption"]))

    story.append(Paragraph("13.3  Middleware Order", st["section"]))
    story.append(Paragraph(
        "auth_middleware (fail-closed) → cors_middleware → rate_limiter → guardrail_middleware → "
        "metrics_middleware → generation → output_safety_filter → response_formatter.",
        st["body"]))
    story.append(PageBreak())

    # ── CHAPTER 14 ────────────────────────────────────────────────────────────
    story += ch("Chapter 14", "Testing and Architecture Debt", st)

    story.append(Paragraph("14.1  Test Coverage", st["section"]))
    story.append(tbl([
        ["Test File", "Component", "Status"],
        ["tests/model/test_multi_token_prediction.py", "Multi-token prediction head", "Implemented"],
        ["tests/optimizers/test_adaptive_clipper.py", "Adaptive gradient clipper", "Implemented"],
        ["tests/test_feature_flag_registry.py + 157 others (Cycle 199)", "FeatureFlagRegistry; integration", "Implemented"],
        ["tests/test_memory_core.py", "AurelianMemoryCore shapes, LTS capacity", "Planned (Critical)"],
        ["tests/test_aurelius_model.py", "AureliusModel1B logits, generate()", "Planned (Critical)"],
        ["tests/test_rlhf_trainers.py", "PPO, GRPO, DAPO end-to-end", "Planned (Critical)"],
        ["tests/test_kv_efficiency.py", "KIVI, DuoAttention, Quest attention", "Planned (High)"],
        ["tests/test_agent_loop.py", "AgentLoopController, ExperienceReplayBuffer", "Planned (High)"],
    ], [2.8*inch, 2.3*inch, 1.65*inch]))
    story.append(Paragraph("Table 14.1 — Test Coverage (May 2026)", st["caption"]))

    story.append(Paragraph("14.2  Architecture Debt", st["section"]))
    story.append(tbl([
        ["Priority", "Issue", "Effort"],
        ["P0", "aurelius_model_3b.py imports agent_loop inline — cycle risk", "30 min"],
        ["P0", "fused_kernels.py imports aurelius_model_1b — reversed dependency", "15 min"],
        ["P0", "Rust memory crate compiled but no Python imports it", "1 hr"],
        ["P1", "Working directory aurelius/ diverged from remote — 55+ local-only files", "PR + reconcile"],
        ["P1", "5 memory support files should consolidate to 2", "1 hr"],
        ["P2", "GraphConsolidator (35-line nn.Module) never called — dead class", "10 min"],
        ["P3", "Import-cycle detection missing from CI", "20 min"],
    ], [0.45*inch, 4.2*inch, 0.8*inch]))
    story.append(Paragraph("Table 14.2 — Architecture Debt Backlog", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 15 ────────────────────────────────────────────────────────────
    story += ch("Chapter 15", "Roadmap and Related Work", st)

    story.append(Paragraph("15.1  Feature Roadmap (Cycles 209–217)", st["section"]))
    story.append(tbl([
        ["Cycle", "Theme", "Deliverables"],
        ["209", "Agent Core", "agent_mode_registry, workflow DAG executor, output compressor"],
        ["210", "Memory & Learning", "layered_memory (5-tier), progressive_search, skill_evolver"],
        ["211", "Tools & Tutoring", "document_converter (stdlib), socratic_tutor, task_scheduler"],
        ["212", "Multi-Agent", "multi_agent_coordination, pipeline_processor, tutorial_engine"],
        ["213", "Monitoring", "sre_golden_signals, ANSI terminal dashboard"],
        ["214–217", "Frontend Stability", "Workflow, chat, dashboard, settings deepening"],
        ["TBD", "2.7B Training", "Merge feat/1-scale-2_7b-config-stack; begin training run"],
        ["TBD", "Rust Bridge", "Wire MemoryPageTable into production Python"],
        ["TBD", "Benchmarks", "GSM8K, MATH, HumanEval, MMLU eval harness; publish"],
    ], [0.6*inch, 1.4*inch, 4.75*inch]))
    story.append(Paragraph("Table 15.1 — Feature Roadmap", st["caption"]))

    story.append(Paragraph("15.2  Comparative Context", st["section"]))
    story.append(tbl([
        ["System", "Memory Approach", "vs Aurelius AMC"],
        ["Standard GPT/LLaMA", "Context window only", "AMC adds per-layer episodic + LTS beyond context window"],
        ["MemGPT", "External memory via tool calls", "AMC is differentiable and in-forward-pass; no API latency"],
        ["RAG systems", "Retrieval only; no persistent write", "Aurelius: RAG + three-tier + episodic learning combined"],
        ["RetNet / RWKV", "Recurrent memory; parallel train", "AMC is differentiable and operates on full KV"],
        ["Titans (Meta, 2025)", "Neural long-term memory; surprise-based", "AMC independently derived; per-transformer-layer integration"],
    ], [1.4*inch, 1.7*inch, 3.65*inch]))
    story.append(Paragraph("Table 15.2 — Memory Architecture Comparison", st["caption"]))
    story.append(PageBreak())

    # ── CHAPTER 16 ────────────────────────────────────────────────────────────
    story += ch("Chapter 16", "Conclusion", st)

    story.append(Paragraph(
        "Aurelius has evolved from a 3-week proof-of-concept into a comprehensive, full-stack AI "
        "research platform spanning 2,155 Python files and ~499,000 lines of code. It covers "
        "the full lifecycle of large language model development: architecture, training, inference "
        "optimization, alignment, safety, serving, agents, memory, retrieval, CI/CD, and frontend.",
        st["body"]))

    story.append(Paragraph(
        "The central architectural contribution — the <b>Aurelian Memory Core</b> — remains the "
        "project's most novel element: a per-layer differentiable three-tier memory hierarchy "
        "that extends effective context without external retrieval API calls. Combined with "
        "an 80+ algorithm alignment stack, 200+ inference optimizations, a self-evolving skill "
        "library, multi-agent coordination, and one of the most complete safety layers in any "
        "independent AI project, Aurelius is a uniquely comprehensive research platform.",
        st["body"]))

    story.append(Paragraph(
        "Security posture is now significantly hardened: all 27 AUR-SEC findings are closed, "
        "sandbox defaults are fail-closed, authentication is fail-closed, SSRF protection is in "
        "place, and CI/CD security scanning runs without continue-on-error. The five merge-ready "
        "branches — especially the security hardening pass and 2.7B config — represent the "
        "immediate priority queue.",
        st["body"]))

    story.append(Spacer(1, 0.18*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY))
    story.append(Spacer(1, 0.1*inch))

    story.append(tbl([
        ["Metric", "Value (May 2026)"],
        ["Python files", "2,155"],
        ["Lines of code", "~499,000"],
        ["Alignment algorithms", "80+"],
        ["Inference modules", "200+"],
        ["Model variants", "5 (150M, 1B, 2.7B dev, 3B, 7B)"],
        ["Security findings closed", "27 (AUR-SEC-2026-0001–0027)"],
        ["Merge-ready branches", "5 of 42 total"],
        ["Memory tiers (AMC)", "3 per transformer layer"],
        ["License", "MIT"],
    ], [2.6*inch, 4.15*inch]))
    story.append(Paragraph("Table 16.1 — Project Summary (May 2026)", st["caption"]))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        f'<font color="#0D1F3C"><b>Prepared by:</b></font> Christien Antonio, Aurelius Research<br/>'
        f'<font color="#555555">Date: {datetime.date.today().strftime("%B %d, %Y")}</font>',
        st["body"]))
    story.append(PageBreak())

    # ── APPENDIX A — REFERENCES ───────────────────────────────────────────────
    story += ch("Appendix A", "References", st)
    for i, ref in enumerate([
        "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.",
        "Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.",
        "Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.",
        "Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
        "Shao, Z. et al. (2024). DeepSeekMath: GRPO. arXiv:2402.03300.",
        "Yu, T. et al. (2025). DAPO: An Open-Source LLM RL System at Scale. arXiv:2503.14476.",
        "Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.",
        "Peng, B. et al. (2025). YaRN: Efficient Context Window Extension of LLMs. arXiv:2309.00071.",
        "Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.",
        "Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention. NeurIPS 2022.",
        "Rajbhandari, S. et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC 2020.",
        "Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.",
        "Liu, Z. et al. (2024). KIVI: Tuning-Free Asymmetric 2-bit Quantization for KV Cache. arXiv:2402.02750.",
        "Xiao, G. et al. (2023). InfLLM: Training-Free Long-Context Extrapolation. arXiv:2402.04617.",
        "Tang, J. et al. (2024). Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. arXiv:2406.10774.",
        "Xiao, G. et al. (2024). DuoAttention: Retrieval and Streaming Heads. arXiv:2410.10819.",
        "Zhang, Z. et al. (2024). TEAL: Training-Free Activation Sparsity in LLMs. arXiv:2410.07303.",
        "Rafailov, R. et al. (2023). Direct Preference Optimization. NeurIPS 2023.",
        "Ethayarajh, K. et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. arXiv:2402.01306.",
        "Hong, J. et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model. arXiv:2403.07691.",
        "Graves, A. et al. (2014). Neural Turing Machines. arXiv:1410.5401.",
    ], 1):
        story.append(Paragraph(f"[{i}]  {ref}", st["body_sm"]))
        story.append(Spacer(1, 0.025*inch))
    story.append(PageBreak())

    # ── APPENDIX B — GLOSSARY ─────────────────────────────────────────────────
    story += ch("Appendix B", "Glossary", st)
    for term, defn in [
        ("AMC", "Aurelian Memory Core — per-layer three-tier differentiable memory architecture."),
        ("BiGRU", "Bidirectional GRU — encodes episodic memory slots in AMC Tier 2."),
        ("DAPO", "Decoupled Clip Policy Optimization — asymmetric clip bounds per advantage sign."),
        ("DPO", "Direct Preference Optimization — preference pairs without a separate reward model."),
        ("DuoAttention", "Streaming heads (sink+recent) + retrieval heads (full KV) per-attention-layer."),
        ("FSDP", "Fully Sharded Data Parallel — shards params, grads, and optimizer state."),
        ("GAE", "Generalized Advantage Estimation — exponential weighted sum for PPO variance reduction."),
        ("GRPO", "Group Relative Policy Optimization — group-relative advantages; no value network."),
        ("GQA", "Grouped-Query Attention — shared KV heads across Q groups; reduces KV cache."),
        ("KIVI", "2-bit asymmetric per-channel KV cache quantization — 87.5% memory reduction."),
        ("KTO", "Kahneman-Tversky Optimization — alignment from binary good/bad feedback."),
        ("LoRA", "Low-Rank Adaptation — parameter-efficient fine-tuning via low-rank weight updates."),
        ("LTS", "Long-Term Store — AMC Tier 3; fixed-capacity content-addressed key-value memory."),
        ("MCTS", "Monte Carlo Tree Search — Aurelius runs this in latent embedding space."),
        ("ORPO", "Odds Ratio Preference Optimization — no reference model needed."),
        ("PPO", "Proximal Policy Optimization — clipped surrogate objective for RLHF."),
        ("Quest", "Page-sparse attention — attend only to top-K KV pages by importance score."),
        ("ReAct", "Reason + Act — agent paradigm: Observe→Think→Act→Reflect (Yao et al., 2022)."),
        ("REBEL", "Regress reward-to-go for LLMs — value-based RL without separate value network."),
        ("RLHF", "Reinforcement Learning from Human Feedback."),
        ("RLVR", "RL from Verifiable Rewards — math/code outcome rewards vs learned reward model."),
        ("RMSNorm", "Root Mean Square Normalization — pre-norm before each sublayer."),
        ("RLOO", "REINFORCE Leave-One-Out — variance reduction baseline."),
        ("RoPE", "Rotary Position Embeddings — encodes position by rotating Q and K."),
        ("S-LoRA", "Scalable LoRA serving — multiple adapters, one base model, LRU GPU management."),
        ("SimPO", "Simple Preference Optimization — reference-free, sequence-average log-probs."),
        ("SPIN", "Self-Play Fine-Tuning — improvement from self-generated negatives."),
        ("SSRF", "Server-Side Request Forgery — server proxied to internal services; now blocked."),
        ("SwiGLU", "Swish-Gated Linear Unit — feed-forward: Swish(xW_gate) ⊙ (xW_up)."),
        ("TEAL", "Training-Free Activation Sparsity — zero activations below magnitude threshold."),
        ("UCB", "Upper Confidence Bound — MCTS node selection balancing exploitation/exploration."),
        ("YaRN", "Yet Another RoPE extensioN — dimension-dependent frequency scaling for context extension."),
        ("ZeRO", "Zero Redundancy Optimizer — eliminates memory redundancy in distributed training."),
    ]:
        story.append(Paragraph(f"<b>{term}</b> — {defn}", st["body_sm"]))
        story.append(Spacer(1, 0.03*inch))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Aurelius dissertation PDF")
    parser.add_argument("--output", type=Path, default=Path("Aurelius_Dissertation.pdf"), help="Output PDF path")
    args = parser.parse_args()
    out = build(args.output)
    print(f"PDF generated: {out}")
