# AURELIUS NEURAL BRAIN ARCHITECTURE

## System Overview

The brain is a hybrid neural-symbolic reasoning system that wraps a language model with learned controllers, memory modules, verification networks, and routing mechanisms. The LM serves as the neural compute substrate; the brain provides the cognitive architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTIVE CONTROLLER                     │
│  (meta-learner: decides think/act/retrieve/verify/finalize) │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                     WORKING MEMORY                          │
│           (differentiable scratchpad + slot buffer)         │
└──────┬──────────────┬──────────────┬────────────────┬───────┘
       │              │              │                │
┌──────┴──────┐ ┌─────┴──────┐ ┌────┴──────┐  ┌──────┴──────┐
│  PERCEPTION │ │  REASONING │ │ PLANNING  │  │    TOOL     │
│   ENCODER   │ │    CORE    │ │  NETWORK  │  │ CONTROLLER  │
└─────────────┘ └─────┬──────┘ └───────────┘  └──────┬──────┘
                      │                              │
┌─────────────────────┴──────────────────────────────┴─────────┐
│                     AGENT ROUTER                             │
│  (researcher│coder│mathematician│critic│evaluator│safety│...)│
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────────────────┐
│                   VERIFIER / CRITIC                          │
│           (confidence scoring + error detection)              │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────────────────┐
│                   REFLECTION MODULE                          │
│         (post-task analysis → memory updates)                │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────────────────┐
│                  LONG-TERM MEMORY                            │
│   (factual + procedural + episodic, retrieval-ranked)        │
└──────────────────────────────────────────────────────────────┘
```

---

## MODULE 1: PERCEPTION / INPUT ENCODER

**Purpose:** Convert all incoming signals into a unified neural representation.

**Inputs:** Raw text tokens, tool outputs (JSON/text), memory retrieval results (vectors + text), system state (step count, budget, mode), multimodal extensions (images, audio — future).

**Outputs:** A fixed-width **situation vector** `s ∈ ℝ^d_brain` that concatenates:
- Token embeddings from the LM's encoder (last hidden state or pooled)
- Tool result embeddings (projected via learned linear layer)
- Memory context embeddings (from LTS retrieval)
- System state features (step counter, confidence, mode one-hot)

**Internal Architecture:**
```
class PerceptionEncoder(nn.Module):
    def __init__(self, d_model, d_brain):
        self.text_proj = nn.Linear(d_model, d_brain)
        self.tool_proj = nn.Linear(d_model, d_brain)
        self.mem_proj = nn.Linear(d_model, d_brain)
        self.state_embed = nn.Embedding(16, d_brain)  # 16 system states
        self.fuse = nn.Linear(4 * d_brain, d_brain)
        self.norm = RMSNorm(d_brain)

    def forward(self, text_h, tool_h, mem_h, state_id):
        text = self.text_proj(text_h.mean(dim=1))
        tool = self.tool_proj(tool_h) if tool_h exists else 0
        mem = self.mem_proj(mem_h) if mem_h exists else 0
        state = self.state_embed(state_id)
        return self.norm(self.fuse(torch.cat([text, tool, mem, state])))
```

**Training:** Learned end-to-end via the brain's RL signal. The projection layers receive gradient from downstream reasoning loss.

**Inference:** Single forward pass per input event.

**Failure modes:** Out-of-distribution tool outputs, truncated context. Mitigated by input validation gates (linear layer + sigmoid that zeros bad projections).

**Scaling to 32B:** d_brain scales from 1024 (3B) to 2048 (32B). Projection matrices scale accordingly.

---

## MODULE 2: WORKING MEMORY MODULE

**Purpose:** Maintain active task state across reasoning steps.

**Inputs:** Situation vector `s_t`, previous working memory state `W_{t-1}`, read from LTM.

**Outputs:** Updated working memory `W_t`, attention weights over slots.

**Internal Architecture:**
```
WorkingMemory:
  - Slot buffer: S ∈ ℝ^{n_slots × d_brain}, n_slots = 64
  - Goal register: g ∈ ℝ^{d_brain} (current objective, updated by executive)
  - Step counter: c ∈ ℤ (incremented per reasoning step)
  - Scratchpad: learned gated recurrence
    h_t = GRU(s_t, h_{t-1})
    write_gate = σ(Linear([s_t; h_t; g]))
    erase_gate = σ(Linear([s_t; h_t]))
    S = S * (1 - erase_gate) + write_gate * h_t.unsqueeze(1)
    read = attention(q=query, kv=S) → attended_slot
```

**Slot Content Attention:**
```
For each reasoning step, the executive produces a query q_t = Linear(g, s_t, h_t).
Slots are ranked by attention score. Top-3 slots are read.
The read output is concatenated with h_t to form the working state.
```

**Gating for Capacity Management:**
- A learned **importance predictor** scores each slot: `imp = σ(MLP(S[i]))`
- Low-importance slots are overwritten (threshold = 0.1)
- High-importance slots are protected (threshold > 0.9)
- This is trained via a regularizer: `L_imp = ||imp||_1` (sparsity pressure)

**Training:** Trained jointly with the reasoning core via REINFORCE on task completion reward. The write/erase gates receive gradients through the task loss.

**Inference behavior:** Slots are initialized from the encoded input. Each reasoning step updates them. At verification failure, slots are rolled back to a checkpointed state.

**Failure modes:** Slot overflow (mitigated by importance gating), stale context (mitigated by recency bonus in attention).

**Scaling:** n_slots grows from 64 (3B) → 128 (14B) → 256 (32B). Slot dimension matches d_brain.

---

## MODULE 3: LONG-TERM MEMORY MODULE

**Purpose:** Store and retrieve reusable knowledge across sessions.

**Three Memory Types:**
1. **Factual Memory** — named entities, facts, data points (key: entity embedding, value: fact vector)
2. **Procedural Memory** — successful action sequences, reasoning patterns (key: task embedding, value: trajectory embedding)
3. **Episodic Memory** — past episodes with outcomes (key: situation embedding, value: outcome + reward)

**Internal Architecture:**
```
LongTermMemory:
  FactualStore = VectorDB(d_key=d_brain, d_value=d_brain, capacity=1M)
  ProceduralStore = VectorDB(d_key=d_brain, d_value=d_brain*4, capacity=100K)
  EpisodicStore = VectorDB(d_key=d_brain, d_value=d_brain*2 + scalar, capacity=500K)
  
  Retrieval:
    query = Linear(s_t, g)  # current situation + goal
    factual_results = FactualStore.search(query, top_k=5)
    procedural_results = ProceduralStore.search(query, top_k=3)
    episodic_results = EpisodicStore.search(query, top_k=3)
    
    # Reranking with learned cross-attention
    candidates = [factual, procedural, episodic]
    scores = MLP([query; candidate; task_type_embed])
    top_3 = candidates[argsort(scores)[:3]]
    
  Compression:
    # New memories are compressed via a learned autoencoder
    # encoder: d_brain → d_brain/4 (key)
    # decoder: d_brain/4 → d_brain (value)
    key = encoder(experience)
    value = experience  # store full for retrieval, compressed key for search
    store.insert(key, value)
  
  Consolidation:
    # Background process: cluster similar memories, create prototypes
    # Frequency-weighted: memories accessed >3x get promoted to prototypes
    # Stale memories (unaccessed for N steps) get archived
```

**Memory Update Strategy:**
```
function UPDATE_MEMORY(task, outcome, trajectory, reward):
  # 1. Extract key via encoder(task_embedding)
  # 2. Store in episodic: key → (trajectory, reward)
  # 3. If reward > threshold:
  #    Extract procedural key via encoder(task_type + successful_pattern)
  #    Store in procedural: key → trajectory
  # 4. If new facts discovered:
  #    Store in factual: key → fact_embedding
  # 5. Consolidation trigger: if insert_count % 100 == 0:
  #    Cluster and create prototypes
  #    Archive low-access memories
```

**Training:** 
- Retrieval: trained via contrastive loss (positive = useful memory, negative = random)
- Compression autoencoder: trained via reconstruction loss
- Reranking: trained via binary cross-entropy (was this memory helpful? 0/1)
- Consolidation: unsupervised clustering (k-means on memory keys)

**Inference:** Retrieval happens at the start of each task and whenever the executive controller requests it (uncertainty > threshold).

**Failure modes:** Retrieval of irrelevant memories (mitigated by reranking), memory overload (mitigated by consolidation/archival).

**Scaling:** Capacity scales with parameter count: 100K → 500K → 2M entries. VectorDB uses approximate nearest neighbor (HNSW) for sub-ms retrieval.

---

## MODULE 4: REASONING CORE

**Purpose:** Perform multi-step inference, decomposition, and verification-guided thinking.

**Inputs:** Working memory state `W_t`, retrieved LTM context, task goal `g`.

**Outputs:** Reasoning step tokens, intermediate conclusions, confidence scores.

**Internal Architecture:**
```
ReasoningCore:
  - Uses the base LM as the neural compute engine
  - Wraps LM calls with structured prompting + learned adapters
  
  Decomposer:
    Linear(g, W_t) → subtask_embeddings
    Subtask attention: which subtask to work on next
    
  StepGenerator:
    prompt = format_prompt(g, W_t, subtask_i, ltm_context)
    output_tokens = LM.generate(prompt, max_steps=256, temperature=0.6)
    step_embedding = LM.encode(output_tokens)
    
  SelfConsistency:
    Generate K reasoning paths (K=5)
    Cluster paths by final answer
    Select most common answer
    confidence = cluster_size / K
    
  RecursiveThought:
    If confidence < threshold or step_count < min_steps:
      Generate new reasoning step incorporating: 
        previous_attempt + error_analysis + what_to_try_next
      This is the recursive loop: the LM is prompted with its own prior attempt
```

**Reasoning Loop Algorithm:**
```
function REASON(g, W, ltm, max_depth=10):
  for depth in 1..max_depth:
    subtask = DECOMPOSE(g, W, depth)
    if subtask is None:  # all subtasks complete
      break
    
    step = GENERATE_STEP(g, W, subtask, ltm)
    W = UPDATE_WM(W, step)
    confidence = ESTIMATE_CONFIDENCE(W, step)
    
    if confidence < 0.3:
      ltm_context = RETRIEVE_LTM(query=step.failure_signal)
      step = REGENERATE(g, W, subtask, ltm_context, hint="previous attempt failed")
      W = UPDATE_WM(W, step)
    
    if depth == max_depth:
      summary = SUMMARIZE(W)
      return summary, confidence
      
  return SUMMARIZE(W), confidence
```

**Training:**
- Supervised: on chain-of-thought datasets (log-likelihood of correct reasoning steps)
- RL: via process reward model (PRM) that scores each step for correctness
- Self-consistency reward: higher reward for paths that agree with majority

**Loss Functions:**
```
L_supervised = -log P(correct_step | prompt)  # teacher forcing
L_rl = -R * log P(chosen_action)               # REINFORCE with PRM
L_consistency = KL(π_θ || π_majority)           # distill consensus back to model
L_total = L_supervised + λ_rl * L_rl + λ_cons * L_consistency
```

**Inference behavior:** The reasoning core runs in a loop controlled by the executive. Each iteration produces a step, updates working memory, and returns confidence.

**Failure modes:** Looping (mitigated by max_depth and diversity penalty), hallucinated reasoning (mitigated by verifier), circular reasoning (mitigated by novelty bonus: penalize repeating prior steps).

**Scaling:** K (self-consistency paths) scales from 5 → 8 → 12. Depth scales from 10 → 16 → 24. The LM's reasoning ability scales with its parameter count.

---

## MODULE 5: PLANNING NETWORK

**Purpose:** Convert goals into structured action plans with dependencies.

**Inputs:** Goal `g`, working memory `W`, LTM context.

**Outputs:** Task graph `G = (V, E)` where V = subtasks, E = dependencies.

**Internal Architecture:**
```
PlanningNetwork:
  GoalDecomposer:
    MLP([g; W; ltm]) → K subtask embeddings
    # Uses a learned decomposition head
    subtask_i = softmax(MLP([g; W; ltm])) * subtask_embeddings
    
  DependencyPredictor:
    # Predicts which subtasks depend on which
    dep_ij = σ(MLP([subtask_i; subtask_j]))
    # Threshold at 0.5 to create edges
    
  Scheduler:
    # Topological sort of DAG
    # Assigns each subtask to a time step
    schedule = topological_sort(dep_matrix)
    
  PlanRefiner:
    # Given execution feedback, replan
    if execution_error detected:
      Identify affected subtasks
      Re-decompose from failure point
      Update dependency matrix
```

**Plan Update Algorithm:**
```
function UPDATE_PLAN(plan, feedback, W):
  if feedback == SUCCESS:
    return plan  # continue
  
  failed_node = IDENTIFY_FAILURE_NODE(feedback, W)
  subgraph = EXTRACT_SUBGRAPH(plan, failed_node)
  new_subtasks = DECOMPOSE(failed_node.goal, W, plan.ltm_context)
  plan = REPLACE_SUBGRAPH(plan, failed_node, new_subtasks)
  plan = RESCHEDULE(plan)
  return plan
```

**Training:** Supervised on planning datasets (task decomposition + dependency annotation). RL via plan execution success reward.

**Failure modes:** Overly fine decomposition (mitigated by penalty for >10 subtasks), missing dependencies (mitigated by verification step after plan creation).

---

## MODULE 6: TOOL CONTROLLER

**Purpose:** Decide when and which tools to use, execute tool calls, integrate results.

**Inputs:** Current subtask, working memory, available tool list (name + description embeddings).

**Outputs:** Tool call (name + parameters), or "no tool needed" decision.

**Internal Architecture:**
```
ToolController:
  ToolSelector:
    tool_embeds = [embed(name_i; desc_i) for each tool]
    query = Linear([subtask; W])
    scores = query @ tool_embeds.T
    tool_idx = argmax(scores)
    confidence = softmax(scores)[tool_idx]
    
    if confidence < 0.6:
      return NO_TOOL  # reason instead
    
  ToolExecutor:
    # Generates tool call parameters using LM
    prompt = format_tool_call_prompt(tool_name, subtask, W)
    params = LM.generate(prompt, max_tokens=128)
    
    # Parses structured output
    call = parse_json(params)
    
    # Executes (or returns call for external execution)
    result = execute_tool(tool_name, call)
    
  ResultIntegrator:
    # Converts tool result back into reasoning context
    if result.error:
      retry_count += 1
      if retry_count > 3:
        return TOOL_FAILURE
      return RETRY with modified params
      
    result_embedding = encoder(result)
    W = UPDATE_WM(W, result_embedding)
    return SUCCESS
```

**Tool Selection Algorithm:**
```
function SELECT_TOOL(subtask, W, tools):
  query = PROJ_TOOL_QUERY([subtask.embedding; W.read()])
  
  # Tool matching
  scores = []
  for tool in tools:
    tool_emb = PROJ_TOOL(tool.name_embed, tool.desc_embed)
    score = cosine_similarity(query, tool_emb)
    scores.append(score)
  
  best_idx = argmax(scores)
  best_score = scores[best_idx]
  
  # Decision: use tool or reason directly
  if best_score > 0.7:
    return tools[best_idx]
  elif best_score > 0.4:
    return tools[best_idx] with LOW_CONFIDENCE flag
  else:
    return None  # no tool needed
```

**Training:** 
- Behavioral cloning from tool-use demonstrations
- RL with tool success reward (did the tool produce a useful result?)
- Negative reward for unnecessary tool calls (cost penalty)

**Failure modes:** Tool hallucination (calling nonexistent tools — mitigated by constrained decoding), parameter errors (mitigated by type checking + retry), timeout (mitigated by max_retries).

---

## MODULE 7: AGENT ROUTER

**Purpose:** Route subtasks to specialized sub-agents, merge results.

**Internal Architecture:**
```
AgentRouter:
  AgentRegistry:
    agents = {
      'researcher': ResearchAgent(d_model),
      'coder': CodeAgent(d_model),
      'mathematician': MathAgent(d_model),
      'critic': CriticAgent(d_model),
      'evaluator': EvaluatorAgent(d_model),
      'safety_reviewer': SafetyAgent(d_model),
      'data_engineer': DataAgent(d_model),
      'infrastructure_planner': InfraAgent(d_model),
      'memory_manager': MemoryAgent(d_model),
      'optimizer': OptimizerAgent(d_model),
    }
    # Each agent is a lightweight LoRA-tuned variant of the base LM
    
  Router:
    query = Linear([subtask; W])
    agent_scores = [cosine(query, agent_i.key) for agent_i in agents]
    selected = top_k(agent_scores, k=min(3, needed))
    
  ResultMerger:
    results = [agent.run(subtask) for agent in selected]
    # Weighted merge based on agent confidence
    weights = softmax([r.confidence for r in results])
    merged = sum(w * r.embedding for w, r in zip(weights, results))
    return merged
```

**Agent Routing Algorithm:**
```
function ROUTE(subtask, W):
  task_type = CLASSIFY_TASK_TYPE(subtask)  # research? code? math? etc.
  
  primary_agent = AGENT_FOR_TYPE(task_type)
  support_agents = []
  
  if task_type.requires_verification:
    support_agents.append('critic')
  if task_type.has_safety_implications:
    support_agents.append('safety_reviewer')
  if task_type.involves_data:
    support_agents.append('data_engineer')
  
  primary_result = primary_agent.run(subtask, W)
  
  for agent in support_agents:
    feedback = agent.review(primary_result, subtask)
    if feedback.score < 0.5:
      primary_result = primary_result.revise(feedback)
  
  return primary_result
```

**Training:** Multi-task learning. Each agent is fine-tuned on its domain. The router is trained via behavioral cloning from expert routing decisions.

**Failure modes:** Wrong agent selection (mitigated by top-k fallback), agent disagreement (mitigated by weighted voting), agent hallucination (mitigated by shared verifier).

---

## MODULE 8: VERIFIER / CRITIC MODULE

**Purpose:** Detect errors, score confidence, flag uncertainties.

**Internal Architecture:**
```
VerifierCritic:
  ErrorDetector:
    # Binary classifier: is there an error in this reasoning step?
    features = [step_embedding; W; previous_error_flags]
    error_prob = σ(MLP(features))
    
  ErrorTypeClassifier:
    error_types = ['logical_error', 'hallucination', 'missing_assumption',
                   'contradiction', 'unsafe', 'weak_evidence', 'math_error',
                   'code_error', 'factual_error']
    type_logits = MLP(features)  # multi-label classification
    
  ConfidenceScorer:
    # Calibrated confidence score
    confidence = σ(Linear([step_embedding; W; retrieval_coverage]))
    # Incorporates: self-consistency agreement, retrieval relevance, step fluency
    
  FixGenerator:
    # Generates a suggested fix
    prompt = format_fix_prompt(step, error_type, W)
    fix = LM.generate(prompt, max_tokens=128)
    return fix
```

**Verifier Algorithm:**
```
function VERIFY(step, W, ltm_context):
  features = [step.embed; W.read(); ERROR_FLAGS]
  
  error_prob = ERROR_DETECTOR(features)
  if error_prob < 0.1:
    return PASS, 1.0  # confident pass
  
  error_types = ERROR_TYPE_CLASSIFIER(features)
  confidence = CONFIDENCE_SCORER(features)
  
  if error_prob > 0.7 or confidence < 0.3:
    fix = FIX_GENERATOR(step, error_types, W)
    return FAIL, confidence, error_types, fix
  
  return AMBIGUOUS, confidence, error_types
```

**Training:**
- Error detector: supervised on labeled reasoning errors
- Error type classifier: multi-label classification on annotated error types
- Confidence scorer: trained via expected calibration error (ECE) minimization
- Fix generator: supervised on paired (error, correction) data

**Loss Functions:**
```
L_error = BCE(error_prob, error_label)
L_type = BCE(type_logits, type_labels)  # multi-label
L_confidence = (accuracy - confidence)^2  # calibration loss
L_fix = -log P(correct_fix | error)       # teacher forcing for fixes
L_verifier = L_error + λ_type * L_type + λ_cal * L_confidence + λ_fix * L_fix
```

**Inference:** Run after each reasoning step and before final output.

**Failure modes:** False positives (rejecting correct reasoning — mitigated by adjustable threshold), false negatives (missing errors — mitigated by ensemble of verifiers at 32B scale).

---

## MODULE 9: REFLECTION MODULE

**Purpose:** Post-task analysis to extract lessons and update memory.

**Internal Architecture:**
```
ReflectionModule:
  TrajectoryAnalyzer:
    # Encodes the full task trajectory
    trajectory = [step_1, step_2, ..., step_n, outcome, reward]
    trajectory_embed = BiGRU(trajectory)
    
  SuccessExtractor:
    if reward > threshold:
      key = encoder(task_goal)
      strategy = trajectory_embed[successful_segments]
      memory_manager.store_procedural(key, strategy)
    
  FailureExtractor:
    if reward < threshold:
      error_segments = attention(trajectory_embed, error_signal)
      key = encoder(task_goal + "FAILURE")
      memory_manager.store_episodic(key, error_segments, reward)
    
  StrategyRefiner:
    # For recurring tasks, refine stored strategies
    similar_memories = retrieve_similar(trajectory_embed)
    if similar_memories:
      aggregated_strategy = average(similar_memories)
      memory_manager.store_procedural(key, aggregated_strategy)
```

**Reflection Algorithm:**
```
function REFLECT(task, trajectory, outcome, reward):
  # 1. Analyze what happened
  summary = SUMMARIZE_TRAJECTORY(trajectory)
  key_moments = IDENTIFY_KEY_MOMENTS(trajectory, outcome)
  
  # 2. Extract successes
  if reward > 0.7:
    for segment in key_moments.positive:
      key = ENCODE_TASK_TYPE(task)
      value = ENCODE_STRATEGY(segment)
      procedural_memory.store(key, value, importance=reward)
  
  # 3. Extract failures
  if reward < 0.3:
    for segment in key_moments.negative:
      key = ENCODE_TASK_TYPE(task) + "_FAILURE"
      value = ENCODE_ERROR_PATTERN(segment)
      episodic_memory.store(key, value, importance=1.0 - reward)
    
    # Update strategy to avoid similar failure
    avoidance_strategy = GENERATE_AVOIDANCE_STRATEGY(segment)
    procedural_memory.store(key + "_AVOID", avoidance_strategy)
  
  # 4. Consolidate
  if RANDOM() < 0.1:  # 10% chance per reflection
    CONSOLIDATE_MEMORIES()
  
  return summary
```

**Training:** The reflection module is trained via self-supervised learning on the model's own trajectories. The success/failure extractors are trained via contrastive learning: positive segments vs negative segments should have separable embeddings.

**Failure modes:** Over-generalization (mitigated by specificity penalty in encoder), under-learning (mitigated by forced reflection every N tasks).

---

## MODULE 10: EXECUTIVE CONTROLLER

**Purpose:** Orchestrate all modules. Decide what to do at each step.

**Internal Architecture:**
```
ExecutiveController:
  StateTracker:
    # Maintains system state one-hot
    states = ['encode', 'retrieve', 'decompose', 'reason', 'plan',
              'use_tool', 'verify', 'reflect', 'output', 'wait']
    state = GRU([s_t; W.read(); prev_state])
    state_logits = MLP(state)
    current_state = softmax(state_logits)
    
  ActionSelector:
    actions = ['think_more', 'act', 'retrieve_memory', 'use_tool',
               'ask_clarification', 'verify', 'finalize', 'reflect']
    action_scores = MLP([W; s_t; state; remaining_budget])
    
    # Budget-aware: penalize expensive actions when budget is low
    cost_penalty = action_cost[action] / remaining_budget
    action = argmax(action_scores - cost_penalty)
    
  LoopDetector:
    # Tracks recent actions to detect loops
    recent = deque(last_10_actions)
    if len(set(recent)) < 3:  # too few unique actions
      forced_action = 'finalize'
    if any action repeats > 4 times:
      forced_action = 'reflect'  # break out of loop
  
  TerminationDecider:
    should_stop = (confidence > 0.9 and all_subtasks_done) or \
                  (step_count > max_steps) or \
                  (budget_exhausted)
```

**Executive Controller Algorithm:**
```
function EXECUTE(input, max_steps=50):
  # Initialize
  s = PERCEIVE(input)
  W = WORKING_MEMORY.init(s)
  ltm = RETRIEVE_LTM(s)
  state = 'decompose'
  step = 0
  
  while step < max_steps:
    step += 1
    s = PERCEIVE(input, tool_results, ltm_results)
    
    # Executive decision
    action = SELECT_ACTION(state, W, s, remaining_budget)
    
    if LOOP_DETECTED():
      action = 'reflect'
    
    match action:
      case 'think_more':
        W, confidence = REASON(s, W, ltm)
        state = 'verify'
      
      case 'retrieve_memory':
        ltm = RETRIEVE_LTM(query=W.uncertainties)
        W = UPDATE_WM(W, ltm)
      
      case 'use_tool':
        subtask = W.current_subtask
        tool = SELECT_TOOL(subtask, W, available_tools)
        if tool is not None:
          result = EXECUTE_TOOL(tool, subtask)
          W = UPDATE_WM(W, result)
        else:
          W = UPDATE_WM(W, 'tool not needed')
      
      case 'verify':
        status, confidence, errors, fix = VERIFY(W.last_step, W, ltm)
        if status == FAIL and step < max_steps - 5:
          W = APPLY_FIX(W, fix)
          state = 'reason'
        elif status == PASS:
          confidence = confidence
      
      case 'reflect':
        summary = REFLECT(task, trajectory, outcome, reward)
        UPDATE_MEMORY(task, outcome, trajectory, reward)
      
      case 'finalize':
        output = GENERATE_OUTPUT(W, ltm)
        return output, confidence
    
    # Update loop tracking
    RECORD_ACTION(action)
    
  # Max steps reached — force output
  output = GENERATE_OUTPUT(W, ltm)
  return output, confidence
```

**Training:** The executive controller is trained via RL (PPO) with a reward that combines:
- Task completion (correctness)
- Efficiency (fewer steps = higher reward)
- Cost (fewer tool calls = higher reward)
- User satisfaction (implicit from task success)

**State:** The executive's state machine is learned. It can choose any action at any step, but is regularized toward sensible sequences (e.g., retrieve before reason, verify before finalize).

**Failure modes:** Infinite loops (mitigated by LoopDetector + max_steps), premature termination (mitigated by minimum steps), thrashing between actions (mitigated by action repetition penalty).

---

## DATA FLOW

```
1. INPUT TOKENIZATION
   text input → tokenizer → input_ids

2. LM ENCODING (shared forward pass)
   input_ids → transformer → hidden_states[H]

3. PERCEPTION ENCODING
   hidden_states[-1] + tool_outputs + system_state → situation_vector s

4. LTM RETRIEVAL (parallel with encoding)
   s → query → factual_store.search(s) → factual_context
   s → query → procedural_store.search(s) → procedural_context  
   s → query → episodic_store.search(s) → episodic_context
   rerank(all_contexts) → top_3_memories

5. WORKING MEMORY UPDATE
   [s; ltm_context; prev_W] → GRU → new_W
   write_gate * new_hidden → slot buffer
   attention → read slots

6. EXECUTIVE DECISION
   [new_W; s; state] → MLP → action_logits
   softmax → action

7. REASONING (if action == think_more)
   prompt = [goal; W; subtask; ltm; step_history]
   LM.generate(prompt) → step_tokens
   step_embedding = LM.encode(step_tokens)
   W = UPDATE_WM(W, step_embedding)

8. TOOL USE (if action == use_tool)
   W.current_subtask → tool_selector → tool_name
   LM.generate(tool_call_prompt) → params
   execute_tool(name, params) → result
   W = UPDATE_WM(W, result_embedding)

9. VERIFICATION (after each reasoning step)
   step_embedding → verifier → (error_prob, confidence, error_types)
   if error_prob > threshold: generate_fix → re-reason

10. FINAL OUTPUT (if action == finalize)
    [W; ltm; verified_steps] → LM.generate → output_tokens

11. REFLECTION (after output is accepted/rejected)
    trajectory → analyze → extract successes/failures → update LTM

12. CONTINUAL LEARNING
    (trajectory, outcome) → RL update → brain parameters
```

---

## TRAINING CURRICULUM

### Phase A: Minimal Reasoning Controller
```
Objective: Train the executive to run a simple reason → output loop
Data: 100K CoT examples (question → reasoning steps → answer)
Training: Supervised fine-tuning of LM + executive
Metrics: Task completion rate, average steps
Duration: 5K steps, batch=64
```

### Phase B: Working Memory + Verifier
```
Objective: Train slot buffer management and error detection
Data: 200K examples with corrupted reasoning steps (labeled errors)
Training: 
  - WM: REINFORCE on task completion (reward = correctness - step_penalty)
  - Verifier: BCE on error detection
Metrics: Verifier F1, WM slot utilization
Duration: 10K steps
```

### Phase C: Long-Term Memory
```
Objective: Train retrieval, reranking, and consolidation
Data: 500K multi-task episodes with memory needs
Training:
  - Retriever: contrastive loss (positive = useful memory)
  - Reranker: BCE (was top-1 useful?)
  - Consolidation: unsupervised clustering
Metrics: Hit rate@5, memory helpfulness (reward delta with/without memory)
Duration: 15K steps
```

### Phase D: Tool Controller
```
Objective: Train tool selection and result integration
Data: 100K tool-use demonstrations (task → tool → params → result)
Training:
  - Tool selector: BCE on correct tool
  - Parameter generator: supervised on correct params
  - Result integrator: end-to-end via task completion
Metrics: Tool selection accuracy, parameter accuracy, result utilization rate
Duration: 10K steps
```

### Phase E: Agent Router
```
Objective: Train routing to specialized sub-agents
Data: 200K multi-agent trajectories
Training:
  - Router: behavioral cloning from expert routes
  - Each agent: domain-specific fine-tuning
Metrics: Routing accuracy, agent specialization score, merged answer quality
Duration: 20K steps
```

### Phase F: Neural Reasoning Core
```
Objective: Train deep multi-step reasoning with recursion
Data: 500K complex reasoning problems (math, code, logic, planning)
Training:
  - Supervised: CoT log-likelihood
  - RL: PRM-guided REINFORCE
  - Self-consistency: KL distillation
Metrics: Pass@1, self-consistency agreement, recursive depth utilization
Duration: 30K steps
```

### Phase G: Reflection + Self-Improvement
```
Objective: Train reflection module and memory update strategy
Data: Self-generated trajectories (no external data — online learning)
Training:
  - Reflection: contrastive (success vs failure segments)
  - Memory update: RL (does stored memory improve future performance?)
  - Consolidated: periodic re-training on self-generated data
Metrics: Performance improvement over time, memory hit rate improvement
Duration: Continuous (online)
```

### Phase H: Full Integration with 32B
```
Objective: Scale all components to 32B-class model
Data: Combined dataset from all phases
Training:
  - Full-scale fine-tuning with all modules active
  - Distributed training (FSDP + TP + PP)
  - Mixed precision (BF16 + FP8 gradients)
Metrics: End-to-end task completion, efficiency (steps/task), memory utilization
Duration: 100K steps
```

---

## LOSS FUNCTION STRATEGY

```
L_total = 
  L_main (LM next-token prediction) +
  λ_wm * L_wm (working memory slot utilization sparsity) +
  λ_ver * L_verifier (error detection + confidence calibration) +
  λ_ret * L_retrieval (contrastive memory) +
  λ_tool * L_tool (tool selection + parameter accuracy) +
  λ_route * L_route (agent routing) +
  λ_rl * L_rl (RL from task reward) +
  λ_cons * L_consistency (self-consistency distillation) +
  λ_ref * L_reflection (contrastive trajectory analysis)

Scale factors:
  Phase A: λ_main=1.0, λ_rl=0.1, others=0
  Phase B: λ_wm=0.1, λ_ver=1.0
  Phase C: λ_ret=1.0
  Phase D: λ_tool=1.0
  Phase E: λ_route=1.0
  Phase F: λ_rl=1.0, λ_cons=0.5
  Phase G: λ_ref=1.0, λ_ret=0.3
  Phase H: all λ=1.0 (full system)
```

---

## MEMORY UPDATE STRATEGY (Formal)

```
For each task with trajectory τ = (s_1, a_1, s_2, a_2, ..., s_n, r):

Factual update:
  For each new fact f encountered:
    key_f = Encoder(f.name)
    value_f = Encoder(f.description)
    memory.factual.upsert(key_f, value_f)

Procedural update:
  if r > 0.8:  # successful task
    key_p = Encoder(task_type, goal)
    value_p = Compressor(τ.positive_segments)
    memory.procedural.upsert(key_p, value_p, importance=r)
  
  if r < 0.2:  # failed task
    key_fail = Encoder(task_type, "FAILURE_" + failure_mode)
    value_fail = Compressor(τ.negative_segments)
    memory.episodic.upsert(key_fail, value_fail, importance=1-r)

Memory consolidation (every N=100 inserts):
  For each memory type:
    clusters = KMeans(memory.keys, k=sqrt(len(memory)))
    for cluster in clusters:
      prototype = mean(cluster.members)
      importance = mean(cluster.member.importance)
      memory.prototypes.upsert(cluster.centroid, prototype, importance)
      if len(cluster.members) > 10:
        memory.archive(cluster.members)  # keep only prototype

Memory retrieval:
  query = Encoder(current_situation, current_goal)
  
  # Retrieve from all stores
  factual = memory.factual.search(query, k=5)
  procedural = memory.procedural.search(query, k=3)
  episodic = memory.episodic.search(query, k=3)
  prototypes = memory.prototypes.search(query, k=2)
  
  # Rerank
  candidates = factual + procedural + episodic + prototypes
  scores = [Reranker(query, c) for c in candidates]
  top_k = candidates[argsort(scores)[:5]]
  
  return top_k
```

---

## REASONING LOOP ALGORITHM (Pseudocode)

```
def reasoning_loop(goal, max_depth=16):
    W = working_memory.init()
    ltm = long_term_memory.retrieve(goal)
    s = perceive(goal)
    step = 0
    trajectory = []
    
    while step < max_depth:
        step += 1
        
        # 1. DECOMPOSE current goal
        subtasks = decomposer(s, W, ltm)
        current = subtasks.next_unsolved()
        
        # 2. RETRIEVE relevant context
        if W.needs_context():
            memory = ltm.retrieve(W.uncertainty_query())
            W.update(memory)
        
        # 3. REASON
        prompt = build_reason_prompt(goal, current, W, ltm)
        thought = lm.generate(prompt, temperature=0.6, max_tokens=256)
        thought_embed = lm.encode(thought)
        W.update(thought_embed)
        
        # 4. VERIFY
        status, confidence, errors, fix = verifier(thought, W)
        
        if status == FAIL:
            if fix is not None:
                W.apply_fix(fix)
                continue  # re-reason with fix
            else:
                thought = lm.generate(FAILURE_RECOVERY_PROMPT)
                W.update(thought)
                continue
        
        # 5. TOOL CHECK
        if W.needs_tool():
            tool = tool_controller.select(current, W)
            if tool is not None:
                result = tool_controller.execute(tool, current)
                W.update(result)
        
        # 6. PLAN UPDATE
        if W.new_information():
            plan = planning_network.update(plan, W)
        
        # 7. CHECK COMPLETION
        if verifier.all_subtasks_complete(W):
            break
        
        trajectory.append((step, thought, status, confidence))
    
    # 8. FINALIZE
    final = lm.generate(FINALIZE_PROMPT(W))
    return final, trajectory
```

---

## TOOL USE ALGORITHM (Pseudocode)

```
def tool_use_loop(subtask, W):
    tools = get_available_tools()
    query = encode(subtask, W)
    
    # Score each tool
    scores = {}
    for tool in tools:
        tool_embed = encode(tool.name, tool.description)
        scores[tool] = cosine_similarity(query, tool_embed)
    
    best_tool = max(scores, key=scores.get)
    best_score = scores[best_tool]
    
    if best_score < 0.3:
        return None  # no suitable tool
    
    attempts = 0
    max_retries = 3
    
    while attempts < max_retries:
        attempts += 1
        
        # Generate parameters
        param_prompt = build_tool_param_prompt(best_tool, subtask, W, attempt=attempts)
        params_text = lm.generate(param_prompt, max_tokens=128)
        params = parse_tool_params(params_text)
        
        if params is None:
            continue  # retry with error context
        
        # Execute
        try:
            result = execute_tool(best_tool.name, params)
        except Exception as e:
            W.update(f"Tool error: {e}")
            continue
        
        # Verify result
        if result.is_error:
            W.update(f"Tool returned error: {result.error}")
            continue
        
        # Success
        result_embed = encode(result)
        W.update(result_embed)
        return result
    
    return None  # all retries exhausted
```

---

## AGENT ROUTING ALGORITHM (Pseudocode)

```
def route_subtask(subtask, W, agents):
    # Classify task type
    task_features = encode(subtask)
    task_type = task_classifier(task_features)
    
    # Map task type to agents
    agent_map = {
        'research': ['researcher', 'critic'],
        'coding': ['coder', 'evaluator', 'safety_reviewer'],
        'mathematics': ['mathematician', 'critic'],
        'planning': ['infrastructure_planner', 'optimizer'],
        'data': ['data_engineer', 'evaluator'],
        'memory': ['memory_manager'],
        'general': ['researcher', 'critic', 'evaluator'],
    }
    
    selected_agents = agent_map.get(task_type, agent_map['general'])
    
    # Execute agents
    results = []
    for agent_name in selected_agents:
        agent = agents[agent_name]
        agent_result = agent.execute(subtask, W)
        results.append(agent_result)
    
    # Merge results
    if len(results) == 1:
        return results[0]
    
    # Weighted merge by confidence
    total_confidence = sum(r.confidence for r in results)
    weights = [r.confidence / total_confidence for r in results]
    merged = merge_weighted(results, weights)
    
    # Conflict resolution
    conflicts = detect_conflicts(results)
    if conflicts:
        for conflict in conflicts:
            arbiter = agents['critic']
            resolution = arbiter.resolve(conflict)
            merged = apply_resolution(merged, resolution)
    
    return merged
```

---

## VERIFIER FEEDBACK ALGORITHM (Pseudocode)

```
def verifier_feedback_loop(step, W, ltm, max_revisions=3):
    revision_count = 0
    
    while revision_count < max_revisions:
        revision_count += 1
        
        # Run verifier
        features = extract_features(step, W, ltm)
        error_prob = error_detector(features)
        error_types = error_classifier(features)
        confidence = confidence_scorer(features)
        
        if error_prob < 0.15 and confidence > 0.8:
            return PASS, confidence, step
        
        # Generate feedback
        feedback = {
            'error_prob': error_prob,
            'error_types': error_types,
            'confidence': confidence,
            'specific_issues': extract_issues(step, error_types),
            'suggested_fix': generate_fix(step, error_types, W),
        }
        
        # Apply fix
        W.store(feedback)  # store feedback in working memory
        fix_prompt = build_fix_prompt(step, feedback, W)
        new_step = lm.generate(fix_prompt, temperature=0.4, max_tokens=256)
        step = new_step
        
        # If fix made things worse, revert
        new_features = extract_features(step, W, ltm)
        new_error_prob = error_detector(new_features)
        if new_error_prob > error_prob:
            step = W.rollback()  # revert to pre-fix state
    
    return FAIL, confidence, step
```

---

## CONTINUOUS SELF-IMPROVEMENT LOOP

```
for each task in tasks:
    # 1. EXECUTE
    output, trajectory = brain.execute(task)
    
    # 2. EVALUATE
    reward = evaluate_output(output, task.ground_truth)
    
    # 3. REFLECT
    summary = reflection.analyze(task, trajectory, output, reward)
    
    # 4. UPDATE MEMORY
    memory.update(task, trajectory, output, reward)
    
    # 5. UPDATE BRAIN (RL step)
    if reward > 0:
        brain.learning_rate *= 1.001  # small increase on success
        update_priority = LOW  # correct behavior, small update
    else:
        brain.learning_rate *= 0.999  # small decrease on failure
        update_priority = HIGH  # incorrect behavior, big update
    
    # PPO update for the executive
    advantage = reward - value_net(task_embedding)
    policy_grad = -advantage * log_prob(executive_action)
    value_grad = advantage^2
    brain.update(policy_grad + value_grad)
    
    # 6. VERIFIER UPDATE (online)
    if verifier.error_prob != reward:
        verifier.update(step_embedding, reward > 0.5)  # online error detection
    
    # 7. MEMORY CONSOLIDATION (periodic)
    if task_count % 100 == 0:
        memory.consolidate()
    
    # 8. SKILL ACQUISITION (periodic)
    if task_count % 500 == 0:
        successful_patterns = memory.procedural.get_top_k(10)
        for pattern in successful_patterns:
            skill_library.add_skill(pattern)
    
    # 9. REFLECTION TRIGGER (periodic)
    if task_count % 1000 == 0:
        reflection_summary = deep_reflect(recent_tasks[-1000:])
        memory.store("REFLECTION_SUMMARY", reflection_summary, importance=1.0)
        brain.learning_rate *= 0.95  # anneal learning rate over time
```

---

## PHASED IMPLEMENTATION PLAN

### Phase A: Minimal Reasoning Controller (2 weeks)
```
Files: brain_controller.py, brain_config.yaml
Components: ExecutiveController (simple), PerceptionEncoder (basic), WorkingMemory (no slots)
Capabilities: input → reason → output loop, step counting, basic confidence
Tests: can answer multi-step math problems, can follow instructions
```

### Phase B: Working Memory + Verifier (2 weeks)
```
Files: working_memory.py, verifier_critic.py
Components: Slot-based WM, ErrorDetector, ConfidenceScorer
Capabilities: maintains reasoning state, detects simple errors, rolls back on failure
Tests: solves 2-step reasoning without losing context, detects contradictions
```

### Phase C: Long-Term Memory (3 weeks)
```
Files: long_term_memory.py, memory_manager.py
Components: FactualStore, ProceduralStore, EpisodicStore, Reranker
Capabilities: cross-session memory, retrieval-augmented reasoning
Tests: remembers facts across sessions, retrieves relevant strategies
```

### Phase D: Tool Controller (2 weeks)
```
Files: tool_controller.py, tool_registry.py
Components: ToolSelector, ToolExecutor, ResultIntegrator
Capabilities: decides when to use tools, executes tools, integrates results
Tests: selects correct tool for task, retries on failure, integrates results
```

### Phase E: Agent Router (3 weeks)
```
Files: agent_router.py, agents/*.py
Components: Router, ResearchAgent, CodeAgent, MathAgent, CriticAgent, etc.
Capabilities: routes subtasks to specialized agents, merges results
Tests: routes research to researcher, code to coder, resolves conflicts
```

### Phase F: Neural Reasoning Core (4 weeks)
```
Files: reasoning_core.py, planner.py
Components: Decomposer, SelfConsistency, RecursiveThought, PlanningNetwork
Capabilities: multi-step decomposition, MCTS-style planning, self-consistency
Tests: solves 5-step reasoning problems, creates valid plans with dependencies
```

### Phase G: Reflection + Self-Improvement (3 weeks)
```
Files: reflection.py, continual_learner.py
Components: TrajectoryAnalyzer, SuccessExtractor, FailureExtractor
Capabilities: post-task reflection, memory update, online learning
Tests: improves task completion rate over time, avoids past failures
```

### Phase H: Full 32B Integration (4 weeks)
```
Files: brain_32b.py, config_32b.yaml
Components: All modules scaled to 32B, distributed training setup
Capabilities: full brain on 32B-class model with all features
Tests: end-to-end on complex multi-step tasks with tools and memory
```

---

## RECURSIVE BRAIN UPGRADE LOOP

```
After every evaluation cycle (eval_interval = 1000 tasks):

1. PERFORMANCE ANALYSIS
   current_accuracy = task_completion_rate(recent_1000)
   current_efficiency = avg_steps_per_task(recent_1000)
   current_memory_util = memory_hit_rate(recent_1000)
   error_distribution = error_types(recent_1000)
   bottleneck = IDENTIFY_WEAKEST_MODULE(accuracy, efficiency, error_dist)

2. TARGETED UPGRADE
   match bottleneck:
     case 'reasoning':
       UPGRADE_REASONING_CORE()  # add self-consistency, increase depth
     case 'memory':
       UPGRADE_MEMORY()  # increase capacity, improve reranker
     case 'verification':
       UPGRADE_VERIFIER()  # more training data, add error types
     case 'planning':
       UPGRADE_PLANNER()  # deeper search, better decomposition
     case 'tool_use':
       UPGRADE_TOOLS()  # add tools, improve selection
     case 'routing':
       UPGRADE_ROUTER()  # add agents, improve routing
     case 'reflection':
       UPGRADE_REFLECTION()  # deeper analysis, better memory updates
     case 'executive':
       UPGRADE_EXECUTIVE()  # better state machine, more actions

3. UPGRADE REASONING CORE (most common bottleneck)
   def UPGRADE_REASONING_CORE():
     current_depth = brain.config.max_reasoning_depth
     current_k = brain.config.self_consistency_k
     
     if current_depth < 24:
       brain.config.max_reasoning_depth += 2
     if current_k < 12:
       brain.config.self_consistency_k += 1
     
     # Add recursive thought if not present
     if not hasattr(brain, 'recursive_thought'):
       brain.add_module('recursive_thought', RecursiveThought())
     
     # Fine-tune on hard examples
     hard_examples = FILTER_HARD_EXAMPLES(recent_1000)
     brain.fine_tune(hard_examples, epochs=1)

4. UPGRADE MEMORY
   def UPGRADE_MEMORY():
     current_capacity = brain.memory.capacity
     brain.memory.capacity *= 1.5
     
     # Improve reranker
     if current_capacity > 100000:
       brain.memory.reranker = DeeperReranker(d_model * 2)
       TRAIN_RERANKER(brain.memory.reranker, recent_1000)
     
     # Add new memory type if needed
     if error_distribution.memory_staleness > 0.2:
       brain.memory.add_type('working_prototypes')

5. UPGRADE VERIFIER
   def UPGRADE_VERIFIER():
     # Add new error types from distribution
     new_types = GET_NEW_ERROR_TYPES(error_distribution)
     brain.verifier.add_error_types(new_types)
     
     # Improve calibration
     brain.verifier.confidence_scorer = CalibratedScorer()
     CALIBRATE(brain.verifier, recent_1000)
     
     # Ensemble: add second verifier for cross-validation
     if not hasattr(brain, 'verifier_ensemble'):
       brain.verifier_ensemble = VerifierEnsemble()
       brain.verifier_ensemble.add_verifier(brain.verifier)
       brain.verifier_ensemble.add_verifier(VerifierCritic(d_model))

6. MEASURE IMPROVEMENT
   new_accuracy = task_completion_rate(next_100)
   improvement = new_accuracy - current_accuracy
   
   if improvement > 0.05:
     COMMIT_UPGRADE()  # keep changes
     LOG("Brain upgraded: +{improvement:.1%} accuracy")
   elif improvement > 0:
     COMMIT_UPGRADE()
     LOG("Brain upgraded: +{improvement:.1%} accuracy (minor)")
   else:
     ROLLBACK_UPGRADE()  # revert changes
     LOG("Brain upgrade rolled back: {improvement:.1%} accuracy change")
     TRY_DIFFERENT_UPGRADE()

7. SCALE (if performance plateaus)
   if improvement < 0.01 for 5 consecutive cycles:
     NEXT_TIER_SCALE()  # e.g., 3B → 7B → 14B → 32B
     # After scaling:
     brain.config.brain_dim *= 1.5
     brain.config.n_slots *= 1.5
     brain.config.memory_capacity *= 2
     # Re-train for 1000 steps to adapt to new capacity
     brain.adapt_to_new_scale(training_data, steps=1000)

8. REPEAT
   LOG_CYCLE()
   RETURN TO STEP 1
```

---

## SUMMARY

The Aurelius Neural Brain is a complete cognitive architecture consisting of 10 interconnected neural modules that wrap a language model. It implements:

- **Working memory** with differentiable slot management and importance gating
- **Long-term memory** with three specialized stores and learned reranking
- **Multi-step reasoning** with decomposition, self-consistency, and recursive thought
- **Planning** with dependency graphs and dynamic replanning
- **Tool use** with learned selection and automatic retry
- **Agent routing** with task-type classification and weighted merging
- **Verification** with error detection, confidence calibration, and automatic fixing
- **Reflection** with trajectory analysis and memory consolidation
- **Executive control** with budget-aware action selection and loop detection
- **Continuous self-improvement** through online RL and recursive upgrade cycles

The system is designed to be implemented in 8 phases, trained through a curriculum of 2M+ examples, and scaled from 3B to 32B parameters with proportional scaling of all components.
