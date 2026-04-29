import { Router, type Request, type Response } from 'express';
import { ProviderRouter } from '../provider_router.js';

const router = Router();
const provider = new ProviderRouter();

interface AgentRoute {
  id: string;
  name: string;
  capabilities: string[];
}

const AGENT_ROUTES: AgentRoute[] = [
  { id: 'coding', name: 'Coding Agent', capabilities: ['code', 'python', 'typescript', 'debug'] },
  { id: 'research', name: 'Research Agent', capabilities: ['search', 'research', 'analyze'] },
  { id: 'general', name: 'General Assistant', capabilities: ['chat', 'general', 'help'] },
];

function routeToAgent(message: string): AgentRoute {
  const lower = message.toLowerCase();
  if (lower.includes('code') || lower.includes('function') || lower.includes('debug') || lower.includes('write')) {
    return AGENT_ROUTES[0];
  }
  if (lower.includes('search') || lower.includes('research') || lower.includes('find') || lower.includes('analyze')) {
    return AGENT_ROUTES[1];
  }
  return AGENT_ROUTES[2];
}

const CONVERSATIONS = new Map<string, { messages: Array<{ role: string; content: string }>; agent: string }>();

// POST /api/chat/completions — unified chat completions (Aurelius → OpenAI fallback)
router.post('/completions', async (req: Request, res: Response) => {
  const { model, messages, max_tokens, temperature, top_p, stream } = req.body;

  try {
    const result = await provider.complete({ model, messages, max_tokens, temperature, top_p, stream: Boolean(stream) });

    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });
      const content = result?.choices?.[0]?.message?.content || '';
      for (let i = 0; i < content.length; i += 5) {
        const chunk = content.slice(i, i + 5);
        res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: chunk }, index: 0 }] })}\n\n`);
        await new Promise((resolve) => setTimeout(resolve, 15));
      }
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }

    res.json(result || { error: 'No provider available' });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to complete request'
    res.status(502).json({
      error: 'Provider unavailable',
      message,
    });
  }
});

// GET /api/chat/models — available models across all providers
router.get('/models', (_req: Request, res: Response) => {
  res.json({ models: provider.getAvailableModels(), stats: provider.getStats() });
});

// POST /api/chat/agent — route message to agent, stream response
router.post('/agent', (req: Request, res: Response) => {
  const { message, conversationId, stream } = req.body;
  if (!message) return res.status(400).json({ error: 'message required' });

  const convId = conversationId || `conv_${Date.now()}`;
  if (!CONVERSATIONS.has(convId)) {
    CONVERSATIONS.set(convId, { messages: [], agent: 'general' });
  }
  const conv = CONVERSATIONS.get(convId)!;
  conv.messages.push({ role: 'user', content: message });

  const agent = routeToAgent(message);
  conv.agent = agent.id;

  if (stream) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });

    res.write(`event: agent\ndata: ${JSON.stringify({ agent: agent.id, name: agent.name })}\n\n`);
    res.write(`event: thought\ndata: ${JSON.stringify({ content: `Routing to ${agent.name}...` })}\n\n`);

    const responseText = `Routed to **${agent.name}** (${agent.capabilities.join(', ')}). This is a streaming response handled by the ${agent.id} agent.`;

    let index = 0;
    const interval = setInterval(() => {
      if (index < responseText.length) {
        const chunk = responseText.slice(index, index + 3);
        res.write(`data: ${JSON.stringify({ content: chunk })}\n\n`);
        index += 3;
      } else {
        clearInterval(interval);
        res.write(`event: done\ndata: {}\n\n`);
        res.end();
      }
    }, 30);

    req.on('close', () => clearInterval(interval));
  } else {
    const responseText = `Routed to **${agent.name}** with capabilities: ${agent.capabilities.join(', ')}.`;
    conv.messages.push({ role: 'assistant', content: responseText });
    res.json({ response: responseText, agent: agent.id, conversationId: convId });
  }
});

// GET /api/chat/conversations
router.get('/conversations', (_req: Request, res: Response) => {
  res.json({
    conversations: Array.from(CONVERSATIONS.entries()).map(([id, conv]) => ({
      id,
      messageCount: conv.messages.length,
      agent: conv.agent,
      lastMessage: conv.messages[conv.messages.length - 1]?.content.slice(0, 100),
    })),
  });
});

export default router;
