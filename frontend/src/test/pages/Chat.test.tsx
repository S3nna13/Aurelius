import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, within, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('Chat', () => {
  const requests: Array<{ url: string; [key: string]: unknown }> = []

  beforeEach(() => {
    localStorage.clear()
    requests.length = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url
        const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
        requests.push({ url, ...body })

        if (url.includes('/api/chat/completions')) {
          return new Response(
            JSON.stringify({
              id: 'chatcmpl-test',
              object: 'chat.completion',
              created: 1714300000,
              model: 'aurelius-1.3b',
              choices: [
                {
                  index: 0,
                  message: { role: 'assistant', content: 'Model reply' },
                  finish_reason: 'stop',
                },
              ],
              usage: {
                prompt_tokens: 8,
                completion_tokens: 4,
                total_tokens: 12,
              },
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            },
          )
        }

        if (url.includes('/api/chat/agent')) {
          return new Response(
            'data: {"content":"Agent reply","agent":"coding"}\ndata: [DONE]',
            {
              status: 200,
              headers: { 'Content-Type': 'text/event-stream' },
            },
          )
        }

        return new Response('', { status: 404 })
      }),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('switches between agent and model modes', async () => {
    const user = userEvent.setup()
    const { default: Chat } = await import('../../pages/Chat')
    render(<Chat />)

    const agentBtn = screen.getByRole('button', { name: /^agent$/i })
    const modelBtn = screen.getByRole('button', { name: /^model$/i })

    await user.click(modelBtn)
    expect(modelBtn).toHaveClass(/bg-\[#4fc3f7\]\/10/)
    expect(screen.getByText('Model Chat')).toBeInTheDocument()

    await user.click(agentBtn)
    expect(agentBtn).toHaveClass(/bg-\[#4fc3f7\]\/10/)
    expect(screen.getByText('Agent Chat')).toBeInTheDocument()
  })

  it('model mode sends backend to /api/chat/completions', async () => {
    const user = userEvent.setup()
    const { default: Chat } = await import('../../pages/Chat')
    render(<Chat />)

    await user.click(screen.getByRole('button', { name: /^model$/i }))

    const backendSelect = screen.getByRole('combobox')
    await user.selectOptions(backendSelect, 'vllm')

    const input = screen.getByPlaceholderText('Chat directly with the model...')
    await user.type(input, 'hello model mode')
    await user.click(screen.getByRole('button', { name: '' }))

    await waitFor(() => expect(requests.some((r) => r.url.includes('/api/chat/completions'))).toBe(true))
    const completionsReq = requests.find((r) => r.url.includes('/api/chat/completions'))
    expect(completionsReq?.backend).toBe('vllm')
  })

  it('agent mode keeps using /api/chat/agent', async () => {
    const user = userEvent.setup()
    const { default: Chat } = await import('../../pages/Chat')
    render(<Chat />)

    await user.click(screen.getByRole('button', { name: /^agent$/i }))

    const input = screen.getByPlaceholderText(/ask the agent/i)
    await user.type(input, 'hello agent mode')
    await user.click(screen.getByRole('button', { name: '' }))

    await waitFor(() => expect(requests.some((r) => r.url.includes('/api/chat/agent'))).toBe(true))
    const agentReq = requests.find((r) => r.url.includes('/api/chat/agent'))
    expect(agentReq?.url).toContain('/api/chat/agent')
  })

  it('backend selection in model mode persists', async () => {
    const user = userEvent.setup()
    const { default: Chat } = await import('../../pages/Chat')
    const { unmount } = render(<Chat />)

    await user.click(screen.getByRole('button', { name: /^model$/i }))

    const backendSelect = screen.getByRole('combobox')
    await user.selectOptions(backendSelect, 'agentic')

    unmount()

    const { default: Chat2 } = await import('../../pages/Chat')
    render(<Chat2 />)

    await user.click(screen.getByRole('button', { name: /^model$/i }))

    const select = screen.getByRole('combobox')
    expect((select as HTMLSelectElement).value).toBe('agentic')
  })

  it('auto omits the backend field from the request', async () => {
    const user = userEvent.setup()
    const { default: Chat } = await import('../../pages/Chat')
    render(<Chat />)

    await user.click(screen.getByRole('button', { name: /^model$/i }))

    const input = screen.getByPlaceholderText('Chat directly with the model...')
    await user.type(input, 'hello auto backend')
    await user.click(screen.getByRole('button', { name: '' }))

    await waitFor(() => expect(requests.some((r) => r.url.includes('/api/chat/completions'))).toBe(true))
    const completionsReq = requests.find((r) => r.url.includes('/api/chat/completions'))
    expect(completionsReq?.backend).toBeUndefined()
  })
})
