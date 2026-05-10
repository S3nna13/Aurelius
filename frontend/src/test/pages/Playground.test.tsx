import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, within, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('Playground', () => {
  const requests: Array<Record<string, unknown>> = []

  beforeEach(() => {
    localStorage.clear()
    requests.length = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input, init) => {
        const url = String(input)
        if (url.includes('/api/config')) {
          return new Response(JSON.stringify({
            config: {
              'chat.model': 'aurelius-2.7b',
              'chat.temperature': '0.42',
              'chat.backend': 'agentic',
            },
          }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }

        const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
        requests.push(body)

        return new Response(
          JSON.stringify({
            id: 'chatcmpl-test',
            object: 'chat.completion',
            created: 1714300000,
            model: String(body.model ?? 'aurelius-1.3b'),
            choices: [
              {
                index: 0,
                message: { role: 'assistant', content: 'Aurelius live reply' },
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
      }),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('sends the selected backend to the completions endpoint', async () => {
    const user = userEvent.setup()
    const { default: Playground } = await import('../../pages/Playground')
    render(<Playground />)
    await screen.findByText(/Auto → agentic/i)

    const prompt = screen.getByPlaceholderText('Enter your prompt here...')
    await user.type(prompt, 'hello backend')

    const backendRow = screen.getByText('Backend').closest('div')
    expect(backendRow).not.toBeNull()
    const backendSelect = within(backendRow as HTMLElement).getByRole('combobox')
    await user.selectOptions(backendSelect, 'agentic')

    const streamRow = screen.getByText('Stream output').closest('div')
    expect(streamRow).not.toBeNull()
    const streamToggle = within(streamRow as HTMLElement).getByRole('checkbox')
    await user.click(streamToggle)

    await user.click(screen.getByRole('button', { name: /run/i }))

    await waitFor(() => expect(requests).toHaveLength(1))
    expect(requests[0].backend).toBe('agentic')
    expect(requests[0].model).toBe('aurelius-2.7b')
    expect(requests[0].temperature).toBe(0.42)
    expect(requests[0].stream).toBe(false)
    expect(requests[0].messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ role: 'user', content: 'hello backend' }),
      ]),
    )
    expect(await screen.findByText('Aurelius live reply')).toBeInTheDocument()
  })

  it('omits the backend when auto mode is selected', async () => {
    const user = userEvent.setup()
    const { default: Playground } = await import('../../pages/Playground')
    render(<Playground />)
    await screen.findByText(/Auto → agentic/i)

    const prompt = screen.getByPlaceholderText('Enter your prompt here...')
    await user.type(prompt, 'hello auto backend')

    const streamRow = screen.getByText('Stream output').closest('div')
    expect(streamRow).not.toBeNull()
    const streamToggle = within(streamRow as HTMLElement).getByRole('checkbox')
    await user.click(streamToggle)

    await user.click(screen.getByRole('button', { name: /run/i }))

    await waitFor(() => expect(requests).toHaveLength(1))
    expect(requests[0].backend).toBeUndefined()
    expect(requests[0].model).toBe('aurelius-2.7b')
    expect(requests[0].temperature).toBe(0.42)
    expect(requests[0].messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ role: 'user', content: 'hello auto backend' }),
      ]),
    )
    expect(await screen.findByText('Aurelius live reply')).toBeInTheDocument()
  })

  it('shows the backend badge in the header', async () => {
    const { default: Playground } = await import('../../pages/Playground')
    render(<Playground />)

    const badge = await screen.findByText(/Auto → agentic/i)
    expect(badge).toBeInTheDocument()
  })

  it('backend badge reflects when backend is changed to vllm', async () => {
    const user = userEvent.setup()
    const { default: Playground } = await import('../../pages/Playground')
    render(<Playground />)

    const backendRow = screen.getByText('Backend').closest('div')
    expect(backendRow).not.toBeNull()
    const backendSelect = within(backendRow as HTMLElement).getByRole('combobox')
    await user.selectOptions(backendSelect, 'vllm')

    const badge = screen.getAllByText((content, element) => {
      if (!element) return false
      const text = element.textContent || ''
      return text.includes('vllm') && element.tagName === 'SPAN'
    })[0]
    expect(badge).toBeInTheDocument()
  })
})
