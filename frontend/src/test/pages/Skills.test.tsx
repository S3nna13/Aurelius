import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('SkillsRegistry', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders without crashing', async () => {
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
  })

  it('displays total skills count', async () => {
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    // The span contains "34 total" since there are 34 skills in SKILL_DATA
    expect(screen.getByText(/34 total/)).toBeInTheDocument()
  })

  it('has search input', async () => {
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    const searchInput = screen.getByPlaceholderText(/search skills/i)
    expect(searchInput).toBeInTheDocument()
  })

  it('has category filter dropdown', async () => {
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    const select = screen.getByRole('combobox')
    expect(select).toBeInTheDocument()
    expect(select).toHaveValue('all')
  })

  it('displays skill cards', async () => {
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    expect(screen.getByText('Code Generation')).toBeInTheDocument()
    expect(screen.getByText('Code Review')).toBeInTheDocument()
    expect(screen.getByText('Web Search')).toBeInTheDocument()
  })

  it('filters skills by category when category button is clicked', async () => {
    const user = userEvent.setup()
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    
    const codingButton = screen.getByRole('button', { name: /coding/i })
    await user.click(codingButton)
    
    await waitFor(() => expect(screen.getByText('Code Generation')).toBeInTheDocument(), { timeout: 5000 })
    expect(screen.queryByText('Web Search')).not.toBeInTheDocument()
  })

  it('filters skills by search query', async () => {
    const user = userEvent.setup()
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    
    const searchInput = screen.getByPlaceholderText(/search skills/i)
    await user.type(searchInput, 'code')
    
    await waitFor(() => expect(screen.getByText('Code Generation')).toBeInTheDocument(), { timeout: 5000 })
    expect(screen.getByText('Code Review')).toBeInTheDocument()
  })

  it('shows category buttons with skills count', async () => {
    const { default: SkillsRegistry } = await import('../../pages/Skills')
    render(<SkillsRegistry />)
    await waitFor(() => expect(screen.getByText('Skills Registry')).toBeInTheDocument(), { timeout: 5000 })
    const codingButton = screen.getByRole('button', { name: /coding/i })
    expect(within(codingButton).getByText(/6 skills/)).toBeInTheDocument()
  })
})
