import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, within, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'
import { fireEvent } from '@testing-library/react'
import { api } from '../../api/AureliusClient'

describe('SettingsPage', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.spyOn(api, 'getConfig').mockResolvedValue({ config: {} } as never)
    vi.spyOn(api, 'setConfig').mockResolvedValue({ success: true, config: {} } as never)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('persists the default backend and upstream settings', async () => {
    const user = userEvent.setup()
    const { default: SettingsPage } = await import('../../pages/Settings')
    render(<SettingsPage />)

    await waitFor(() => expect(api.getConfig).toHaveBeenCalledTimes(1))

    await user.click(screen.getByRole('button', { name: /models/i }))

    const backendLabel = screen.getByText('Default Backend')
    const backendContainer = backendLabel.closest('div')
    expect(backendContainer).not.toBeNull()
    const backendSelect = within(backendContainer as HTMLElement).getByRole('combobox')
    await user.selectOptions(backendSelect, 'agentic')

    await user.click(screen.getByRole('button', { name: /save settings/i }))

    await waitFor(() => expect(api.setConfig).toHaveBeenCalledTimes(1))
    expect(api.setConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        'chat.backend': 'agentic',
        'chat.model': 'aurelius-1.3b',
        'chat.temperature': '0.7',
        'chat.upstream_url': 'http://127.0.0.1:8080',
        'chat.vllm_upstream_url': 'http://127.0.0.1:8080',
        'chat.agentic_upstream_url': 'http://127.0.0.1:8080',
      }),
    )
  })

  it('can select auto as Default Backend and sends chat.backend auto to API', async () => {
    const user = userEvent.setup()
    const { default: SettingsPage } = await import('../../pages/Settings')
    render(<SettingsPage />)

    await waitFor(() => expect(api.getConfig).toHaveBeenCalledTimes(1))

    await user.click(screen.getByRole('button', { name: /models/i }))

    await waitFor(() => expect(screen.queryByText('Default Backend')).toBeInTheDocument())

    const backendLabel = screen.getByText('Default Backend')
    const backendContainer = backendLabel.closest('div')
    expect(backendContainer).not.toBeNull()
    const backendSelect = within(backendContainer as HTMLElement).getByRole('combobox') as HTMLSelectElement

    const autoOption = backendSelect.querySelector('option[value="auto"]') as HTMLOptionElement
    expect(autoOption).not.toBeNull()

    backendSelect.value = 'auto'
    fireEvent.change(backendSelect)

    await user.click(screen.getByRole('button', { name: /save settings/i }))

    await waitFor(() => expect(api.setConfig).toHaveBeenCalledTimes(1))
    expect(api.setConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        'chat.backend': 'auto',
      }),
    )
  })
})
