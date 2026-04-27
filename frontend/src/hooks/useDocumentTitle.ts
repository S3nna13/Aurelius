import { useEffect } from 'react'

const BASE_TITLE = 'Aurelius'

export function useDocumentTitle(title?: string): void {
  useEffect(() => {
    document.title = title ? `${title} | ${BASE_TITLE}` : BASE_TITLE
  }, [title])
}
