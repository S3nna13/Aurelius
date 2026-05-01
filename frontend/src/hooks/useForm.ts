import { useState, useCallback, type FormEvent } from 'react'

type ValidationRules<T> = Partial<Record<keyof T, (value: T[keyof T], values: T) => string | null>>

interface UseFormOptions<T extends Record<string, unknown>> {
  initialValues: T
  validation?: ValidationRules<T>
  onSubmit: (values: T) => Promise<void> | void
}

interface UseFormReturn<T> {
  values: T
  errors: Partial<Record<keyof T, string>>
  touched: Partial<Record<keyof T, boolean>>
  submitting: boolean
  setValue: (key: keyof T, value: T[keyof T]) => void
  setValues: (values: Partial<T>) => void
  setTouched: (key: keyof T) => void
  handleSubmit: (e: FormEvent) => Promise<void>
  reset: (values?: T) => void
  validate: () => boolean
}

export function useForm<T extends Record<string, unknown>>(options: UseFormOptions<T>): UseFormReturn<T> {
  const { initialValues, validation = {} as ValidationRules<T>, onSubmit } = options
  const [values, setValuesState] = useState<T>(initialValues)
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({})
  const [touched, setTouchedState] = useState<Partial<Record<keyof T, boolean>>>({})
  const [submitting, setSubmitting] = useState(false)

  const setTouched = useCallback((key: keyof T) => {
    setTouchedState((prev) => ({ ...prev, [key]: true }))
  }, [])

  const validateField = useCallback((key: keyof T, val: T[keyof T], allValues: T) => {
    const rule = validation[key]
    if (rule) return rule(val, allValues)
    return null
  }, [validation])

  const validate = useCallback((): boolean => {
    const newErrors: Partial<Record<keyof T, string>> = {}
    let valid = true
    for (const key of Object.keys(values) as Array<keyof T>) {
      const error = validateField(key, values[key], values)
      if (error) {
        newErrors[key] = error
        valid = false
      }
    }
    setErrors(newErrors)
    return valid
  }, [values, validateField])

  const setValue = useCallback((key: keyof T, value: T[keyof T]) => {
    setValuesState((prev) => {
      const next = { ...prev, [key]: value }
      const error = validateField(key, value, next)
      setErrors((prevErrors) => ({ ...prevErrors, [key]: error || undefined }))
      return next
    })
  }, [validateField])

  const setValues = useCallback((patch: Partial<T>) => {
    setValuesState((prev) => ({ ...prev, ...patch }))
  }, [])

  const handleSubmit = useCallback(async (e: FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    try {
      await onSubmit(values)
    } finally {
      setSubmitting(false)
    }
  }, [validate, onSubmit, values])

  const reset = useCallback((vals?: T) => {
    setValuesState(vals || initialValues)
    setErrors({} as Partial<Record<keyof T, string>>)
    setTouchedState({} as Partial<Record<keyof T, boolean>>)
  }, [initialValues])

  return { values, errors, touched, submitting, setValue, setValues, setTouched, handleSubmit, reset, validate }
}
