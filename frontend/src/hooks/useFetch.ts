import { useEffect, useState } from 'react'

export interface FetchState<T> {
  data: T | null
  loading: boolean
  error: Error | null
  retry: () => void
}

export function useFetch<T>(fetcher: () => Promise<T>, deps: readonly unknown[]): FetchState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetcher()
      .then((result) => {
        if (!cancelled) {
          setData(result)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setData(null)
          setError(err instanceof Error ? err : new Error(String(err)))
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
    // `fetcher` is intentionally excluded: callers pass a fresh closure each render
    // (e.g. `() => api.getSeries(id)`), and the deps array below is the real trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt])

  return { data, loading, error, retry: () => setAttempt((a) => a + 1) }
}
