// 127.0.0.1 (not "localhost"): on dual-stack Windows setups another WSL2-relayed
// service can own localhost's IPv6 loopback; the IPv4 loopback is unambiguous.
const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export interface MetricSeries {
  id: number
  name: string
  unit: string | null
  description: string | null
  source: string
  tags: Record<string, string> | null
  created_at: string
}

export interface MetricSeriesDetail extends MetricSeries {
  point_count: number
  first_ts: string | null
  last_ts: string | null
}

export interface Point {
  ts: string
  value: number
  source_label: boolean | null
}

export interface PointsPage {
  series_id: number
  start: string | null
  end: string | null
  max_points: number
  downsampled: boolean
  bucket_seconds: number | null
  count: number
  points: Point[]
}

export interface StoredForecast {
  id: number
  series_id: number
  created_at: string
  horizon_minutes: number
  target_ts: string
  quantiles: Record<string, number>
  model_name: string
  model_version: number
}

export interface StoredForecastsPage {
  series_id: number
  count: number
  forecasts: StoredForecast[]
}

export interface AnomalyResult {
  id: number
  series_id: number
  ts: string
  score: number
  is_anomaly: boolean
  threshold: number | null
  model_name: string
  model_version: number
  created_at: string
}

export interface AnomalyResultsPage {
  series_id: number
  count: number
  results: AnomalyResult[]
}

export interface ModelVersionInfo {
  version: number
  status: string
  run_id: string | null
  aliases: string[]
}

export interface RegisteredModelInfo {
  name: string
  versions: ModelVersionInfo[]
}

export interface ChampionInfo {
  task: string
  model_name: string
  model_version: number
  promoted_at: string | null
  metrics: Record<string, unknown> | null
  loaded: boolean
}

export interface DriftEvent {
  id: number
  series_id: number
  detected_at: string
  kind: string
  status: string
  feature: string | null
  psi_value: number | null
  threshold: number | null
  details: Record<string, unknown>
}

export interface DriftEventPage {
  count: number
  events: DriftEvent[]
}

export interface DriftStatusEntry {
  series_id: number
  series_name: string
  psi: { status: string; detected_at: string; feature: string | null; psi_value: number | null; details: Record<string, unknown> } | null
  residual: { status: string; detected_at: string; details: Record<string, unknown> } | null
  overall: string
}

export interface PromotionDecision {
  id: number
  decided_at: string
  task: string
  champion_name: string
  champion_version: number
  challenger_name: string
  challenger_version: number | null
  decision: string
  reason: string
  metrics: Record<string, unknown>
}

export interface Alert {
  id: number
  series_id: number
  created_at: string
  kind: string
  severity: string
  message: string
  payload: Record<string, unknown>
  acknowledged: boolean
}

export interface AlertsPage {
  count: number
  unacknowledged: number
  alerts: Alert[]
}

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, options)
  } catch {
    throw new ApiError(`Cannot reach the PulseGuard API at ${API_BASE}`, 0)
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // keep the status text when the error body is not the standard shape
    }
    throw new ApiError(`API error ${res.status}: ${detail}`, res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  listSeries: () => request<MetricSeries[]>('/api/v1/series'),
  getSeries: (id: number) => request<MetricSeriesDetail>(`/api/v1/series/${id}`),
  getPoints: (id: number, maxPoints = 2000) =>
    request<PointsPage>(`/api/v1/series/${id}/points?max_points=${maxPoints}`),
  getStoredForecasts: (id: number, limit = 200) =>
    request<StoredForecastsPage>(`/api/v1/series/${id}/forecasts?limit=${limit}`),
  getStoredAnomalies: (id: number, limit = 500) =>
    request<AnomalyResultsPage>(`/api/v1/series/${id}/anomalies?limit=${limit}`),
  listRegisteredModels: () => request<RegisteredModelInfo[]>('/api/v1/models'),
  getChampion: (task: 'forecast' | 'anomaly') =>
    request<ChampionInfo>(`/api/v1/models/${task}/champion`),
  getDriftStatus: () => request<{ series: DriftStatusEntry[] }>('/api/v1/drift/status'),
  getDriftEvents: (limit = 20) => request<DriftEventPage>(`/api/v1/drift/events?limit=${limit}`),
  getPromotions: (limit = 25) => request<{ count: number; decisions: PromotionDecision[] }>(`/api/v1/promotions?limit=${limit}`),
  getAlerts: (limit = 200) => request<AlertsPage>(`/api/v1/alerts?limit=${limit}`),
  ackAlert: (id: number) =>
    request<{ id: number; acknowledged: boolean }>(`/api/v1/alerts/${id}/ack`, {
      method: 'POST',
    }),
}
