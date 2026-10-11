/** Typed client for the Polyscriptor FastAPI backend. */

const API_KEY_STORAGE = 'polyscriptor.api_key';

export function getApiKey(): string {
	return typeof localStorage !== 'undefined' ? localStorage.getItem(API_KEY_STORAGE) ?? '' : '';
}

export function setApiKey(key: string): void {
	if (key) localStorage.setItem(API_KEY_STORAGE, key);
	else localStorage.removeItem(API_KEY_STORAGE);
}

export class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
	}
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
	const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
	const key = getApiKey();
	if (key) headers['X-API-Key'] = key;
	const res = await fetch(path, { ...init, headers, credentials: 'same-origin' });
	if (!res.ok) {
		let detail = res.statusText;
		try {
			const body = await res.json();
			detail = body.detail ?? JSON.stringify(body);
		} catch {
			/* keep statusText */
		}
		throw new ApiError(res.status, detail);
	}
	if (res.status === 204) return undefined as T;
	return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
	method: 'POST',
	headers: { 'Content-Type': 'application/json' },
	body: JSON.stringify(body)
});

// ---- types -----------------------------------------------------------------

export interface EngineInfo {
	name: string;
	description: string;
	available: boolean;
	unavailable_reason?: string | null;
	requires_line_segmentation: boolean;
	has_config_schema: boolean;
}

export interface SchemaField {
	key: string;
	type: 'select' | 'number' | 'checkbox' | 'text' | 'password';
	label: string;
	options?: { label: string; value: string }[];
	default?: string | number | boolean;
	min?: number;
	max?: number;
	step?: number;
	hint?: string;
	custom_key?: string;
	custom_placeholder?: string;
	key_status?: string;
}

export interface SegResult {
	num_lines: number;
	bboxes: [number, number, number, number][];
	source: string;
	regions?: { id: string; bbox: number[]; num_lines: number }[];
}

export interface LineResult {
	index: number;
	text: string;
	confidence: number | null;
	error?: string | null;
	bbox: number[];
	region: number;
	thinking_text?: string;
}

export interface EngineStatus {
	loaded: boolean;
	engine_name: string | null;
	config: Record<string, unknown>;
	pool_key: string | null;
}

export interface ResultSlot {
	id: string;
	label: string;
	engine: string;
	kind: string;
	lines: LineResult[];
}

// ---- endpoints ---------------------------------------------------------------

export const api = {
	engines: () => request<{ engines: EngineInfo[] } | EngineInfo[]>('/api/engines'),

	engineStatus: () => request<EngineStatus>('/api/engine/status'),

	configSchema: (name: string) =>
		request<{ fields: SchemaField[] }>(`/api/engine/${encodeURIComponent(name)}/config-schema`),

	loadEngine: (engine_name: string, config: Record<string, unknown>) =>
		request<{ success: boolean; message?: string; error?: string }>(
			'/api/engine/load',
			json({ engine_name, config })
		),

	unloadEngine: () => request<{ success: boolean }>('/api/engine/unload', json({})),

	pool: () => request<{ slots: unknown[] }>('/api/engine/pool'),

	gpu: () => request<Record<string, unknown>>('/api/gpu'),

	upload: (file: File) => {
		const fd = new FormData();
		fd.append('file', file);
		return request<{ image_id: string; width: number; height: number; filename: string }>(
			'/api/image/upload',
			{ method: 'POST', body: fd }
		);
	},

	imageUrl: (id: string) => `/api/image/${id}`,

	segment: (id: string, params: Record<string, string>) => {
		const qs = new URLSearchParams(params).toString();
		return request<SegResult>(`/api/image/${id}/segment?${qs}`);
	},

	resegment: (
		id: string,
		body: { rects: number[][]; device?: string; max_columns?: number; split_width_fraction?: number; min_lines_to_split?: number; text_direction?: string; model_path?: string | null }
	) => request<SegResult>(`/api/image/${id}/resegment`, json(body)),

	deleteRegion: (id: string, index: number) =>
		request<SegResult>(`/api/image/${id}/region/${index}`, { method: 'DELETE' }),

	uploadGroundTruth: (id: string, file: File) => {
		const fd = new FormData();
		fd.append('file', file);
		return request<{ success: boolean; line_count: number }>(
			`/api/image/${id}/ground-truth`,
			{ method: 'POST', body: fd }
		);
	},

	compareRun: (image_id: string) =>
		request<{ success: boolean; slot: ResultSlot }>('/api/compare/run', json({ image_id })),

	compareGroundTruth: (image_id: string, slot_id?: string) =>
		request<Record<string, unknown>>('/api/compare/ground-truth', json({ image_id, slot_id })),

	exportXml: async (id: string) => {
		const res = await fetch(`/api/image/${id}/export-xml`, {
			method: 'POST',
			credentials: 'same-origin'
		});
		if (!res.ok) throw new ApiError(res.status, 'export failed');
		return res.blob();
	},

	exportBatch: (image_ids: string[], fmt: 'txt' | 'csv' | 'xml') =>
		requestPromiseBlob(`/api/batch/export-${fmt}`, image_ids),

	setApiKeyLocalOnly: true
};

async function requestPromiseBlob(path: string, image_ids: string[]): Promise<Blob> {
	const res = await fetch(path, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ image_ids }),
		credentials: 'same-origin'
	});
	if (!res.ok) throw new ApiError(res.status, 'export failed');
	return res.blob();
}

// ---- SSE over POST (fetch streaming) ----------------------------------------

export interface SseHandlers {
	onEvent: (event: string, data: any) => void;
	signal?: AbortSignal;
}

export async function transcribeStream(
	body: Record<string, unknown>,
	{ onEvent, signal }: SseHandlers
): Promise<void> {
	const headers: Record<string, string> = { 'Content-Type': 'application/json' };
	const key = getApiKey();
	if (key) headers['X-API-Key'] = key;
	const res = await fetch('/api/transcribe', {
		method: 'POST',
		headers,
		body: JSON.stringify(body),
		signal,
		credentials: 'same-origin'
	});
	if (!res.ok || !res.body) {
		let detail = res.statusText;
		try {
			detail = (await res.json()).detail ?? detail;
		} catch {
			/* ignore */
		}
		throw new ApiError(res.status, detail);
	}

	const reader = res.body.getReader();
	const decoder = new TextDecoder();
	let buf = '';
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		buf += decoder.decode(value, { stream: true });
		let sep: number;
		while ((sep = buf.indexOf('\n\n')) >= 0) {
			const chunk = buf.slice(0, sep);
			buf = buf.slice(sep + 2);
			let event = 'message';
			const dataLines: string[] = [];
			for (const line of chunk.split('\n')) {
				if (line.startsWith('event: ')) event = line.slice(7).trim();
				else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
			}
			if (dataLines.length) {
				try {
					onEvent(event, JSON.parse(dataLines.join('\n')));
				} catch {
					/* skip unparsable frames */
				}
			}
		}
	}
}
