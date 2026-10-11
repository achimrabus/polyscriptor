<script lang="ts">
	import '../app.css';
	import { setApiKey, getApiKey } from '$lib/api';

	let { children } = $props();
	let dark = $state(true);
	let showKeys = $state(false);
	let keyInput = $state(getApiKey());

	$effect(() => {
		const stored = localStorage.getItem('polyscriptor.theme');
		dark = stored ? stored === 'dark' : !window.matchMedia('(prefers-color-scheme: light)').matches;
	});
	$effect(() => {
		document.documentElement.classList.toggle('dark', dark);
		localStorage.setItem('polyscriptor.theme', dark ? 'dark' : 'light');
	});

	function saveKey() {
		setApiKey(keyInput.trim());
		showKeys = false;
	}
</script>

<div class="flex h-full flex-col">
	<header
		class="flex items-center gap-3 border-b border-zinc-200 bg-white px-4 py-2 dark:border-zinc-800 dark:bg-zinc-900"
	>
		<div class="flex items-baseline gap-2">
			<span class="text-lg font-bold tracking-tight text-sky-600 dark:text-sky-400">Polyscriptor</span>
			<span class="text-xs text-zinc-400">multi-engine HTR</span>
		</div>
		<div class="ml-auto flex items-center gap-2">
			{#if getApiKey()}
				<span class="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">API key set</span>
			{:else}
				<span class="rounded bg-zinc-100 px-2 py-0.5 text-xs text-zinc-500 dark:bg-zinc-800">anonymous</span>
			{/if}
			<button class="btn" onclick={() => (keyInput = getApiKey(), (showKeys = !showKeys))}>API key</button>
			<button class="btn" title="Toggle theme" onclick={() => (dark = !dark)}>
				{dark ? '☀' : '☾'}
			</button>
		</div>
	</header>

	{#if showKeys}
		<div class="border-b border-zinc-200 bg-zinc-50 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900/60">
			<label class="field-label" for="apikey">API key (stored in this browser only)</label>
			<div class="flex gap-2">
				<input id="apikey" class="input max-w-md" type="password" placeholder="psk_…" bind:value={keyInput} />
				<button class="btn btn-primary" onclick={saveKey}>Save</button>
				<button class="btn" onclick={() => (setApiKey(''), (keyInput = ''), (showKeys = false))}>Clear</button>
			</div>
			<p class="mt-1 text-xs text-zinc-500">
				Create keys on the server with <code class="rounded bg-zinc-200 px-1 dark:bg-zinc-800">python -m polyscriptor.web.generate_api_key</code>.
			</p>
		</div>
	{/if}

	<main class="min-h-0 flex-1 overflow-hidden">
		{@render children()}
	</main>
</div>
