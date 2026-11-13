<script lang="ts">
	import { onMount } from 'svelte';
	import type { Attachment } from 'svelte/attachments';
	import { loadWasmModule, type ImagePlacement, type Panorama, type PanoStitcher, type Point, type StitchResult } from '$lib/wasm-loader';

	type PlacedImage = ImagePlacement & { outline: Point[] };

	let stitcher = $state.raw<PanoStitcher | null>(null);
	let greeting = $state('');
	let loadError = $state('');

	let files = $state.raw<File[]>([]);
	// Mirrors the Rust defaults in pano-stitcher/src/lib.rs.
	let focalPerLongSide = $state(0.75);
	let ratio = $state(0.8);
	let maxFeatures = $state(3000);
	let featureMaxDim = $state(1280);
	let cylindricalMatching = $state(true);
	let result = $state.raw<StitchResult | null>(null);
	let running = $state(false);
	let runError = $state('');
	let elapsedMs = $state<number | null>(null);

	const alignment = $derived(result?.alignment ?? null);
	const panorama = $derived(result?.panorama ?? null);

	function drawPanorama(p: Panorama): Attachment<HTMLCanvasElement> {
		return (canvas) => {
			// Resizing clears the canvas, so it must happen before the draw.
			canvas.width = p.width;
			canvas.height = p.height;
			canvas.getContext('2d')?.putImageData(new ImageData(p.pixels, p.width, p.height), 0, 0);
		};
	}

	const placed = $derived(alignment?.images.filter((img): img is PlacedImage => img.outline !== null) ?? []);
	const unplaced = $derived(alignment?.images.filter((img) => img.outline === null) ?? []);
	const movedPairs = $derived(alignment?.pairs.filter((p) => p.cameraMoved).length ?? 0);

	const layout = $derived.by(() => {
		const canvas = alignment?.canvas;
		if (!alignment || !canvas) return null;
		const extent = Math.max(canvas.width, canvas.height);
		const pad = extent * 0.03;
		return {
			canvas,
			// Canvas pixel edges run from -0.5 to size - 0.5.
			viewBox: `${-0.5 - pad} ${-0.5 - pad} ${canvas.width + 2 * pad} ${canvas.height + 2 * pad}`,
			fontSize: extent * 0.04,
			spanDegrees: ((canvas.width / (canvas.scale * alignment.radius)) * 180) / Math.PI,
			megabytes: (canvas.width * canvas.height * 4) / 1e6,
		};
	});

	onMount(() => {
		loadWasmModule()
			.then((module) => {
				stitcher = module;
				greeting = module.hello('demo');
			})
			.catch((error: unknown) => {
				loadError = error instanceof Error ? error.message : String(error);
			});
	});

	function onFilesChange(event: Event & { currentTarget: HTMLInputElement }) {
		// Capture apps name files sequentially, so name order approximates capture order.
		files = [...(event.currentTarget.files ?? [])].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
		result = null;
		runError = '';
		elapsedMs = null;
	}

	async function stitchImages() {
		if (!stitcher || files.length < 2) return;
		running = true;
		runError = '';
		result = null;
		elapsedMs = null;
		try {
			const images = await Promise.all(files.map(async (file) => new Uint8Array(await file.arrayBuffer())));
			// stitch_row blocks the main thread; yield a frame so "Stitching…" paints first.
			await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve)));
			const start = performance.now();
			result = stitcher.stitch_row(images, { focalPerLongSide, ratio, maxFeatures, featureMaxDim, cylindricalMatching });
			elapsedMs = performance.now() - start;
		} catch (error) {
			runError = error instanceof Error ? error.message : String(error);
		} finally {
			running = false;
		}
	}

	function outlineColor(index: number) {
		return `hsl(${(index * 67) % 360} 70% 42%)`;
	}

	function centroid(points: Point[]) {
		return {
			x: points.reduce((sum, [x]) => sum + x, 0) / points.length,
			y: points.reduce((sum, [, y]) => sum + y, 0) / points.length,
		};
	}
</script>

<main class="mx-auto max-w-4xl space-y-8 p-6 font-sans text-neutral-900">
	<header class="space-y-1">
		<h1 class="text-2xl font-semibold">pano-stitcher demo</h1>
		<p class="text-sm text-neutral-600">
			Phase 1 harness: AKAZE features, symmetric ratio-tested matching, ARRSAC rotation fits between adjacent photos, the longest joined run placed on a cylinder, and feather-blended into one
			panorama.
		</p>
	</header>

	<section class="rounded border border-neutral-200 p-4" aria-live="polite">
		<h2 class="mb-2 font-medium">WASM status</h2>
		{#if loadError}
			<p class="text-red-700">Failed to load WASM: {loadError}</p>
			<p class="mt-1 text-sm text-neutral-600">
				Build it with <code>npm run wasm</code>; it should appear in <code>static/wasm/</code>.
			</p>
		{:else if stitcher}
			<p class="font-mono text-green-800">{greeting}</p>
		{:else}
			<p class="text-neutral-600">Loading…</p>
		{/if}
	</section>

	<section class="space-y-4 rounded border border-neutral-200 p-4">
		<h2 class="font-medium">Align a row of images</h2>

		<label class="block space-y-1">
			<span class="text-sm text-neutral-700">Photos in capture order (sorted by file name)</span>
			<input
				type="file"
				accept="image/png,image/jpeg"
				multiple
				onchange={onFilesChange}
				class="block text-sm file:mr-3 file:rounded file:border-0 file:bg-neutral-800 file:px-3 file:py-1.5 file:text-white" />
		</label>

		{#if files.length > 0}
			<ol class="list-inside list-decimal text-sm text-neutral-700">
				{#each files as file, i (i)}
					<li>{file.name}</li>
				{/each}
			</ol>
		{/if}

		<fieldset class="grid grid-cols-2 gap-4 sm:grid-cols-4">
			<legend class="mb-2 text-sm font-medium text-neutral-700">Settings</legend>
			<label class="space-y-1">
				<span class="block text-sm text-neutral-700">Focal (× long side)</span>
				<input type="number" min="0.2" max="2" step="0.05" bind:value={focalPerLongSide} class="block w-28 rounded border border-neutral-300 px-2 py-1 text-sm" />
				<span class="block text-xs text-neutral-500">0.75 main camera, ~0.4 for 0.6× ultra-wide.</span>
			</label>
			<label class="space-y-1">
				<span class="block text-sm text-neutral-700">Lowe ratio</span>
				<input type="number" min="0.5" max="1" step="0.05" bind:value={ratio} class="block w-28 rounded border border-neutral-300 px-2 py-1 text-sm" />
				<span class="block text-xs text-neutral-500">Higher keeps more matches, and more wrong ones.</span>
			</label>
			<label class="space-y-1">
				<span class="block text-sm text-neutral-700">Max keypoints</span>
				<input type="number" min="20" step="500" bind:value={maxFeatures} class="block w-28 rounded border border-neutral-300 px-2 py-1 text-sm" />
				<span class="block text-xs text-neutral-500">Matching time grows with the square.</span>
			</label>
			<label class="space-y-1">
				<span class="block text-sm text-neutral-700">Feature size (px)</span>
				<input type="number" min="160" step="160" bind:value={featureMaxDim} class="block w-28 rounded border border-neutral-300 px-2 py-1 text-sm" />
				<span class="block text-xs text-neutral-500">Long side AKAZE runs at. Never upscales.</span>
			</label>
			<label class="col-span-2 flex items-start gap-2 sm:col-span-4">
				<input type="checkbox" bind:checked={cylindricalMatching} class="mt-1" />
				<span class="text-sm text-neutral-700">
					Unroll onto a cylinder before matching
					<span class="block text-xs text-neutral-500">Turns camera yaw into a sideways shift so edge features look alike. Uses the focal above, so set that first.</span>
				</span>
			</label>
		</fieldset>

		<button type="button" onclick={stitchImages} disabled={!stitcher || files.length < 2 || running} class="rounded bg-neutral-900 px-4 py-2 text-sm text-white disabled:opacity-40">
			{running ? 'Stitching…' : 'Stitch images'}
		</button>
		{#if files.length === 1}
			<p class="text-sm text-neutral-600">Select at least two photos.</p>
		{/if}

		<div aria-live="polite">
			{#if runError}
				<p class="text-red-700">{runError}</p>
			{/if}
			{#if elapsedMs !== null}
				<p class="text-sm text-neutral-600">
					Done in {(elapsedMs / 1000).toFixed(1)} s. Per-stage AKAZE timings are in the console.
				</p>
			{/if}
		</div>
	</section>

	{#if panorama}
		<section class="space-y-3 rounded border border-neutral-200 p-4">
			<h2 class="font-medium">Panorama</h2>
			<canvas
				{@attach drawPanorama(panorama)}
				class="h-auto w-full bg-[repeating-conic-gradient(#e5e5e5_0_25%,#fafafa_0_50%)] bg-[length:16px_16px]"
				aria-label="Stitched panorama, {panorama.width} by {panorama.height} pixels"></canvas>
			<p class="text-sm text-neutral-600">Feather blend, no exposure compensation yet. The checkerboard shows uncovered (transparent) areas.</p>
		</section>
	{/if}

	{#if alignment}
		<section class="space-y-3 rounded border border-neutral-200 p-4">
			<h2 class="font-medium">Output canvas</h2>
			{#if layout}
				<svg
					viewBox={layout.viewBox}
					class="h-auto max-h-[28rem] w-full bg-neutral-50"
					role="img"
					aria-label="Outlines of {placed.length} aligned images on the {layout.canvas.width} by {layout.canvas.height} pixel output canvas">
					<rect x="-0.5" y="-0.5" width={layout.canvas.width} height={layout.canvas.height} fill="white" stroke="#a3a3a3" stroke-width="1" stroke-dasharray="6 4" vector-effect="non-scaling-stroke" />
					{#each placed as img (img.index)}
						{@const c = centroid(img.outline)}
						<polygon
							points={img.outline.map((p) => p.join(',')).join(' ')}
							fill={outlineColor(img.index)}
							fill-opacity="0.08"
							stroke={outlineColor(img.index)}
							stroke-width="2"
							vector-effect="non-scaling-stroke" />
						<text x={c.x} y={c.y} font-size={layout.fontSize} text-anchor="middle" dominant-baseline="middle" fill={outlineColor(img.index)}>
							{img.index + 1}
						</text>
					{/each}
				</svg>
				<p class="text-sm text-neutral-600">
					Canvas: {layout.canvas.width} × {layout.canvas.height} px ({layout.megabytes.toFixed(0)} MB as RGBA) at scale {layout.canvas.scale.toFixed(2)}, spanning {layout.spanDegrees.toFixed(0)}° of
					yaw (radius {Math.round(alignment.radius)} px).
				</p>
			{/if}
			{#if placed.length > 0}
				<p class="text-sm text-neutral-600">
					Placed images {placed[0].index + 1}–{placed[placed.length - 1].index + 1}, centred on image {alignment.reference + 1}.
				</p>
			{/if}
			{#if unplaced.length > 0}
				<ul class="text-sm text-amber-700">
					{#each unplaced as img (img.index)}
						<li>
							Image {img.index + 1} not drawn: {img.unplaced === 'alongAxis'
								? 'the camera points straight up or down, which has no position on the cylinder'
								: 'it is not in the longest run of images joined by accepted pairs'}.
						</li>
					{/each}
				</ul>
			{/if}
		</section>

		<section class="space-y-3 rounded border border-neutral-200 p-4">
			<h2 class="font-medium">Adjacent pairs</h2>
			{#if movedPairs > 0}
				<p class="rounded bg-amber-50 p-3 text-sm text-amber-900">
					The camera seems to have moved between {movedPairs === 1 ? 'one pair' : `${movedPairs} pairs`} of shots, or the focal setting is off. Nearby objects shift differently from far ones when the lens
					moves, so no single rotation lines them up. Stand in one spot and turn the phone around its own lens rather than turning your body.
				</p>
			{/if}
			<table class="w-full text-left text-sm">
				<thead class="border-b border-neutral-300">
					<tr>
						<th class="py-1 pr-4 font-medium">Pair</th>
						<th class="py-1 pr-4 font-medium">Matches</th>
						<th class="py-1 pr-4 font-medium" title="Matches a pure camera rotation explains">Rotation fit</th>
						<th class="py-1 pr-4 font-medium" title="Matches the best homography explains">Plane fit</th>
						<th class="py-1 pr-4 font-medium">Yaw</th>
						<th class="py-1 font-medium">Status</th>
					</tr>
				</thead>
				<tbody>
					{#each alignment.pairs as report (report.from)}
						<tr class="border-b border-neutral-100 align-top">
							<td class="py-2 pr-4">{report.from + 1} → {report.to + 1}</td>
							<td class="py-2 pr-4">{report.matches}</td>
							<td class="py-2 pr-4">{report.inliers}</td>
							<td class="py-2 pr-4">{report.homographyInliers ?? '—'}</td>
							<td class="py-2 pr-4">{report.yawDegrees === null ? '—' : `${report.yawDegrees.toFixed(1)}°`}</td>
							<td class="py-2">
								{#if report.rotation && report.cameraMoved}
									<span class="text-amber-700">Accepted, but the camera likely moved: expect ghosting</span>
								{:else if report.rotation}
									<span class="text-green-800">Accepted</span>
								{:else if report.cameraMoved}
									<span class="text-amber-700">Rejected: the camera moved between these shots</span>
								{:else}
									<span class="text-amber-700">Rejected: too few matches, likely little overlap</span>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}
</main>
