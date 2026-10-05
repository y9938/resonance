<script lang="ts">
  import { onMount } from "svelte";

  let { content }: { content: HTMLElement } = $props();
  let layer: HTMLDivElement;
  let paused = $state(false);

  // Stable decoration, independent of application state and viewport updates.
  const fraction = (value: number) => value - Math.floor(value);
  const stars = Array.from({ length: 18 }, (_, index) => {
    const seed = fraction(Math.sin(index * 127.1 + 43.7) * 43758.5453);
    return {
      x: 5 + 76 * seed ** 2,
      y: 6 + 88 * fraction(index * 0.61803398875),
      size: index % 6 === 0 ? 10 : 3 + seed * 3,
      opacity: 0.4 + seed * 0.35,
      dx: -10 + seed * 20,
      dy: -14 + fraction(index * 0.37) * 28,
      duration: 5 + seed * 4,
      delay: -index * 2.7,
      scale: 0.85 + seed * 0.27,
      shape: index % 6 === 0 ? "spark" : index % 4 === 0 ? "diamond" : "dot",
    };
  });

  onMount(() => {
    let frame = 0;
    const syncVisibility = () => { paused = document.hidden; };
    syncVisibility();

    const measure = () => {
      frame = 0;
      const bounds = content.getBoundingClientRect();
      const width = document.documentElement.clientWidth;
      const top = Math.max(0, bounds.top);
      const height = Math.max(0, Math.min(window.innerHeight, bounds.bottom) - top);
      // Leave a quiet buffer outside the actual main box, including its padding.
      const left = Math.max(0, bounds.left - 24);
      const right = Math.max(0, width - bounds.right - 24);
      layer.style.setProperty("--top", `${top}px`);
      layer.style.setProperty("--height", `${height}px`);
      layer.style.setProperty("--left", `${left}px`);
      layer.style.setProperty("--right", `${right}px`);
      layer.hidden = Math.min(left, right) < 96 || height < 120;
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(content);
    observer.observe(document.body);
    window.addEventListener("resize", schedule);
    window.addEventListener("scroll", schedule, { passive: true });
    document.addEventListener("visibilitychange", syncVisibility);
    measure();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", schedule);
      window.removeEventListener("scroll", schedule);
      document.removeEventListener("visibilitychange", syncVisibility);
    };
  });
</script>

<div class="star-field" class:paused bind:this={layer} aria-hidden="true" hidden>
  {#each ["left", "right"] as side}
    <div class="gutter {side}">
      {#each stars as star, index}
        <svg
          class="star"
          class:animated={index % 6 === 0 || index === 16}
          class:spark={star.shape === "spark"}
          viewBox="0 0 10 10"
          width={star.size}
          height={star.size}
          focusable="false"
          style:left={side === "left" ? `${star.x}%` : `${100 - star.x}%`}
          style:top={`${side === "left" ? star.y : 100 - star.y}%`}
          style:--opacity={star.opacity}
          style:--dx={`${star.dx}px`}
          style:--dy={`${star.dy}px`}
          style:--duration={`${star.duration}s`}
          style:--delay={`${star.delay}s`}
          style:--scale={star.scale}
        >
          {#if star.shape === "spark"}
            <path d="M5 0 6 4 10 5 6 6 5 10 4 6 0 5 4 4Z" />
          {:else if star.shape === "diamond"}
            <path d="M5 1 8 5 5 9 2 5Z" />
          {:else}
            <circle cx="5" cy="5" r={index % 3 === 0 ? 3 : 4} />
          {/if}
        </svg>
      {/each}
    </div>
  {/each}
</div>

<style>
  .star-field {
    position: fixed;
    inset: 0;
    overflow: clip;
    pointer-events: none;
    color: var(--accent);
  }
  .star-field[hidden] { display: none; }
  .gutter {
    position: absolute;
    top: var(--top);
    height: var(--height);
    overflow: clip;
  }
  .left { left: 0; width: var(--left); }
  .right { right: 0; width: var(--right); }
  .star {
    position: absolute;
    fill: currentColor;
    opacity: var(--opacity);
    transform-origin: center;
  }
  .animated {
    animation: drift var(--duration) ease-in-out var(--delay) infinite alternate;
  }
  .paused .star { animation-play-state: paused; }
  @keyframes drift {
    to {
      transform: translate(var(--dx), var(--dy)) scale(var(--scale));
      opacity: calc(var(--opacity) * 0.65);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .star { animation: none; }
  }
  .spark { color: var(--accent-hover); }
</style>
