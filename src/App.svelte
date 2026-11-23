<script lang="ts">
import EngineTools from './EngineTools.svelte';
import ThemePicker from './ThemePicker.svelte';
</script>

<svelte:head><title>Choux · Engine &amp; model</title></svelte:head>
<main>
  <header>
    <p class="wordmark">Choux</p>
    <div class="header-tools">
      <a href="./">Play chess</a>
      <ThemePicker />
    </div>
    <h1>Engine &amp; model</h1>
  </header>
  <EngineTools />
</main>
