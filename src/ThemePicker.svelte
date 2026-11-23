<script lang="ts">
import { savedTheme, saveTheme, type Theme } from './theme';
</script>

<label class="theme-picker">
  Theme
  <select
    value={savedTheme()}
    onchange={(event) => saveTheme(event.currentTarget.value as Theme)}
  >
    <option value="system">System</option>
    <option value="light">Light</option>
    <option value="dark">Dark</option>
  </select>
</label>
