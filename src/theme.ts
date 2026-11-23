export type Theme = 'system' | 'light' | 'dark';
const key = 'choux-theme';

export function savedTheme(): Theme {
  try {
    const value = localStorage.getItem(key);
    if (value === 'light' || value === 'dark') return value;
  } catch {
    return 'system';
  }
  return 'system';
}

export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
}

export function saveTheme(theme: Theme) {
  applyTheme(theme);
  try {
    localStorage.setItem(key, theme);
  } catch {
    // The selection still applies when browser storage is unavailable.
  }
}
