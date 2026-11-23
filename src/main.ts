import { mount } from 'svelte';
import App from './App.svelte';
import Game from './Game.svelte';
import { applyTheme, savedTheme } from './theme';
import './theme.css';
import './style.css';

applyTheme(savedTheme());
const fontKit = import.meta.env.VITE_ADOBE_FONTS_KIT?.trim();
if (fontKit && /^[a-z0-9]{7}$/.test(fontKit)) {
  const stylesheet = document.createElement('link');
  stylesheet.rel = 'stylesheet';
  stylesheet.href = `https://use.typekit.net/${fontKit}.css`;
  document.head.append(stylesheet);
}
const target = document.getElementById('app');
if (target)
  mount(new URLSearchParams(location.search).has('benchmark') ? App : Game, {
    target,
  });
