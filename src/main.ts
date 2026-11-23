import { mount } from 'svelte';
import App from './App.svelte';
import Game from './Game.svelte';
import { applyTheme, savedTheme } from './theme';
import './theme.css';
import './style.css';
import './game-tools.css';

applyTheme(savedTheme());
const target = document.getElementById('app');
const tools = new URLSearchParams(location.search);
if (target)
  mount(tools.has('engine') || tools.has('benchmark') ? App : Game, {
    target,
  });
