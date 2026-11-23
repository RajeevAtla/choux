import { mount } from 'svelte';
import App from './App.svelte';
import Game from './Game.svelte';
import { applyTheme, savedTheme } from './theme';
import './theme.css';
import './style.css';

applyTheme(savedTheme());
const target = document.getElementById('app');
if (target)
  mount(new URLSearchParams(location.search).has('benchmark') ? App : Game, {
    target,
  });
