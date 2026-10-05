import { mount } from "svelte";
import "tom-select/dist/css/tom-select.css";
import "./style.css";
import App from "./App.svelte";
mount(App, { target: document.getElementById("app")! });
