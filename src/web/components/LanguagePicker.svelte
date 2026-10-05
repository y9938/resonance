<script lang="ts">
  import { onMount } from "svelte";
  import TomSelect from "tom-select/base";
  import { t } from "../i18n/state.svelte";
  type Option = {
    value: string;
    label: string;
    native?: string;
    english?: string;
  };
  let {
    id,
    options,
    value,
    onchange,
    disabled = false,
    labelId,
    localePicker = false,
  }: {
    id: string;
    options: Option[];
    value: string;
    onchange: (value: string) => void;
    disabled?: boolean;
    labelId?: string;
    localePicker?: boolean;
  } = $props();
  let element: HTMLSelectElement;
  let instance = $state.raw<TomSelect | null>(null);
  function position() {
    const select = instance;
    if (!select?.isOpen) return;
    const viewport = window.visualViewport;
    const top = viewport?.offsetTop || 0,
      bottom = top + (viewport?.height || window.innerHeight),
      control = select.control.getBoundingClientRect();
    const below = Math.max(0, bottom - control.bottom - 10),
      above = Math.max(0, control.top - top - 10);
    const up =
      below < Math.min(150, select.dropdown.scrollHeight) && above > below;
    select.wrapper.classList.toggle("dropdown-up", up);
    select.dropdown_content.style.maxHeight = `${Math.max(0, Math.min(240, (up ? above : below) - 8))}px`;
  }
  onMount(() => {
    const select = new TomSelect(element, {
      valueField: "value",
      labelField: "label",
      searchField: ["label", "native", "english", "value"],
      maxItems: 1,
      maxOptions: null,
      closeAfterSelect: true,
      create: false,
      refreshThrottle: 0,
      render: {
        option: (data: Option, escape: (value: string) => string) =>
          `<div class="locale-option-content"><span class="locale-option-code">${escape(data.value.toUpperCase())}</span><span class="locale-option-label"><span class="locale-option-name">${escape(data.label)}</span><span class="locale-option-native">${escape(data.native || data.label)}${localePicker ? "" : " · " + escape(data.value)}</span></span></div>`,
        item: (data: Option, escape: (value: string) => string) =>
          `<div>${escape(localePicker ? data.value.toUpperCase() : data.label)}</div>`,
        no_results: () =>
          `<div class="locale-picker-empty">${t("localeSearchEmpty")}</div>`,
      },
      onChange: (next: string | string[]) => {
        if (next) onchange(String(next));
      },
      onDropdownOpen: position,
      onType: position,
    });
    instance = select;
    select.wrapper.classList.add(
      "language-picker",
      localePicker ? "locale-language-picker" : "stt-language-picker",
    );
    if (labelId) select.focus_node.setAttribute("aria-labelledby", labelId);
    else select.focus_node.setAttribute("aria-label", t("ariaLangGroup"));
    let wasOpen = false;
    const down = () => (wasOpen = select.isOpen);
    const click = () => {
      if (!wasOpen && !select.isOpen) select.refreshOptions(true);
      wasOpen = false;
    };
    select.control.addEventListener("pointerdown", down, true);
    select.control.addEventListener("click", click);
    window.addEventListener("resize", position);
    window.visualViewport?.addEventListener("resize", position);
    return () => {
      window.removeEventListener("resize", position);
      window.visualViewport?.removeEventListener("resize", position);
      select.control.removeEventListener("pointerdown", down, true);
      select.control.removeEventListener("click", click);
      select.destroy();
      instance = null;
    };
  });
  $effect(() => {
    const select = instance;
    if (!select) return;
    const next = options;
    const selected = value;
    select.clear(true);
    select.clearOptions();
    select.addOptions(next);
    select.setValue(selected, true);
    if (!labelId)
      select.focus_node.setAttribute("aria-label", t("ariaLangGroup"));
    select.settings.placeholder = t("languageSearchPlaceholder");
    select.inputState();
    if (disabled) select.disable();
    else select.enable();
    if (select.isOpen) {
      select.refreshOptions(false);
      position();
    }
  });
</script>

<!-- Tom Select exclusively owns the options and generated subtree inside this host. -->
<div><select {id} bind:this={element} aria-hidden="true"></select></div>
