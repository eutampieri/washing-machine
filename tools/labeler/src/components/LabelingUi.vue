<script setup>
import { computed } from 'vue';
import { invoke } from "@tauri-apps/api/core";

const props = defineProps(["folder"]);
const images = await invoke("get_pictures", {basePath: props.folder});
console.log(images);
</script>
<template>
    <h1>Stai etichettando le immagini in {{folder}}</h1>
    <h2 v-if="images.length == 0">Non ci sono immagini da etichettare</h2>
</template>