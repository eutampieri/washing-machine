<script setup>
import {open} from "@tauri-apps/plugin-dialog";

const emit = defineEmits(["folderSelected"]);

async function selectFolder() {
    const folder = await open({
        multiple: false,
        directory: true,
    });
    emit("folderSelected", folder);
}
</script>
<template>
    <h1>Scegli una cartella</h1>
    <button @click="selectFolder">Apri</button>
</template>