<script setup>
import { ref } from 'vue';
import { convertFileSrc, invoke } from "@tauri-apps/api/core";

const props = defineProps(["folder"]);
const images = ref(undefined);
const phases = ref([]);

invoke("get_phases", {}).then(x => phases.value = x);
refresh_pictures();

const hours = ref(0);
const minutes = ref(0);
const phase = ref('');

function refresh_pictures() {
    return invoke("get_pictures", {basePath: props.folder}).then(x => images.value = x);
}

async function save() {
    let filename = images.value[0];
    let m = hours.value * 60 + minutes.value;
    await invoke("label", {file: filename, phase: phase.value, minutes: m});
    await refresh_pictures();
}

</script>
<template>
    <h1>Stai etichettando le immagini in {{folder}}</h1>
    <template v-if="images !== undefined">
    <h2 v-if="images.length == 0">Non ci sono immagini da etichettare</h2>
    <section v-else>
        <img :src="convertFileSrc(images[0])">
        <form>
            <section>
                <label for="phase">Fase:</label>
                <select v-model="phase" id="phase">
                    <option v-for="p of phases" :value="p[1]">{{p[0]}}</option>
                </select>
            </section>
            <section>
                <label for="hours">Ore residue:</label>
                <input v-model="hours" type="number" id="hours" min="0" max="1" value="0">
            </section>
            <section>
                <label for="minutes">Minuti residui:</label>
                <input v-model="minutes" type="number" id="minutes" min="0" max="59" value="0">
            </section>
            <input type="button" @click="save" value="Salva">
        </form>
    </section>
    </template>
    <template v-else>
        Loading...
    </template>
</template>

<style>
img {
    max-width: 90vw;
    display: block;
    margin: auto;
}
</style>