<!-- src/App.vue -->
<template>
  <div class="container py-5">
    <h1 class="text-center mb-4">📷 Live Camera Feed</h1>

    <div class="text-center mt-4">
      <button @click="captureImage" class="btn btn-success">
        <i class="bi bi-camera"></i> Capture
      </button>
    </div>

    <div v-if="status" class="alert alert-info mt-3 text-center">
      {{ status }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const video = ref(null)
const status = ref('')

onMounted(async () => {

  video.value.srcObject = stream
})

const captureImage = async () => {
  const canvas = document.createElement('canvas')
  canvas.width = video.value.videoWidth
  canvas.height = video.value.videoHeight
  canvas.getContext('2d').drawImage(video.value, 0, 0)
  const image = canvas.toDataURL('image/jpeg')

  status.value = '📤 Uploading image...'
  try {
    const response = await axios.post('http://localhost:8000/upload-base64', {
      object: 'live_capture',
      image: image,
    })
    status.value = `✅ 3D model uploaded! [View model](${response.data.model_url})`
  } catch (error) {
    status.value = `❌ Error: ${error.message}`
  }
}
</script>
