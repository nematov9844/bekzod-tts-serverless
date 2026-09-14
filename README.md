# Bekzod Voice 140k — RunPod Serverless TTS Engine

Ushbu endpoint **RunPod Serverless** platformasida ishga tushirilgan bo'lib, **Pay-As-You-Go** (faqat so'rov vaqtida to'lov, ishlatilmaganda **$0.00**) rejimida ishlaydi.

---

## 🚀 Endpoint Ma'lumotlari

- **Endpoint ID:** `brpr0c2l9ia4y8`
- **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)
- **Min Workers:** `0` (Avtomatik 0 ga tushadi, bo'sh turganda xarajat 0)
- **Max Workers:** `2`
- **Idle Timeout:** `120s` (2 daqiqa harakatsizlikdan so'ng to'xtaydi)
- **Sync URL (`/runsync`):** `https://api.runpod.ai/v2/brpr0c2l9ia4y8/runsync`
- **Async URL (`/run`):** `https://api.runpod.ai/v2/brpr0c2l9ia4y8/run`
- **Status URL:** `https://api.runpod.ai/v2/brpr0c2l9ia4y8/status/<JOB_ID>`

---

## ⚡ Tezkorlik Ko'rsatkichlari

| Rejim | Bajarilish vaqti | Izoh |
|---|---|---|
| **Warm Start (faol paytda)** | **0.8s – 2.0s** | Real-vaqtdan 4 barobar tezroq sintez |
| **Cold Start (0 dan yoqilganda)** | **~25s** | Model va muhit yuklangandan so'ng navbatdagi so'rovlar warm bo'ladi |

---

## 📡 API So'rov Shakli

### Headerlar
```http
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

### Request Body
```json
{
  "input": {
    "text": "Assalomu alaykum, bu RunPod Serverless orqali generatsiya qilingan ovoz.",
    "voice": "modern",
    "speed": 1.0,
    "format": "mp3"
  }
}
```

### Parametrlar:
- `text` *(string, majburiy)*: Generatsiya qilinadigan matn.
- `voice` *(string, ixtiyoriy)*: `"modern"` (tiniq, dinamik) yoki `"classic"` (chuqur bariton). Default: `"modern"`.
- `speed` *(float, ixtiyoriy)*: `0.8` dan `1.5` gacha. Default: `1.0`.
- `format` *(string, ixtiyoriy)*: `"mp3"` yoki `"wav"`. Default: `"mp3"`.

### Response Body (`/runsync`)
```json
{
  "id": "sync-85492c99-ae53-43a6-aee8-2c405ea43f68-e1",
  "status": "COMPLETED",
  "executionTime": 893,
  "delayTime": 129,
  "output": {
    "status": "COMPLETED",
    "audio_base64": "SUQzBAAAAAAA...",
    "duration": 6.04,
    "format": "mp3",
    "voice": "modern",
    "sample_rate": 24000
  }
}
```

---

## 💻 Frontend Integratsiya Misollari

### 1. JavaScript / React / Vue (`fetch`)

```javascript
async function synthesizeVoice(text, voice = "modern") {
  const RUNPOD_API_KEY = "rpa_..."; // RunPod API Key
  const ENDPOINT_ID = "brpr0c2l9ia4y8";

  const response = await fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${RUNPOD_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      input: {
        text: text,
        voice: voice,
        speed: 1.0,
        format: "mp3"
      }
    })
  });

  const data = await response.json();

  if (data.status === "COMPLETED" && data.output?.audio_base64) {
    // Brauzerda darhol ijro etish
    const audio = new Audio(`data:audio/mp3;base64,${data.output.audio_base64}`);
    audio.play();
    return data.output;
  } else {
    throw new Error(data.error || "Generatsiyada xatolik");
  }
}
```

### 2. cURL orqali sinash

```bash
curl -X POST "https://api.runpod.ai/v2/brpr0c2l9ia4y8/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Salom! Bu sinov xabari.",
      "voice": "modern",
      "format": "mp3"
    }
  }'
```

### 3. Tayyor HTML Demo
Ushbu papkadagi [`client_example.html`](file:///home/arch/Project/bekzod-tts-serverless/client_example.html) faylini istalgan brauzerda ochib, to'g'ridan-to'g'ri sinab ko'rishingiz mumkin.
