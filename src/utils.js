const AUDIO_PATH = "./7400.low.mp3"
const MODEL_PATH = "./test.onnx"

const SAMPLE_RATE = 44100
const HOP_LENGTH = 441
const N_FFT = 1024
const N_MELS = 128
const F_MIX = 0
const F_MAX = 22050

const CLIP_LENGTH = 1000

function flatten(melSpec) {
    const flattened = []

    for (let j = 0; j < CLIP_LENGTH; j++) {
        for (let i = 0; i < N_MELS; i++) {
            flattened.push(melSpec[j][i])
        }
    }
    return flattened
}

async function LoadMp3(audio_path) {
    console.log("Loading mp3...")
    const audioBuffer = await loadAudioFromUrl(audio_path)

    console.log(audioBuffer)
    return audioBuffer
}

async function PreprocessData(audioBuffer) {
    console.log("Resampling and converting signal to mono...")
    const resampledMono = await resampleAndMakeMono(audioBuffer, 44100)

    console.log("Creating mel-spectrogram...")
    const melSpec = melSpectrogram(resampledMono, {
        sampleRate: SAMPLE_RATE,
        hopLength: HOP_LENGTH,
        // winLength?: number,
        nFft: N_FFT,
        nMels: N_MELS,
        // power?: number;
        fMin: F_MIX,
        fMax: F_MAX
    });

    console.log("Padding or Cropping...")
    const originalLength = melSpec.length
    // TODO
    // padding
    let adjustedMelSpectrogram;
    if (originalLength > CLIP_LENGTH) {
        const cropSize = originalLength - CLIP_LENGTH
        const start = Math.floor(Math.random() * cropSize)
        adjustedMelSpectrogram = melSpec.slice(start, start + CLIP_LENGTH)
    }

    console.log("Flattening...")
    const processedData = flatten(adjustedMelSpectrogram)

    console.log("Finish data preprocessing")
    console.log(processedData)
    return processedData
}

async function RunModel(processedData, modelPath) {
    console.log("Converting data to onnx tensor...")
    const inputTensor = new onnx.Tensor(processedData, 'float32', [1, 1, 128, 1000])

    console.log("Loading model...")
    let session = new onnx.InferenceSession()
    await session.loadModel(modelPath);

    console.log("Running Model...")
    let outputMap = await session.run([inputTensor])

    console.log("Result")
    let outputData = outputMap.values().next().value.data

    console.log("Top 5 classes (index)")
    const TopN = outputData.slice().sort((a, b) => b - a).slice(0, 5)
    const TopNIndex = []
    for (let i = 0; i < 5; i++) {
        TopNIndex.push(outputData.indexOf(TopN[i]))
    }
    return TopNIndex
}

async function Demo(audioPath, modelPath) {
    const audioBuffer = await LoadMp3(audioPath)
    const processedData = await PreprocessData(audioBuffer)
    const result = await RunModel(processedData, modelPath)
    console.log(result)
    return result
}

export { AUDIO_PATH, MODEL_PATH, Demo }