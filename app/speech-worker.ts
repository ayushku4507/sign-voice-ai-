import { env, pipeline } from "@huggingface/transformers";

type WorkerRequest =
  | { type: "load" }
  | { type: "transcribe"; audio: Float32Array; language: string };

type WorkerResponse =
  | { type: "progress"; status: string; progress?: number }
  | { type: "ready" }
  | { type: "result"; text: string }
  | { type: "error"; message: string };

type Transcriber = (
  audio: Float32Array,
  options: { language: string; task: "transcribe" },
) => Promise<{ text?: string }>;

let transcriberPromise: Promise<Transcriber> | null = null;

function post(message: WorkerResponse) {
  self.postMessage(message);
}

function getTranscriber() {
  if (!transcriberPromise) {
    env.allowLocalModels = false;
    transcriberPromise = pipeline(
      "automatic-speech-recognition",
      "Xenova/whisper-tiny",
      {
        progress_callback: (progress: { status?: string; progress?: number }) => {
          post({
            type: "progress",
            status: progress.status ?? "Downloading speech model",
            progress: progress.progress,
          });
        },
      },
    ) as unknown as Promise<Transcriber>;
  }
  return transcriberPromise;
}

self.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
  try {
    const request = event.data;
    const transcriber = await getTranscriber();
    if (request.type === "load") {
      post({ type: "ready" });
      return;
    }
    const result = await transcriber(request.audio, {
      language: request.language,
      task: "transcribe",
    });
    post({ type: "result", text: result.text?.trim() ?? "" });
  } catch (error) {
    transcriberPromise = null;
    post({
      type: "error",
      message: error instanceof Error ? error.message : "The speech model could not load.",
    });
  }
});
