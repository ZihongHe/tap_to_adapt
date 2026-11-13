/**
 * @file AudioWorklet processor that converts microphone samples into transferable PCM frames. The worklet runs on the audio thread and sends buffers to the main-thread transport without performing model inference.
 */

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.sourceRate = sampleRate;
    this.ratio = this.sourceRate / this.targetRate;
    this.pending = [];
    this.position = 0;
    this.output = [];
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i += 1) this.pending.push(input[i]);

    while (this.position + 1 < this.pending.length) {
      const left = Math.floor(this.position);
      const mix = this.position - left;
      const value = this.pending[left] * (1 - mix) + this.pending[left + 1] * mix;
      this.output.push(Math.max(-1, Math.min(1, value)));
      this.position += this.ratio;
      if (this.output.length >= 3200) {
        const pcm = new Int16Array(this.output.length);
        for (let i = 0; i < this.output.length; i += 1) {
          pcm[i] = Math.round(this.output[i] * 32767);
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this.output = [];
      }
    }

    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.pending.splice(0, consumed);
      this.position -= consumed;
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
