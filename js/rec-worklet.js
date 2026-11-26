// Records the input as Float32 chunks (one array per channel) and posts them to the page.
class BmyRecorder extends AudioWorkletProcessor {
  constructor() { super(); this.on = true; this.port.onmessage = e => { if (e.data === 'stop') this.on = false; }; }
  process(inputs) {
    const inp = inputs[0];
    if (this.on && inp && inp.length) this.port.postMessage(inp.map(c => c.slice(0)));
    return this.on;
  }
}
registerProcessor('bmy-recorder', BmyRecorder);
