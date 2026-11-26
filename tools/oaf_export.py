# Magenta OaF Drums (E-GMD) onsets + velocity -> PyTorch -> ONNX. Input mel [1, T, 250] (log power, Magenta front end).
# Outputs onset probability and velocity (0..1, x127 = MIDI) for 7 drum classes, [1, T, 7].
import numpy as np, torch, torch.nn as nn, tensorflow as tf
R = tf.train.load_checkpoint('/tmp/oaf_ckpt/model.ckpt-569400'); W = lambda k: torch.tensor(R.get_tensor(k))
PITCHES = [36, 38, 48, 46, 51, 53, 49]          # kick, snare, toms, hi-hat, ride, ride bell, crash (Magenta's 8-hit map bases)
IDX = [p - 21 for p in PITCHES]

class Branch(nn.Module):
    def __init__(s, scope):
        super().__init__()
        s.convs = nn.ModuleList(); s.bns = nn.ModuleList()
        for i, (ci, co) in enumerate([(1, 16), (16, 16), (16, 32)]):
            c = nn.Conv2d(ci, co, 3, padding=1, bias=False); c.weight.data = W(f'{scope}/conv{i}/weights').permute(3, 2, 0, 1).contiguous()
            b = nn.BatchNorm2d(co, eps=1e-3)          # slim.batch_norm: center=True, scale=False, eps 0.001
            b.weight.data = torch.ones(co); b.bias.data = W(f'{scope}/conv{i}/BatchNorm/beta')
            b.running_mean.data = W(f'{scope}/conv{i}/BatchNorm/moving_mean'); b.running_var.data = W(f'{scope}/conv{i}/BatchNorm/moving_variance')
            s.convs.append(c); s.bns.append(b)
        s.fc = nn.Linear(1984, 256); s.fc.weight.data = W(f'{scope}/fc_end/weights').T.contiguous(); s.fc.bias.data = W(f'{scope}/fc_end/biases')
    def forward(s, x):                                # x [1, 1, T, 250]  (NCHW: H = time, W = mel)
        for i in range(3):
            x = torch.relu(s.bns[i](s.convs[i](x)))
            if i > 0: x = nn.functional.max_pool2d(x, (1, 2), (1, 2))
        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T, F * C)   # TF flatten of [B, T, F, C]
        return torch.relu(s.fc(x))

def tf_lstm_to_torch(lstm, scope, d, inp, u):
    k = R.get_tensor(f'{scope}/lstm/stack_bidirectional_rnn/cell_0/bidirectional_rnn/{d}/basic_lstm_cell/kernel')
    b = R.get_tensor(f'{scope}/lstm/stack_bidirectional_rnn/cell_0/bidirectional_rnn/{d}/basic_lstm_cell/bias')
    gi, gj, gf, go = [slice(n * u, (n + 1) * u) for n in range(4)]           # TF order i, j, f, o
    Wx, Wh = k[:inp], k[inp:]
    order = [gi, gf, gj, go]                                                  # torch order i, f, g, o
    sfx = '' if d == 'fw' else '_reverse'
    getattr(lstm, 'weight_ih_l0' + sfx).data = torch.tensor(np.concatenate([Wx[:, g] for g in order], 1).T.copy())
    getattr(lstm, 'weight_hh_l0' + sfx).data = torch.tensor(np.concatenate([Wh[:, g] for g in order], 1).T.copy())
    bias = np.concatenate([b[gi], b[gf] + 1.0, b[gj], b[go]])                # BasicLSTMCell forget_bias = 1.0
    getattr(lstm, 'bias_ih_l0' + sfx).data = torch.tensor(bias); getattr(lstm, 'bias_hh_l0' + sfx).data = torch.zeros(4 * u)

class OaFDrums(nn.Module):
    def __init__(s):
        super().__init__()
        s.on = Branch('onsets'); s.vel = Branch('velocity')
        s.lstm = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        tf_lstm_to_torch(s.lstm, 'onsets', 'fw', 256, 64); tf_lstm_to_torch(s.lstm, 'onsets', 'bw', 256, 64)
        s.on_fc = nn.Linear(128, 88); s.on_fc.weight.data = W('onsets/onset_logits/weights').T.contiguous(); s.on_fc.bias.data = W('onsets/onset_logits/biases')
        s.v_fc = nn.Linear(256, 88); s.v_fc.weight.data = W('velocity/onset_velocities/weights').T.contiguous(); s.v_fc.bias.data = W('velocity/onset_velocities/biases')
        s.register_buffer('idx', torch.tensor(IDX))
    def forward(s, mel):                              # mel [1, T, 250]
        x = mel.unsqueeze(1)
        h, _ = s.lstm(s.on(x))
        onset = torch.sigmoid(s.on_fc(h)).index_select(2, s.idx)
        vel = s.v_fc(s.vel(x)).index_select(2, s.idx)
        return onset, vel

m = OaFDrums().eval()
for name in ['synth', 'ewtrtw', 'groove1', 'csend']:
    mel = torch.tensor(np.load(f'/tmp/oaf/{name}.npy'))[None]
    with torch.no_grad(): o, v = m(mel)
    tp = np.load(f'/tmp/oaf/{name}_tfp.npy')[:, IDX] if name != 'synth' else None
    tv = np.load(f'/tmp/oaf/{name}_tfv.npy')[:, IDX] if name != 'synth' else None
    if tp is not None: print(name, 'torch vs TF  onset max diff', float(np.abs(o[0].numpy() - tp).max()), ' velocity max diff', float(np.abs(v[0].numpy() - tv).max()))
torch.onnx.export(m, (torch.zeros(1, 100, 250),), '/tmp/oaf/oaf_drums.onnx', input_names=['mel'], output_names=['onset', 'velocity'],
                  dynamic_axes={'mel': {1: 'T'}, 'onset': {1: 'T'}, 'velocity': {1: 'T'}}, opset_version=17, dynamo=False)
import onnxruntime as ort, os
s = ort.InferenceSession('/tmp/oaf/oaf_drums.onnx')
for name in ['ewtrtw', 'groove1', 'csend']:
    o, v = s.run(None, {'mel': np.load(f'/tmp/oaf/{name}.npy')[None]})
    tp = np.load(f'/tmp/oaf/{name}_tfp.npy')[:, IDX]; tv = np.load(f'/tmp/oaf/{name}_tfv.npy')[:, IDX]
    print(name, 'ONNX vs TF   onset max diff', float(np.abs(o[0] - tp).max()), ' velocity max diff', float(np.abs(v[0] - tv).max()))
print('onnx size', os.path.getsize('/tmp/oaf/oaf_drums.onnx') // 1024, 'KB')
