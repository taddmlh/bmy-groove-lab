// Per-song memory in IndexedDB (this browser on this iPad only).
//   songs: { key, name, updated, dur, bpm, patterns, state }   (small, listed in Recent)
//   files: { key, bytes, type }                                  (the original audio file)
//   stems: { key, regions: [{a, b, L, R}] }                      (separated drums, extracted bars only)
const DB = 'bmy-groove-lab', VER = 1;
let dbp = null;
function db() {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    const r = indexedDB.open(DB, VER);
    r.onupgradeneeded = () => { for (const s of ['songs', 'files', 'stems']) if (!r.result.objectStoreNames.contains(s)) r.result.createObjectStore(s, { keyPath: 'key' }); };
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
  return dbp;
}
async function tx(store, mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(store, mode), st = t.objectStore(store);
    const r = fn(st);
    t.oncomplete = () => res(r && 'result' in r ? r.result : undefined);
    t.onerror = () => rej(t.error); t.onabort = () => rej(t.error);
  });
}
export const putSong = rec => tx('songs', 'readwrite', s => s.put(rec));
export const getSong = key => tx('songs', 'readonly', s => s.get(key));
export const putFile = rec => tx('files', 'readwrite', s => s.put(rec));
export const getFile = key => tx('files', 'readonly', s => s.get(key));
export const hasFile = async key => !!(await tx('files', 'readonly', s => s.getKey(key)));
export const putStems = rec => tx('stems', 'readwrite', s => s.put(rec));
export const getStems = key => tx('stems', 'readonly', s => s.get(key));
export async function listSongs() {
  const all = await tx('songs', 'readonly', s => s.getAll());
  return (all || []).map(({ state, ...meta }) => meta).sort((a, b) => b.updated - a.updated);
}
export async function deleteSong(key) {
  for (const s of ['songs', 'files', 'stems']) await tx(s, 'readwrite', st => st.delete(key));
}
export async function askPersist() {
  try { if (navigator.storage && navigator.storage.persist) return await navigator.storage.persist(); } catch (e) {}
  return false;
}
export async function usage() {
  try { const e = await navigator.storage.estimate(); return e; } catch (e) { return null; }
}
