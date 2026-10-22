"use strict";
// Bounded local header guidance, never an audio decoder or security boundary.
(() => {
  const MAX_HEADER_BYTES = 65536, MAX_CHUNKS = 4096, MIB = 1024 * 1024;
  const names = {fixed:'Fixed filter', compatible:'Standard WAV', long:'Long WAV', precision:'High-resolution WAV', compressed:'MP3 / AAC M4A'};
  const ascii = (view, offset, count) => Array.from(new Uint8Array(view.buffer, view.byteOffset + offset, count), c => String.fromCharCode(c)).join('');
  const base = (status, kind, message, extra = {}) => ({schema_version:1, status, kind, message,
    header_only:true, sample_values_checked:false, ...extra});

  async function inspect(file, {isCancelled = () => false} = {}) {
    let bytes = 0, calls = 0;
    const stats = () => ({header_bytes_requested:bytes, header_read_calls:calls});
    if (isCancelled()) return base('cancelled','unknown','A newer file selection replaced this header check.',stats());
    if (!file || !Number.isSafeInteger(file.size) || file.size <= 0) return base('invalid','unknown','Choose a nonempty audio file.');
    const compressed = /\.(mp3|m4a)$/i.test(file.name);
    if (compressed) return base(file.size <= 32*MIB ? 'deferred' : 'invalid','compressed',
      file.size <= 32*MIB ? 'Compressed metadata and sample validity are checked only during local decoding. The extension is not codec verification.' : 'Compressed input exceeds 32 MiB.',
      {container_bytes:file.size, ...stats()});
    if (!/\.wav$/i.test(file.name)) return base('invalid','unknown','Choose WAV, MP3 or AAC M4A. No format is inferred from a renamed extension.');
    if (file.size > 128*MIB) return base('invalid','wav','WAV input exceeds the 128 MiB maximum.',{container_bytes:file.size});
    const begun = Date.now();
    const cancelled = () => { if (isCancelled()) throw new Error('cancelled'); };
    async function read(offset, length) {
      cancelled();
      if (Date.now()-begun > 10000 || bytes+length > MAX_HEADER_BYTES) throw new Error('budget');
      if (!Number.isSafeInteger(offset) || offset < 0 || length < 0 || offset+length > file.size) throw new Error('structure');
      bytes += length; calls += 1;
      let timer;
      let data;
      try {
        data = await Promise.race([
          file.slice(offset, offset+length).arrayBuffer(),
          new Promise((resolve,reject) => { timer=setTimeout(() => reject(new Error('budget')),Math.max(1,10000-(Date.now()-begun))); })
        ]);
      } finally { clearTimeout(timer); }
      cancelled();
      if (Date.now()-begun > 10000) throw new Error('budget');
      if (data.byteLength !== length) throw new Error('structure');
      return new DataView(data);
    }
    try {
      const riff = await read(0,12);
      if (ascii(riff,0,4) !== 'RIFF' || ascii(riff,8,4) !== 'WAVE') return base('invalid','wav','Only little-endian RIFF WAVE is supported; not RF64 or RIFX.',stats());
      const end = riff.getUint32(4,true)+8;
      if (end !== file.size) throw new Error('structure');
      let offset=12, fmt=null, data=null, fact=null, chunks=0, needsServer=false;
      while (offset < end) {
        cancelled();
        if (++chunks > MAX_CHUNKS) throw new Error('budget');
        const head=await read(offset,8), tag=ascii(head,0,4), size=head.getUint32(4,true);
        const start=offset+8, next=start+size+(size%2);
        if (!Number.isSafeInteger(next) || next>end || next<=offset) throw new Error('structure');
        if (tag==='fmt ') {
          if (fmt || size<16) throw new Error('structure');
          const v=await read(start,Math.min(size,40));
          const tagCode=v.getUint16(0,true), channels=v.getUint16(2,true), rate=v.getUint32(4,true);
          const byteRate=v.getUint32(8,true), alignment=v.getUint16(12,true), bits=v.getUint16(14,true);
          let subtype=tagCode;
          if (tagCode===65534) {
            if (size<40) throw new Error('structure');
            const extension=v.getUint16(16,true), validBits=v.getUint16(18,true), mask=v.getUint32(20,true);
            subtype=v.getUint32(24,true);
            const suffix=Array.from(new Uint8Array(v.buffer,v.byteOffset+28,12));
            if (suffix.join(',') !== '0,0,16,0,128,0,0,170,0,56,155,113') throw new Error('nonstandard');
            if (extension<22 || extension+18!==size || validBits!==bits || !(channels===1 ? [0,4] : [0,3]).includes(mask)) needsServer=true;
          } else if (!(size===16 || (size===18 && v.getUint16(16,true)===0))) needsServer=true;
          const format=subtype===1 && bits===16 ? 'PCM16' : subtype===1 && bits===24 ? 'PCM24' : subtype===3 && bits===32 ? 'Float32' : 'unsupported';
          if (!rate || !channels || !alignment || alignment!==channels*bits/8 || byteRate!==rate*alignment) throw new Error('nonstandard');
          fmt={sample_rate_hz:rate, channels, bits_per_sample:bits, sample_format:format, block_align_bytes:alignment};
        } else if (tag==='data') {
          if (!fmt || data || size%fmt.block_align_bytes) throw new Error('structure');
          const frames=size/fmt.block_align_bytes;
          data={frames,duration_seconds:frames/fmt.sample_rate_hz};
        } else if (tag==='fact') {
          if (fact!==null || size!==4) needsServer=true;
          else fact=(await read(start,4)).getUint32(0,true);
        }
        offset=next;
      }
      if (!fmt || !data) throw new Error('structure');
      if (fact!==null && fact!==data.frames) needsServer=true;
      return base('metadata','wav',needsServer
        ? 'Header metadata is available, but this layout needs final server validation. Different modes accept different RIFF extensions.'
        : 'Header metadata read. Samples, peaks, noise-only claims and audio quality have not been checked.',
        {...fmt,...data,container_bytes:file.size,needs_server_layout_check:needsServer,container_chunk_count:chunks,...stats()});
    } catch (error) {
      if (isCancelled() || error.message==='cancelled') return base('cancelled','wav','A newer file selection replaced this header check.',stats());
      return base('unchecked','wav',error.message==='budget'
        ? 'Header check reached its bounded read/time limit. No format is confirmed; final server validation is still required.'
        : 'Header structure could not be confirmed. No format is confirmed; the selected mode will perform final server validation.',
        {container_bytes:file.size,...stats()});
    }
  }

  function recommend(info, {compressedAvailable=false} = {}) {
    if (!info) return null;
    if (info.kind==='compressed') return compressedAvailable && info.status!=='invalid' ? 'compressed' : null;
    if (info.status!=='metadata' || ![44100,48000].includes(info.sample_rate_hz) || ![1,2].includes(info.channels)
      || !(info.duration_seconds>0 && info.duration_seconds<=600) || info.container_bytes>128*MIB) return null;
    if (['PCM24','Float32'].includes(info.sample_format)) return 'precision';
    if (info.sample_format!=='PCM16') return null;
    if (info.duration_seconds>60 || info.container_bytes>32*MIB) return 'long';
    return info.channels===2 || info.sample_rate_hz===44100 ? 'compatible' : 'fixed';
  }

  function assess(info, mode, capabilities = {}) {
    const suggested=recommend(info,capabilities);
    const result=(allowed,level,message) => ({allowed,level,message,recommended_mode:suggested});
    if (!['fixed','tracked','guarded','harmonics','bands','profile','compatible','long','precision','compressed','persistent'].includes(mode)) return result(false,'error','Choose a supported processing mode.');
    if (!info || info.status==='cancelled') return result(false,'checking','Read the selected file header before processing.');
    if (info.status==='invalid') return result(false,'error',info.message);
    if (info.kind==='compressed') {
      if (!capabilities.compressedAvailable) return result(false,'error','MP3 / AAC M4A requires optional PyAV 19.x. WAV remains available; nothing is installed automatically.');
      if (mode!=='compressed') return result(false,'error','Choose MP3 / AAC M4A explicitly. The file will be decoded locally and the output will be WAV.');
      return result(true,'warning',info.message);
    }
    if (mode==='persistent' && !capabilities.persistentAvailable) return result(false,'error','Persistent hum requires explicit support from this local processor. Other WAV modes remain available.');
    if (mode==='compressed') return result(false,'error','Use a WAV mode for this WAV file. No input format is switched automatically.');
    const longMode=mode==='long' || mode==='precision', nativePersistent=mode==='persistent';
    const maxBytes=(longMode || nativePersistent ? 128:32)*MIB;
    if (info.container_bytes>maxBytes) return result(false,'error',`This mode accepts up to ${longMode?128:32} MiB. Select a compatible file mode explicitly.`);
    if (info.status==='unchecked') return result(true,'warning',info.message);
    if (info.status!=='metadata') return result(false,'error','The file format is not supported.');
    const rates=longMode || nativePersistent || mode==='compatible' ? [44100,48000] : [48000];
    const layouts=longMode || nativePersistent || mode==='compatible' ? [1,2] : [1];
    const formats=mode==='precision' || nativePersistent ? ['PCM16','PCM24','Float32'] : ['PCM16'];
    const maxSeconds=longMode?600:60;
    if (!rates.includes(info.sample_rate_hz) || !layouts.includes(info.channels) || !formats.includes(info.sample_format)
      || !(info.duration_seconds>0 && info.duration_seconds<=maxSeconds)) {
      return result(false,'error',`The reported format does not fit this mode. It requires ${rates.join('/')} Hz, ${layouts.length===2?'mono/stereo':'mono'}, ${formats.join('/')} and at most ${maxSeconds} seconds.${suggested ? ` Suggested: ${names[suggested]}.` : ' No supported mode was found from this header.'}`);
    }
    return result(true,info.needs_server_layout_check?'warning':'ready',info.message);
  }
  window.HumPreflight=Object.freeze({inspect,assess,recommend,MAX_HEADER_BYTES,MAX_CHUNKS});
})();
