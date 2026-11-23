// Separa los cuadros de un GIF de referencia y los dibuja en una grilla numerada dentro de la página.
// Uso: abrir en el navegador una página del mismo sitio que el GIF (o el GIF mismo), poner su URL en GIF_URL y
// ejecutar esto con la herramienta de JavaScript del navegador. Después, captura de pantalla y zoom.
// Ojo: las páginas de imagen (tenor, etc.) suelen servir el GIF real desde otra URL: buscala en document.images.
// Devuelve { frames, duracionSeg, msPorCuadro } para calcular el ritmo (a 30 cuadros/s del .vrma).
const GIF_URL = 'PEGAR_ACA_LA_URL_DEL_GIF';
(async () => {
  const buf = await (await fetch(GIF_URL)).arrayBuffer();
  const dec = new ImageDecoder({ data: buf, type: 'image/gif' });
  await dec.tracks.ready;
  await dec.completed; // sin esto frameCount viene incompleto
  const n = dec.tracks.selectedTrack.frameCount;
  const cols = 4, w = 200, h = 167;
  const cv = document.createElement('canvas');
  cv.width = cols * w; cv.height = Math.ceil(n / cols) * h;
  const ctx = cv.getContext('2d');
  const ms = [];
  for (let i = 0; i < n; i++) {
    const { image } = await dec.decode({ frameIndex: i });
    ms.push(Math.round(image.duration / 1000));
    ctx.drawImage(image, (i % cols) * w, Math.floor(i / cols) * h, w, h);
    ctx.fillStyle = 'red'; ctx.font = 'bold 16px sans-serif';
    ctx.fillText(String(i), (i % cols) * w + 4, Math.floor(i / cols) * h + 18);
    image.close();
  }
  document.body.innerHTML = '';
  document.body.appendChild(cv);
  return { frames: n, duracionSeg: ms.reduce((a, b) => a + b, 0) / 1000, msPorCuadro: ms };
})();
