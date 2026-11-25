// Avatar 3D (VRoid): lee la misma cara que el 2D (emoción + visema) y el estado (idle/speaking/thinking)
// y lo traduce a expresiones del VRM, más parpadeo, respiración, mirada a cámara y gestos.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRMExpression, VRMExpressionMorphTargetBind } from '@pixiv/three-vrm';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';

// Gestos animados en Blender (.vrma, ver scripts/blender/). Solo se toman estos huesos del archivo:
// el resto (respiración, el otro brazo, los dedos) sigue en código. Si el archivo no carga, queda el gesto en código
const RIGHT_FINGERS = ['Index', 'Middle', 'Ring', 'Little'].flatMap(f => ['Proximal', 'Intermediate', 'Distal'].map(j => `right${f}${j}`))
  .concat(['rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal']);
const LEFT_FINGERS = ['Index', 'Middle', 'Ring', 'Little'].flatMap(f => ['Proximal', 'Intermediate', 'Distal'].map(j => `left${f}${j}`))
  .concat(['leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal']);
const BODY = ['spine', 'chest', 'upperChest', 'neck', 'head', 'leftShoulder', 'leftUpperArm', 'leftLowerArm']; // el cuerpo acompaña
const CLIPS = {
  saludar: { file: 'avatar3d/saludo_agy.vrma', bones: ['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand', ...RIGHT_FINGERS, ...BODY, 'leftHand', 'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal', 'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal', 'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal', 'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal', 'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal'] }, // el de agy: más contenido y hacia la cámara
  despedir: { file: 'avatar3d/despedida.vrma', bones: ['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand', ...RIGHT_FINGERS, ...BODY] }, // más amplio: chau a quien se va
  tomar_te: { file: 'avatar3d/tomar_te.vrma', bones: ['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand', ...RIGHT_FINGERS, ...BODY] }, // la taza y su inclinación siguen en código (tilt)
  six_seven: { file: 'avatar3d/six_seven.vrma', bones: ['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand', ...RIGHT_FINGERS, ...BODY, 'leftHand', ...LEFT_FINGERS] },
};

const EMOTION = { happy: 'happy', sad: 'sad', angry: 'angry', surprised: 'surprised', thinking: 'relaxed' };
const VISEME = { a: ['aa', 1], ei: ['ee', 0.8], o: ['oh', 1], u: ['ou', 0.9], fv: ['ih', 0.4], l: ['ih', 0.5] };
const SHAPES = ['happy', 'happyTalk', 'sad', 'angry', 'surprised', 'relaxed', 'aa', 'ee', 'ih', 'oh', 'ou', 'blink'];

// Dedos: curl 0 = mano abierta, ~1.2 = puño. En pose T la palma mira abajo: se cierran girando en z
function hand(side, curl, thumb = curl * 0.6, fingers = ['Index', 'Middle', 'Ring', 'Little']) {
  const s = side === 'right' ? 1 : -1, out = {};
  for (const f of fingers)
    for (const j of ['Proximal', 'Intermediate', 'Distal']) out[`${side}${f}${j}`] = [0, 0, s * curl];
  for (const j of ['Proximal', 'Distal']) out[`${side}Thumb${j}`] = [0, s * -thumb, 0];
  return out;
}

// Pose de reposo (los huesos normalizados arrancan en pose T): [x, y, z] en radianes
const REST = {
  leftUpperArm: [0, 0, -1.25], rightUpperArm: [0, 0, 1.25], leftLowerArm: [0, 0, -0.15], rightLowerArm: [0, 0, 0.15],
  leftHand: [0, 0, 0], rightHand: [0, 0, 0], leftShoulder: [0, 0, 0], rightShoulder: [0, 0, 0],
  leftUpperLeg: [0, 0, 0], rightUpperLeg: [0, 0, 0], leftLowerLeg: [0, 0, 0], rightLowerLeg: [0, 0, 0],
  ...hand('left', 0.25), ...hand('right', 0.25), // relajadas, un poco cerradas
};

// Taza de té agarrada del asa: el índice pasa por el asa, el pulgar acostado encima y el resto doblado debajo
const GRIP = { ...hand('right', 1.35, 0.35), ...hand('right', 0.95, 0.35, ['Index']), rightThumbProximal: [0.7, 0, 0], rightThumbDistal: [0.6, 0, 0] };

// Gestos: pose(p) devuelve rotaciones que se mezclan con el reposo (p = segundos desde que arrancó).
// Los nombres tienen que coincidir con GESTURES de server.js
const GESTURES = {
  // Saludo con anticipación y superposición: hombro y codo suben primero, el antebrazo después y la
  // muñeca última, que además va retrasada al agitar
  saludar: {
    dur: 2.6, ownEase: true, pose: (p, rest) => {
      const lead = inOut(p, 2.6, 0), mid = inOut(p, 2.6, 0.08), late = inOut(p, 2.6, 0.16);
      const dip = p < 0.25 ? Math.sin(p / 0.25 * Math.PI) * 0.12 : 0; // baja apenas antes de subir
      const w = 11 * (p - 0.35), amp = smooth((p - 0.35) / 0.3) * late;
      // pose buscada midiendo posiciones: mano al lado de la mejilla y 5 cm adelante, codo abajo y adelante del hombro
      const up = mix(rest('rightUpperArm'), [0, 0.45, 0.6], lead);
      up[2] += dip;
      return {
        rightShoulder: mix(rest('rightShoulder'), [0, 0, -0.12], lead),
        rightUpperArm: up,
        rightLowerArm: via(rest('rightLowerArm'), [0, 1.2, -0.6], [0, -0.05, -2.2 + Math.sin(w) * 0.12 * amp], mid), // sube por adelante: directo, la mano se abría 40 cm al costado
        rightHand: mix(rest('rightHand'), [-1.2, Math.sin(w - 0.9) * 0.35 * amp, 0], late), // -1.2: palma a cámara y dedos arriba (medido)
        head: mix(rest('head'), [0.04, -0.06, 0.07], lead), // acompaña con la cabeza
        ...hand('right', 0.25 - 0.17 * late, 0.15 - 0.05 * late), // dedos sueltos, no tiesos
      };
    },
  },
  asentir: { dur: 1.3, pose: p => ({ head: [0.09 - Math.cos(p * 14) * 0.09, 0, 0] }) }, // baja y vuelve (un "sí" casi no sube la cabeza)
  negar: { dur: 1.4, pose: p => ({ head: [0, Math.sin(p * 12) * 0.3, 0] }) },
  festejar: {
    dur: 1.8, expr: 'happy', pose: p => ({
      // codos al costado y puños arriba adelante del pecho (con los brazos arriba el buzo se rompía en las axilas;
      // con los codos adelante quedaban estirados como zombi). Revisado de frente, de costado y a 3/4
      leftUpperArm: [-0.3, 0, -1.2], rightUpperArm: [-0.3, 0, 1.2],
      leftLowerArm: [0, -1.9 - Math.sin(p * 12) * 0.25, 0], rightLowerArm: [0, 1.9 + Math.sin(p * 12) * 0.25, 0],
      ...hand('left', 1.3), ...hand('right', 1.3),
    }),
  },
  // Té: levanta la taza desde abajo (brazo primero, antebrazo después), la sostiene frente al pecho, la lleva
  // a los labios con el codo adelante mientras la cabeza baja a buscarla, la va inclinando al llegar y la baja.
  // Revisado cuadro por cuadro de frente, de costado y a 3/4
  tomar_te: {
    dur: 4.6, ownEase: true, cup: true, tilt: p => band(p, 1.55, 2.75, 0.6), pose: (p, rest) => {
      const up = inOut(p, 4.6, 0), fo = inOut(p, 4.6, 0.1);
      const su = band(p, 1.3, 2.75, 0.7), sf = band(p, 1.4, 2.85, 0.7), tl = band(p, 1.55, 2.75, 0.6);
      const fingers = Object.fromEntries(Object.entries(GRIP).map(([k, v]) => [k, mix(rest(k), v, fo)]));
      return {
        rightUpperArm: mix(rest('rightUpperArm'), mix([-0.2, 0.3, 1.2], [-0.7, 0.2, 1.12], su), up),
        rightLowerArm: mix(rest('rightLowerArm'), mix([0, 1.9, 0], [0, 2.35, 0], sf), fo),
        rightShoulder: [0, 0, -0.06 * su],
        head: mix(rest('head'), [0.08 * su - 0.1 * tl, 0, rest('head')[2]], up), // x+ = baja la cabeza
        ...fingers,
      };
    },
  },
  // Se da vuelta, sale caminando por la derecha, se queda afuera un rato y vuelve
  irse_enojada: { dur: 6.2, expr: 'angry', walk: true, pose: () => ({}) },
  // Respaldo si no carga six_seven.vrma
  six_seven: {
    dur: 2.67, pose: p => {
      const s = Math.sin(p * 13);
      return {
        leftUpperArm: [-0.25, 0, -1.2], rightUpperArm: [-0.25, 0, 1.2],
        leftLowerArm: [0, -1.25 + 0.35 * s, 0], rightLowerArm: [0, 1.25 + 0.35 * s, 0],
        head: [0, 0, 0.1 * s],
      };
    },
  },
};
GESTURES.despedir = GESTURES.saludar; // respaldo si no carga despedida.vrma
const smooth = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
const mix = (a, b, k) => a.map((v, i) => v + (b[i] - v) * k);
// 0 → 1 entre a y a+r, 1 hasta b, y 1 → 0 entre b y b+r
const band = (p, a, b, r) => smooth((p - a) / r) * (1 - smooth((p - b) / r));
// de a hasta c pasando por b (un arco en vez de una línea recta)
const via = (a, b, c, k) => k < 0.5 ? mix(a, b, smooth(k * 2)) : mix(b, c, smooth(k * 2 - 1));
// sube con retraso d y baja en orden inverso (el que subió último termina de bajar último)
const inOut = (p, dur, d) => smooth((p - d) / 0.45) * smooth((dur - 0.16 + d - p) / 0.5);
// Pasos mientras se traslada: piernas alternadas, rodilla que se dobla al llevar la pierna adelante y brazos
// al revés que las piernas. En pose T x+ lleva la pierna (o el brazo colgando) hacia atrás. Sin esto se deslizaba
function walkPose(p) {
  const a = band(p, 0.6, 1.85, 0.2) + band(p, 4, 5.25, 0.2); // solo mientras camina
  if (!a) return {};
  const s = Math.sin(p * 7) * a, kneeL = Math.max(0, Math.sin(p * 7 + 1.2)) * a, kneeR = Math.max(0, -Math.sin(p * 7 + 1.2)) * a;
  return {
    leftUpperLeg: [-0.45 * s, 0, 0], rightUpperLeg: [0.45 * s, 0, 0],
    leftLowerLeg: [0.7 * kneeL, 0, 0], rightLowerLeg: [0.7 * kneeR, 0, 0],
    leftUpperArm: [0.35 * s, 0, -1.25], rightUpperArm: [-0.35 * s, 0, 1.25],
  };
}
function walkOff(p) {
  if (p < 0.6) return { turn: smooth(p / 0.6) * Math.PI / 2, x: 0 };
  if (p < 2) return { turn: Math.PI / 2, x: smooth((p - 0.6) / 1.4) * 1.2 };
  if (p < 4) return { turn: -Math.PI / 2, x: 1.2 }; // afuera: ya mira para volver
  if (p < 5.4) return { turn: -Math.PI / 2, x: (1 - smooth((p - 4) / 1.4)) * 1.2 };
  return { turn: -Math.PI / 2 * (1 - smooth((p - 5.4) / 0.6)), x: 0 };
}

export async function start(container, url, face, getState) {
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const light = new THREE.DirectionalLight(0xffffff, Math.PI);
  light.position.set(1, 1, 1);
  scene.add(light);
  const camera = new THREE.PerspectiveCamera(20, 1, 0.1, 20);

  const loader = new GLTFLoader();
  loader.register(parser => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(vrm); // VRM 0.x mira para el otro lado; en 1.0 no hace nada
  vrm.scene.traverse(o => { o.frustumCulled = false; });
  scene.add(vrm.scene);

  // Alegría = sonrisa tranquila, como en el 2D: la cara feliz del modelo (Fcl_ALL_Joy) es una carcajada con ojos
  // cerrados y boca abierta (no es NYRA, y sumada a la boca al hablar la abría de más). Ojos apenas entrecerrados,
  // cejas felices y sonrisa cerrada; al hablar, la boca la abren los visemas
  const joyTalk = new VRMExpression('happyTalk');
  let joyBinds = 0;
  vrm.scene.traverse(o => {
    const d = o.morphTargetDictionary;
    if (!o.isMesh || d?.Fcl_EYE_Joy === undefined) return;
    for (const [n, w] of [['Fcl_EYE_Joy', 0.3], ['Fcl_BRW_Joy', 0.6], ['Fcl_MTH_Fun', 0.7]]) {
      if (d[n] === undefined) continue;
      joyTalk.addBind(new VRMExpressionMorphTargetBind({ primitives: [o], index: d[n], weight: w }));
      joyBinds++;
    }
  });
  vrm.scene.add(joyTalk);
  vrm.expressionManager.registerExpression(joyTalk);

  const clipLoader = new GLTFLoader();
  clipLoader.register(parser => new VRMAnimationLoaderPlugin(parser));
  for (const [name, { file, bones }] of Object.entries(CLIPS)) {
    clipLoader.loadAsync(file).then(c => {
      const clip = createVRMAnimationClip(c.userData.vrmAnimations[0], vrm);
      const nodes = new Map(bones.map(b => [vrm.humanoid.getNormalizedBoneNode(b)?.name, vrm.humanoid.getNormalizedBoneNode(b)]));
      const tracks = clip.tracks.filter(t => t.name.endsWith('.quaternion') && nodes.has(t.name.split('.')[0]))
        .map(t => ({ node: nodes.get(t.name.split('.')[0]), interp: t.createInterpolant() }));
      const q = new THREE.Quaternion();
      GESTURES[name] = { ...GESTURES[name], dur: clip.duration, clip: (t, k) => {
        for (const { node, interp } of tracks) node.quaternion.slerp(q.fromArray(interp.evaluate(Math.min(t, clip.duration))), k);
      } };
    }).catch(err => console.warn(`No cargó ${file}, sigue el gesto en código:`, err));
  }
  const baseTurn = vrm.scene.rotation.y;

  const bone = name => vrm.humanoid.getNormalizedBoneNode(name);
  for (const [name, r] of Object.entries(REST)) bone(name)?.rotation.set(...r); // un modelo sin algún dedo no rompe

  // Taza: "Tea Set 01" de Poly Haven (CC0), solo la taza. Va en la escena, no en la mano: cada cuadro se
  // ubica con el asa entre los dedos y se orienta sola (derecha al sostenerla, hacia los labios en el sorbo)
  const cup = new THREE.Group();
  cup.visible = false;
  cup.scale.setScalar(0.8); // la original mide 10,6 cm: grande para la cara de NYRA
  scene.add(cup);
  loader.loadAsync('avatar3d/taza.glb').then(t => {
    const tea = new THREE.Mesh(new THREE.CircleGeometry(0.046, 32), new THREE.MeshStandardMaterial({ color: 0x8a4b1e, roughness: 0.2 }));
    tea.rotation.x = -Math.PI / 2;
    tea.position.y = 0.058;
    // con la luz de un solo lado la cerámica quedaba negra de perfil: un poco de luz propia
    t.scene.traverse(o => { if (o.isMesh && o.material.map) { o.material.emissive.set(0xffffff); o.material.emissiveMap = o.material.map; o.material.emissiveIntensity = 0.35; } });
    cup.add(t.scene, tea);
  }).catch(err => console.warn('No cargó la taza:', err));
  // el asa se engancha en la segunda falange del índice (antes quedaba apoyada en los nudillos o encima del puño)
  const raw = name => vrm.humanoid.getRawBoneNode(name);
  const pinch = [raw('rightIndexIntermediate'), raw('rightIndexDistal')].filter(Boolean);
  const pinchB = new THREE.Vector3();
  const HANDLE = new THREE.Vector3(0, 0.048, -0.066).multiplyScalar(0.8); // centro del hueco del asa (medido: y 0,030-0,067, z -0,056/-0,080)
  const cupDir = new THREE.Vector3(), cupQ = new THREE.Quaternion(), gripPos = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);
  const HANDLE_OUT = new THREE.Quaternion().setFromAxisAngle(UP, Math.PI / 2); // asa (-Z) hacia la mano, afuera a la derecha
  const handF = [raw('rightHand'), raw('rightIndexProximal'), raw('rightLittleProximal')].filter(Boolean);
  // Encuadre de busto, como el sprite 2D
  vrm.update(0);
  const head = new THREE.Vector3();
  bone('head').getWorldPosition(head);
  const aim = new THREE.Vector3(0, head.y - 0.18, 0);
  camera.position.set(0, aim.y + 0.03, 2.4);
  camera.lookAt(aim);
  vrm.lookAt.target = camera;

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = container;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();

  const weights = Object.fromEntries(SHAPES.map(s => [s, 0]));
  let nextBlink = 2, blinkUntil = 0;
  let current = null; // { g, t0 }
  const clock = new THREE.Clock();

  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1), t = clock.elapsedTime;
    if (!document.body.classList.contains('mode3d')) return; // en 2D no gasta GPU
    const state = getState();
    const still = document.body.classList.contains('no-float');

    const p = current ? t - current.t0 : 0;
    if (current && p > current.g.dur) current = null;
    const g = current?.g;
    // ownEase: el gesto hace su propia entrada/salida por hueso (uno detrás del otro); si no, todo junto
    const env = g ? (g.ownEase ? 1 : Math.max(0, Math.min(1, p / 0.3, (g.dur - p) / 0.3))) : 0;

    // Expresiones: suavizadas para que no salten de un frame al otro
    const target = Object.fromEntries(SHAPES.map(s => [s, 0]));
    const emo = g?.expr || EMOTION[face.emo];
    const happyShape = emo === 'happy' && joyBinds ? 'happyTalk' : emo; // un modelo sin esas formas sigue con happy
    if (emo) target[happyShape] = face.emo === 'thinking' && !g?.expr ? 0.4 : 1;
    const v = VISEME[face.mouth];
    if (v) target[v[0]] = v[1];
    if (t > nextBlink) { blinkUntil = t + 0.13; nextBlink = t + 2.8 + Math.random() * 3.2; }
    // la cara feliz ya cierra los ojos: parpadear encima los deforma
    target.blink = t < blinkUntil && weights.happy < 0.3 ? 1 : 0; // la sonrisa (happyTalk) deja los ojos abiertos: parpadea
    for (const s of SHAPES) {
      const speed = s === 'blink' ? 40 : s.length === 2 ? 25 : 8; // la boca (aa, ee...) rápida, las emociones lentas
      weights[s] += (target[s] - weights[s]) * Math.min(1, dt * speed);
      vrm.expressionManager.setValue(s, weights[s]);
    }

    // Cuerpo: respirar, balancearse al hablar, ladear la cabeza al pensar
    const k = still ? 0 : 1;
    bone('spine').rotation.x = k * Math.sin(t * 1.6) * 0.015;
    bone('chest').rotation.z = k * Math.sin(t * 0.7) * 0.012;
    const talk = state.includes('speaking'), think = state.includes('thinking');
    bone('neck').rotation.x = k * (talk ? Math.sin(t * 9) * 0.025 : 0);
    const tilt = think ? 0.14 : k * Math.sin(t * 0.5) * 0.03;
    const headRest = bone('head').userData.rest ??= [0, 0, 0];
    headRest[2] += (tilt - headRest[2]) * Math.min(1, dt * 4);

    // Gesto encima del reposo
    const base = { ...REST, head: headRest };
    const pose = g && !g.clip ? { ...g.pose(p, name => base[name]), ...(g.walk ? walkPose(p) : {}) } : {};
    for (const [name, rest] of Object.entries(base)) {
      const to = pose[name] || rest;
      bone(name)?.rotation.set(...rest.map((r, i) => r + (to[i] - r) * env));
    }
    if (g?.clip) g.clip(p, Math.min(1, p / 0.2, (g.dur - p) / 0.2)); // entra y sale suave sobre la pose en código
    cup.visible = !!g?.cup && env > 0.5;
    const walk = g?.walk ? walkOff(p) : { turn: 0, x: 0 };
    vrm.scene.rotation.y = baseTurn + walk.turn;
    vrm.scene.position.x = walk.x;
    vrm.scene.position.y = g?.walk ? Math.abs(Math.sin(p * 7)) * 0.015 * (band(p, 0.6, 1.85, 0.2) + band(p, 4, 5.25, 0.2)) : 0; // sube y baja con cada paso

    vrm.update(dt);
    if (cup.visible && g.clip && handF[0]) {
      // Con animación de Blender la taza va fija a la mano: la inclinan el brazo y la muñeca, no la taza sola.
      // Como en el agarre real: el eje de la taza sigue la fila de nudillos (meñique → índice) y el asa queda del lado
      // de la palma (scripts/blender/tomar_te.py hace la misma cuenta para llevar el borde a la boca)
      const [h, i1, l1] = handF.map(b => b.getWorldPosition(new THREE.Vector3()));
      const cz = i1.clone().sub(h).cross(l1.clone().sub(h)).normalize(); // palma; el asa está en -Z de la taza
      const cy = i1.sub(l1); cy.sub(cz.clone().multiplyScalar(cy.dot(cz))).normalize();
      cupQ.setFromRotationMatrix(new THREE.Matrix4().makeBasis(cy.clone().cross(cz), cy, cz));
      cup.quaternion.copy(cupQ);
      gripPos.set(0, 0, 0);
      for (const b of pinch) gripPos.add(b.getWorldPosition(pinchB));
      cup.position.copy(gripPos.divideScalar(pinch.length)).sub(HANDLE.clone().applyQuaternion(cupQ));
    } else if (cup.visible) {
      const k = g.tilt?.(p) || 0;
      cupDir.set(0, 1 - 0.55 * k, -0.9 * k).normalize(); // el modelo mira a +Z: -Z es hacia su cara
      cupQ.setFromUnitVectors(UP, cupDir).multiply(HANDLE_OUT).premultiply(vrm.scene.quaternion);
      cup.quaternion.copy(cupQ);
      gripPos.set(0, 0, 0);
      for (const b of pinch) { b.updateWorldMatrix(true, false); gripPos.add(b.getWorldPosition(pinchB)); }
      cup.position.copy(gripPos.divideScalar(pinch.length)).sub(HANDLE.clone().applyQuaternion(cupQ));
    }
    renderer.render(scene, camera);
  });

  return {
    gestures: Object.keys(GESTURES),
    gesture(name) { if (GESTURES[name]) current = { g: GESTURES[name], t0: clock.elapsedTime }; },
  };
}
